"""
serving/tests/test_feature_hydration.py
Automated Unit & Invariant Tests for Online Feature Hydration Layer.

Verifies:
- Gate B: Hydration pipeline, schema normalization, point-in-time correctness
- Hard Hydration Gate: rejection of incomplete/unknown events without silent imputation
- Leakage Gate: Zero future lookahead, zero target leakage (isFraud NOT in X)
- Feature Dimensionality & Ordering: Exactly 406 canonical features
- Gate C: Prediction parity between offline inference and hydrated serving
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add project root and serving directory to sys.path
SERVING_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SERVING_DIR.parent
for p in [str(REPO_ROOT), str(SERVING_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from serving.hydration.adapter import OnlineFeatureHydrationAdapter
from serving.hydration.entity_store import EntityProfileStore, EntityProfile
from serving.hydration.provenance import (
    classify_feature,
    get_provenance_manifest,
    TIER_DIRECT_EVENT,
    TIER_DERIVED_EVENT,
    TIER_ENTITY_PROFILE,
    TIER_HISTORICAL_AGG,
    TIER_UNRECONSTRUCTABLE,
)
from serving.hydration.schemas import StreamingTransactionPayload, VelocityMetrics
from serving.inference.offline_inference import (
    E1_DECISION_THRESHOLD,
    OfflineInferenceEngine,
)
from serving.preprocessing.serving_wrapper import ServingPreprocessor


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def seeded_adapter() -> OnlineFeatureHydrationAdapter:
    """Fixture providing a seeded Feature Hydration Adapter."""
    store = EntityProfileStore()
    store.seed_mock_profiles()
    return OnlineFeatureHydrationAdapter(entity_store=store)


@pytest.fixture
def inference_engine() -> OfflineInferenceEngine:
    """Fixture providing the canonical E1 Offline Inference Engine."""
    return OfflineInferenceEngine(threshold=E1_DECISION_THRESHOLD)


# ── Test Suite ────────────────────────────────────────────────────────────────

def test_hydration_complete_event_scoring(seeded_adapter, inference_engine):
    """
    Test 1: Complete valid event for registered card passes the hard gate
    and scores successfully with E1.
    """
    payload = {
        "TransactionID": "TX-1001",
        "card_id": "CARD-13926",
        "TransactionAmt": 125.50,
        "TransactionDT": 13400000.0,
        "ProductCD": "W",
    }
    result = seeded_adapter.hydrate(payload)

    # 1. Verify hydration gate passed
    assert result.complete is True
    assert result.point_in_time_valid is True
    assert result.rejection_reason is None
    assert result.dataframe is not None
    assert len(result.dataframe) == 1

    # 2. Verify essential fields populated
    df = result.dataframe
    assert df.iloc[0]["card1"] == 13926.0
    assert df.iloc[0]["card4"] == "discover"
    assert df.iloc[0]["card6"] == "credit"
    assert df.iloc[0]["TransactionAmt"] == 125.50
    assert df.iloc[0]["TransactionDT"] == 13400000.0

    # 3. Score via inference engine
    score_res = seeded_adapter.score_or_reject(payload, inference_engine)
    assert score_res["scoreable"] is True
    assert score_res["scored"] is True
    assert 0.0 <= score_res["fraud_probability"] <= 1.0
    assert score_res["decision"] in ("FRAUD", "LEGIT")
    print("[PASS] test_hydration_complete_event_scoring")


def test_hydration_gate_rejects_missing_event_field(seeded_adapter):
    """
    Test 2: Event missing essential fields (e.g. negative or missing amount)
    is blocked by the hard gate.
    """
    invalid_payload = {
        "TransactionID": "TX-1002",
        "card_id": "CARD-13926",
        "TransactionDT": 13400000.0,
        # TransactionAmt is missing!
    }
    result = seeded_adapter.hydrate(invalid_payload)

    assert result.complete is False
    assert result.dataframe is None
    assert "TransactionAmt" in str(result.rejection_reason)
    print("[PASS] test_hydration_gate_rejects_missing_event_field")


def test_hydration_gate_rejects_unknown_card(seeded_adapter, inference_engine):
    """
    Test 3: Unregistered card fails profile lookup and is REJECTED by gate
    rather than silently scoring with synthetic median fallbacks.
    """
    unknown_payload = {
        "TransactionID": "TX-1003",
        "card_id": "CARD-UNKNOWN-99999",
        "TransactionAmt": 50.0,
        "TransactionDT": 13400000.0,
        "ProductCD": "W",
    }
    result = seeded_adapter.hydrate(unknown_payload)

    assert result.complete is False
    assert result.dataframe is None
    assert "Unknown card_id" in str(result.rejection_reason)

    score_res = seeded_adapter.score_or_reject(unknown_payload, inference_engine)
    assert score_res["scoreable"] is False
    assert score_res["scored"] is False
    assert score_res["error"] == "REJECTED_BY_HYDRATION_GATE"
    print("[PASS] test_hydration_gate_rejects_unknown_card")


def test_zero_future_lookahead_leakage():
    """
    Test 4: Strict causality invariant — history queries retrieve ONLY
    records where timestamp < current_dt. Future transactions are excluded.
    """
    store = EntityProfileStore()
    store.seed_mock_profiles()

    card_id = "CARD-13926"
    # Seed historical records: 2 past, 1 present (equal), 1 future
    store.record_transaction(card_id=card_id, timestamp=100.0, amount=10.0, transaction_id="PAST-1")
    store.record_transaction(card_id=card_id, timestamp=200.0, amount=20.0, transaction_id="PAST-2")
    store.record_transaction(card_id=card_id, timestamp=300.0, amount=30.0, transaction_id="EQUAL")
    store.record_transaction(card_id=card_id, timestamp=400.0, amount=40.0, transaction_id="FUTURE")

    # Current transaction is at timestamp 300.0
    history = store.get_causal_history(card_id, current_dt=300.0)

    # Invariant: ONLY PAST-1 (100) and PAST-2 (200) may be returned!
    history_ids = [h.transaction_id for h in history]
    assert history_ids == ["PAST-1", "PAST-2"]
    assert "EQUAL" not in history_ids
    assert "FUTURE" not in history_ids
    assert all(h.timestamp < 300.0 for h in history)
    print("[PASS] test_zero_future_lookahead_leakage")


def test_history_updated_post_scoring_strictly(seeded_adapter, inference_engine):
    """
    Test 5: History for an entity is updated strictly AFTER scoring.
    """
    card_id = "CARD-13926"
    initial_history_len = len(seeded_adapter.entity_store.get_causal_history(card_id, current_dt=20000000.0))

    payload = {
        "TransactionID": "TX-NEW-01",
        "card_id": card_id,
        "TransactionAmt": 75.0,
        "TransactionDT": 13500000.0,
        "ProductCD": "W",
    }
    # During hydration, history count is unchanged
    res = seeded_adapter.hydrate(payload)
    assert len(seeded_adapter.entity_store.get_causal_history(card_id, current_dt=20000000.0)) == initial_history_len

    # After scoring via score_or_reject, history is updated
    score_res = seeded_adapter.score_or_reject(payload, inference_engine)
    assert score_res["scored"] is True
    post_history = seeded_adapter.entity_store.get_causal_history(card_id, current_dt=20000000.0)
    assert len(post_history) == initial_history_len + 1
    assert post_history[-1].transaction_id == "TX-NEW-01"
    print("[PASS] test_history_updated_post_scoring_strictly")


def test_zero_target_leakage_in_hydration(seeded_adapter):
    """
    Test 6: isFraud and is_fraud are strictly forbidden and never present in hydrated output.
    """
    payload_with_leak = {
        "TransactionID": "TX-LEAK",
        "card_id": "CARD-13926",
        "TransactionAmt": 50.0,
        "TransactionDT": 13400000.0,
        "ProductCD": "W",
        "isFraud": 1,
        "is_fraud": 1,
        "fraud_bool": 1,
    }
    res = seeded_adapter.hydrate(payload_with_leak)
    assert res.complete is True
    df = res.dataframe
    assert "isFraud" not in df.columns
    assert "is_fraud" not in df.columns
    assert "fraud_bool" not in df.columns
    print("[PASS] test_zero_target_leakage_in_hydration")


def test_feature_provenance_manifest():
    """
    Test 7: All 5 provenance tiers are populated and accurate.
    """
    manifest = get_provenance_manifest()
    assert TIER_DIRECT_EVENT in manifest
    assert TIER_DERIVED_EVENT in manifest
    assert TIER_ENTITY_PROFILE in manifest
    assert TIER_HISTORICAL_AGG in manifest
    assert TIER_UNRECONSTRUCTABLE in manifest

    assert classify_feature("TransactionAmt") == TIER_DIRECT_EVENT
    assert classify_feature("hour_sin") == TIER_DERIVED_EVENT
    assert classify_feature("card1") == TIER_ENTITY_PROFILE
    assert classify_feature("C1") == TIER_HISTORICAL_AGG
    assert classify_feature("D1") == TIER_HISTORICAL_AGG
    assert classify_feature("isFraud") == TIER_UNRECONSTRUCTABLE
    print("[PASS] test_feature_provenance_manifest")


def test_hydrated_dataframe_to_preprocessor(seeded_adapter):
    """
    Test 8: Hydrated DataFrame transforms cleanly through ServingPreprocessor
    yielding exactly (1, 406) features with zero target leakage.
    """
    payload = {
        "TransactionID": "TX-PREP-01",
        "card_id": "CARD-13926",
        "TransactionAmt": 100.0,
        "TransactionDT": 13400000.0,
        "ProductCD": "W",
    }
    res = seeded_adapter.hydrate(payload)
    assert res.complete is True

    sp = ServingPreprocessor()
    X, meta = sp.transform(res.dataframe)

    assert X.shape == (1, 406)
    assert list(X.columns) == sp.expected_feature_names
    assert "isFraud" not in X.columns
    assert "TransactionID" not in X.columns
    print("[PASS] test_hydrated_dataframe_to_preprocessor")


def test_velocity_metrics_integration(seeded_adapter):
    """
    Test 9: Flink real-time velocity metrics are properly absorbed into behavioral state.
    """
    payload = {
        "TransactionID": "TX-VELOCITY",
        "card_id": "CARD-13926",
        "TransactionAmt": 300.0,
        "TransactionDT": 13450000.0,
        "ProductCD": "W",
        "velocity": {
            "count_5m": 7,
            "amount_5m": 850.0,
            "velocity_ratio": 2.33,
        },
    }
    res = seeded_adapter.hydrate(payload)
    assert res.complete is True
    # C1 should reflect the max of history count and Flink 5m burst count
    assert res.dataframe.iloc[0]["C1"] >= 7.0
    print("[PASS] test_velocity_metrics_integration")


def test_e1_parity_on_complete_transaction(seeded_adapter, inference_engine):
    """
    Test 10 (Gate C): Hydrated pipeline execution produces deterministic
    probabilities identical to direct offline inference.
    """
    payload = {
        "TransactionID": "3544193",
        "card_id": "CARD-3544193",
        "TransactionAmt": 100.0,
        "TransactionDT": 13400000.0,
        "ProductCD": "W",
    }
    # Direct offline inference on hydrated frame
    res = seeded_adapter.hydrate(payload)
    assert res.complete is True

    df1, _ = inference_engine.predict_transaction(res.dataframe.copy())
    prob1 = float(df1.iloc[0]["fraud_probability"])

    # Inference via score_or_reject
    score_res = seeded_adapter.score_or_reject(payload, inference_engine)
    prob2 = float(score_res["fraud_probability"])

    diff = abs(prob1 - prob2)
    assert diff < 1e-9, f"Probability mismatch: {prob1} vs {prob2} (diff={diff})"
    assert score_res["decision"] == df1.iloc[0]["decision"]
    print(f"[PASS] test_e1_parity_on_complete_transaction (diff = {diff:.10f})")


# ── Standalone CLI Runner ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING FEATURE HYDRATION & LEAKAGE TEST SUITE (GATES B & C)")
    print("=" * 70)

    store = EntityProfileStore()
    store.seed_mock_profiles()
    adapter = OnlineFeatureHydrationAdapter(entity_store=store)
    engine = OfflineInferenceEngine(threshold=E1_DECISION_THRESHOLD)

    test_hydration_complete_event_scoring(adapter, engine)
    test_hydration_gate_rejects_missing_event_field(adapter)
    test_hydration_gate_rejects_unknown_card(adapter, engine)
    test_zero_future_lookahead_leakage()
    test_history_updated_post_scoring_strictly(adapter, engine)
    test_zero_target_leakage_in_hydration(adapter)
    test_feature_provenance_manifest()
    test_hydrated_dataframe_to_preprocessor(adapter)
    test_velocity_metrics_integration(adapter)
    test_e1_parity_on_complete_transaction(adapter, engine)

    print("=" * 70)
    print("ALL 10 FEATURE HYDRATION TESTS PASSED (10/10)!")
    print("=" * 70)
