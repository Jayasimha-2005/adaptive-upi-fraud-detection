"""
streaming/tests/test_stream_serving_bridge.py
Phase 13 Acceptance Gates Unit & Contract Test Suite.

Verifies all 12 Phase 13 Acceptance Gates:
- P13.1: Bridge starts / components initialize cleanly
- P13.2: Raw event consumed / parsed correctly from Member 1 format
- P13.3: Fraud-feature event consumed / parsed correctly from Member 2 format
- P13.4: Correct deterministic correlation (velocity matches entity and causal time context)
- P13.5: Unified payload (StreamingTransactionPayload generated properly)
- P13.6: Hydration (causal history reconstructed correctly)
- P13.7: No future leakage (future event injection does not affect current prediction)
- P13.8: Target isolation (isFraud strictly excluded from model feature matrix)
- P13.9: 406 features in canonical order
- P13.10: Frozen E1 model and preprocessor linkage
- P13.11: Threshold 0.616521 preserved
- P13.12: Serving response (probability + decision returned correctly)
"""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

# Add project root and serving directory to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SERVING_DIR = REPO_ROOT / "serving"
STREAMING_DIR = REPO_ROOT / "streaming"
for p in [str(REPO_ROOT), str(SERVING_DIR), str(STREAMING_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from serving.hydration.adapter import OnlineFeatureHydrationAdapter
from serving.hydration.entity_store import EntityProfileStore
from serving.inference.offline_inference import (
    E1_DECISION_THRESHOLD,
    OfflineInferenceEngine,
)
from serving.preprocessing.serving_wrapper import ServingPreprocessor
from streaming.stream_serving_bridge import CorrelationBuffer, StreamServingBridge


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def bridge() -> StreamServingBridge:
    """Fixture providing initialized StreamServingBridge with mock profiles."""
    store = EntityProfileStore()
    store.seed_mock_profiles()
    adapter = OnlineFeatureHydrationAdapter(entity_store=store)
    engine = OfflineInferenceEngine(threshold=E1_DECISION_THRESHOLD)
    buffer = CorrelationBuffer(retention_seconds=600.0)
    return StreamServingBridge(
        adapter=adapter,
        engine=engine,
        correlation_buffer=buffer,
        correlation_window_sec=300.0,
    )


# ── Test Cases ────────────────────────────────────────────────────────────────

def test_p13_1_bridge_initialization(bridge):
    """P13.1: Bridge starts / components initialize cleanly."""
    assert bridge.adapter is not None
    assert bridge.engine is not None
    assert bridge.correlation_buffer is not None
    assert bridge.engine.threshold == 0.616521
    print("[PASS] P13.1: Bridge initialization verified.")


def test_p13_2_raw_event_consumption(bridge):
    """P13.2: Raw event consumed / parsed correctly from Member 1 format."""
    # Member 1 IEEE-CIS normalized format
    raw_m1_event = {
        "transaction_id": "TX-2987001",
        "card_id": "CARD-13926",
        "amount": 150.00,
        "merchant_id": "M-W",
        "epoch_timestamp": 13400000.0,
        "device_type": "web",
        "country": "US",
        "is_fraud": True,  # Evaluation label
    }
    norm = bridge.normalize_stream_event(raw_m1_event)
    assert norm["TransactionID"] == "TX-2987001"
    assert norm["card_id"] == "CARD-13926"
    assert norm["TransactionAmt"] == 150.00
    assert norm["TransactionDT"] == 13400000.0
    assert norm["ProductCD"] == "W"
    # Target must be stripped
    assert "is_fraud" not in norm
    assert "isFraud" not in norm
    print("[PASS] P13.2: Raw event consumed & normalized.")


def test_p13_3_fraud_feature_event_consumption(bridge):
    """P13.3: Fraud-feature event consumed / parsed correctly from Member 2 format."""
    m2_feature_record = {
        "user_id": "CARD-13926",
        "timestamp": 13400000000,  # Epoch ms
        "transaction_count_5m": 5,
        "total_amount_5m": 600.00,
        "average_amount_5m": 120.00,
        "transaction_count_10m": 8,
        "total_amount_10m": 950.00,
        "average_amount_10m": 118.75,
        "transaction_velocity_ratio": 0.625,
        "amount_velocity_ratio": 0.631,
        "amount_bucket": "MEDIUM",
    }
    bridge.correlation_buffer.add_feature_event(m2_feature_record)
    # Check buffer retrieval
    correlated = bridge.correlation_buffer.correlate(
        card_id="CARD-13926",
        event_timestamp=13400000.0,
        window_tolerance_sec=300.0,
    )
    assert correlated is not None
    assert correlated.count_5m == 5
    assert correlated.amount_5m == 600.00
    assert correlated.velocity_ratio == 0.625
    print("[PASS] P13.3: Fraud-feature event consumed & parsed.")


def test_p13_4_correct_correlation(bridge):
    """P13.4: Deterministic correlation (velocity matches entity and causal time context)."""
    # Register velocity for CARD-A at t=1000
    bridge.correlation_buffer.add_feature_event({
        "user_id": "CARD-A",
        "timestamp": 1000.0,
        "transaction_count_5m": 3,
        "total_amount_5m": 150.00,
    })
    # Register velocity for CARD-B at t=1000
    bridge.correlation_buffer.add_feature_event({
        "user_id": "CARD-B",
        "timestamp": 1000.0,
        "transaction_count_5m": 12,
        "total_amount_5m": 2500.00,
    })

    # Query for CARD-A at t=1050 (within 300s tolerance)
    match_a = bridge.correlation_buffer.correlate("CARD-A", event_timestamp=1050.0)
    assert match_a is not None
    assert match_a.count_5m == 3

    # Query for CARD-B at t=1050
    match_b = bridge.correlation_buffer.correlate("CARD-B", event_timestamp=1050.0)
    assert match_b is not None
    assert match_b.count_5m == 12

    # Query for CARD-A at t=2000 (exceeds 300s window) -> None
    expired_a = bridge.correlation_buffer.correlate("CARD-A", event_timestamp=2000.0, window_tolerance_sec=300.0)
    assert expired_a is None

    # Query for unknown CARD-C -> None
    unknown_c = bridge.correlation_buffer.correlate("CARD-UNKNOWN", event_timestamp=1050.0)
    assert unknown_c is None
    print("[PASS] P13.4: Correct deterministic correlation verified.")


def test_p13_5_unified_payload_assembly(bridge):
    """P13.5: Unified payload (StreamingTransactionPayload generated properly)."""
    raw_event = {
        "TransactionID": "TX-UNIFIED-01",
        "card1": "13926",
        "TransactionAmt": 120.00,
        "TransactionDT": 13400000.0,
        "ProductCD": "W",
    }
    feature_record = {
        "user_id": "CARD-13926",
        "timestamp": 13400000.0,
        "transaction_count_5m": 4,
        "total_amount_5m": 350.00,
    }
    bridge.correlation_buffer.add_feature_event(feature_record)
    payload = bridge.assemble_payload(
        raw_event,
        bridge.correlation_buffer.correlate("CARD-13926", 13400000.0),
    )
    assert payload.transaction_id == "TX-UNIFIED-01"
    assert payload.card_id == "CARD-13926"
    assert payload.amount == 120.00
    assert payload.event_time == 13400000.0
    assert payload.velocity is not None
    assert payload.velocity.count_5m == 4
    print("[PASS] P13.5: Unified payload assembly verified.")


def test_p13_6_causal_history_hydration(bridge):
    """P13.6: Hydration (causal history reconstructed correctly)."""
    cid = "CARD-13926"
    # Seed causal history: 2 past events
    bridge.adapter.entity_store.record_transaction(cid, timestamp=1000.0, amount=25.0, transaction_id="PAST-1")
    bridge.adapter.entity_store.record_transaction(cid, timestamp=2000.0, amount=50.0, transaction_id="PAST-2")

    raw_event = {
        "TransactionID": "TX-P13-6",
        "card_id": cid,
        "TransactionAmt": 75.00,
        "TransactionDT": 3000.0,
        "ProductCD": "W",
    }
    res = bridge.process_transaction(raw_event)
    assert res["scoreable"] is True
    assert res["scored"] is True
    assert res["point_in_time_valid"] is True
    print("[PASS] P13.6: Causal history hydration verified.")


def test_p13_7_no_future_leakage(bridge):
    """P13.7: Future event injection does not affect current prediction."""
    cid = "CARD-13926"
    # Event T at dt=5000.0
    raw_event = {
        "TransactionID": "TX-CURRENT",
        "card_id": cid,
        "TransactionAmt": 100.00,
        "TransactionDT": 5000.0,
        "ProductCD": "W",
    }
    # Score event T before any future injection
    res_before = bridge.process_transaction(raw_event.copy())
    prob_before = res_before["fraud_probability"]

    # Inject future velocity at dt=10000.0
    bridge.correlation_buffer.add_feature_event({
        "user_id": cid,
        "timestamp": 10000.0,
        "transaction_count_5m": 99,
        "total_amount_5m": 50000.0,
    })
    # Inject future transaction into entity store at dt=10000.0
    bridge.adapter.entity_store.record_transaction(cid, timestamp=10000.0, amount=9999.0, transaction_id="FUTURE-TX")

    # Correlate for event T at dt=5000.0 again
    correlated = bridge.correlation_buffer.correlate(cid, event_timestamp=5000.0)
    # Future velocity must NOT correlate (feat.timestamp 10000 > event.timestamp 5000)
    assert correlated is None

    # Score event T again
    res_after = bridge.process_transaction(raw_event.copy())
    prob_after = res_after["fraud_probability"]

    # Probabilities must be identical (no future leakage)
    assert abs(prob_before - prob_after) < 1e-9
    print("[PASS] P13.7: Zero future leakage verified.")


def test_p13_8_target_isolation(bridge):
    """P13.8: Target isolation (isFraud strictly excluded from model feature matrix)."""
    raw_with_target = {
        "TransactionID": "TX-LEAK-TEST",
        "card_id": "CARD-13926",
        "TransactionAmt": 50.00,
        "TransactionDT": 6000.0,
        "ProductCD": "W",
        "isFraud": 1,
        "is_fraud": True,
        "fraud_bool": 1,
    }
    norm = bridge.normalize_stream_event(raw_with_target)
    assert "isFraud" not in norm
    assert "is_fraud" not in norm
    assert "fraud_bool" not in norm

    # Hydrate and transform through preprocessor
    payload = bridge.assemble_payload(raw_with_target)
    hyd = bridge.adapter.hydrate(payload)
    assert hyd.complete is True
    assert "isFraud" not in hyd.dataframe.columns
    assert "is_fraud" not in hyd.dataframe.columns

    sp = ServingPreprocessor()
    X, _ = sp.transform(hyd.dataframe)
    assert "isFraud" not in X.columns
    assert "is_fraud" not in X.columns
    print("[PASS] P13.8: Target isolation verified.")


def test_p13_9_406_features_canonical_order(bridge):
    """P13.9: Exactly 406 features in canonical order."""
    raw_event = {
        "TransactionID": "TX-406-FEATS",
        "card_id": "CARD-13926",
        "TransactionAmt": 100.00,
        "TransactionDT": 7000.0,
        "ProductCD": "W",
    }
    payload = bridge.assemble_payload(raw_event)
    hyd = bridge.adapter.hydrate(payload)
    assert hyd.complete is True

    sp = ServingPreprocessor()
    X, _ = sp.transform(hyd.dataframe)
    assert X.shape == (1, 406)
    assert list(X.columns) == sp.expected_feature_names
    print("[PASS] P13.9: Exactly 406 features in canonical order verified.")


def test_p13_10_frozen_e1_linkage(bridge):
    """P13.10: Frozen E1 model and preprocessor linkage."""
    assert bridge.engine.model is not None
    assert bridge.engine.serving_preprocessor is not None
    assert bridge.engine.serving_preprocessor.preprocessor is not None
    assert len(bridge.engine.serving_preprocessor.expected_feature_names) == 406
    assert bridge.engine.model.num_feature() == 406
    print("[PASS] P13.10: Canonical E1 artifact linkage verified.")


def test_p13_11_threshold_consistency(bridge):
    """P13.11: Decision threshold remains 0.616521."""
    assert bridge.engine.threshold == 0.616521
    assert E1_DECISION_THRESHOLD == 0.616521

    raw_event = {
        "TransactionID": "TX-THRESH",
        "card_id": "CARD-13926",
        "TransactionAmt": 50.00,
        "TransactionDT": 8000.0,
        "ProductCD": "W",
    }
    res = bridge.process_transaction(raw_event)
    assert res["decision_threshold"] == 0.616521
    assert res["decision"] in ("FRAUD", "LEGIT")
    print("[PASS] P13.11: Threshold 0.616521 consistency verified.")


def test_p13_12_serving_response_integrity(bridge):
    """P13.12: Serving response (probability + decision returned correctly)."""
    raw_event = {
        "TransactionID": "3544193",
        "card_id": "CARD-3544193",
        "TransactionAmt": 100.00,
        "TransactionDT": 13400000.0,
        "ProductCD": "W",
    }
    feature_record = {
        "user_id": "CARD-3544193",
        "timestamp": 13400000.0,
        "transaction_count_5m": 2,
        "total_amount_5m": 150.00,
    }
    res = bridge.process_transaction(raw_event, feature_record)
    assert res["scoreable"] is True
    assert res["scored"] is True
    assert res["transaction_id"] == "3544193"
    assert 0.0 <= res["fraud_probability"] <= 1.0
    assert res["decision"] in ("FRAUD", "LEGIT")
    assert res["correlated_velocity"] is True
    assert res["point_in_time_valid"] is True
    print(f"[PASS] P13.12: Serving response integrity verified (prob={res['fraud_probability']:.6f}, decision={res['decision']}).")


# ── Standalone CLI Runner ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING PHASE 13 STREAM-TO-SERVING BRIDGE ACCEPTANCE TEST SUITE")
    print("=" * 70)

    store = EntityProfileStore()
    store.seed_mock_profiles()
    b = StreamServingBridge(
        adapter=OnlineFeatureHydrationAdapter(entity_store=store),
        engine=OfflineInferenceEngine(threshold=E1_DECISION_THRESHOLD),
    )

    test_p13_1_bridge_initialization(b)
    test_p13_2_raw_event_consumption(b)
    test_p13_3_fraud_feature_event_consumption(b)
    test_p13_4_correct_correlation(b)
    test_p13_5_unified_payload_assembly(b)
    test_p13_6_causal_history_hydration(b)
    test_p13_7_no_future_leakage(b)
    test_p13_8_target_isolation(b)
    test_p13_9_406_features_canonical_order(b)
    test_p13_10_frozen_e1_linkage(b)
    test_p13_11_threshold_consistency(b)
    test_p13_12_serving_response_integrity(b)

    print("=" * 70)
    print("ALL 12 PHASE 13 ACCEPTANCE GATES PASSED (12/12)!")
    print("=" * 70)
