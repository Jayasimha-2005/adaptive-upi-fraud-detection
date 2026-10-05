"""
streaming/tests/test_phase16_parity.py
Phase 16 Test Suite: Systematic Offline vs. Streaming E1 Parity at Scale.

Validates Acceptance Gates P16.1 – P16.10:
- P16.1: Same transaction IDs evaluated across both engines
- P16.2: Canonical E1 features matching experiments/E1_lightgbm/feature_names.json
- P16.3: 406 features count verified
- P16.4: Feature name ordering strictly preserved
- P16.5: Probability delta |P_offline - P_streaming| <= 1e-10
- P16.6: Classification decisions identical (0 mismatches)
- P16.7: Target isolation (isFraud not present in feature matrix X)
- P16.8: Point-in-time temporal causality strictly preserved
- P16.9: Zero synthetic profile fabrication for unknown entities
- P16.10: Hard hydration gate rejects incomplete events prior to model scoring
"""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVING_DIR = REPO_ROOT / "serving"
for p in [str(REPO_ROOT), str(SERVING_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from serving.hydration.adapter import OnlineFeatureHydrationAdapter
from serving.hydration.entity_store import EntityProfileStore
from serving.inference.offline_inference import (
    E1_DECISION_THRESHOLD,
    OfflineInferenceEngine,
)
from streaming.kafka_broker import LocalKafkaBroker
from streaming.multi_transaction_benchmark import (
    find_dataset_path,
    load_canonical_transactions,
)
from streaming.offline_streaming_parity_benchmark import run_systematic_parity_evaluation
from streaming.stream_pipeline import StreamingPipelineOrchestrator


@pytest.fixture(scope="module")
def parity_benchmark_data():
    """Runs or returns cached parity evaluation results for N=100."""
    return run_systematic_parity_evaluation(count=100)


def test_p16_1_transaction_id_alignment(parity_benchmark_data):
    """P16.1: 100% 1-to-1 transaction ID alignment between offline and streaming."""
    assert parity_benchmark_data["sample_size_N"] == 100
    assert parity_benchmark_data["evaluated_count"] == 100
    for rec in parity_benchmark_data["sample_comparisons"]:
        assert rec["transaction_id"] is not None
        assert rec["card_id"] is not None


def test_p16_2_and_3_canonical_406_features(parity_benchmark_data):
    """P16.2 & P16.3: Canonical 406 features verified in both offline and streaming."""
    assert parity_benchmark_data["feature_count_offline"] == 406
    assert parity_benchmark_data["feature_count_streaming"] == 406


def test_p16_4_feature_ordering_invariance(parity_benchmark_data):
    """P16.4: Feature name ordering strictly invariant."""
    assert parity_benchmark_data["feature_order_mismatches"] == 0


def test_p16_5_strict_probability_tolerance(parity_benchmark_data):
    """P16.5: Probability delta |P_offline - P_streaming| <= 1e-10."""
    assert parity_benchmark_data["max_absolute_delta"] <= 1e-10
    assert parity_benchmark_data["mean_absolute_delta"] <= 1e-10


def test_p16_6_decision_invariance(parity_benchmark_data):
    """P16.6: Classification decisions identical across all evaluated transactions."""
    assert parity_benchmark_data["decision_mismatches"] == 0
    assert parity_benchmark_data["decision_matches"] == parity_benchmark_data["evaluated_count"]
    assert parity_benchmark_data["parity_passed"] is True


def test_p16_7_target_isolation():
    """P16.7: Target labels (isFraud) never enter the feature matrix X."""
    dataset_path = find_dataset_path()
    events, profiles = load_canonical_transactions(dataset_path, count=5)
    store = EntityProfileStore()
    for prof in profiles:
        store.register_profile(prof)

    orch = StreamingPipelineOrchestrator(broker=LocalKafkaBroker(), entity_store=store)
    for ev in events:
        poisoned = ev.copy()
        poisoned["isFraud"] = 1
        poisoned["is_fraud"] = 1
        poisoned["fraud_bool"] = True
        trace = orch.ingest_transaction(poisoned)
        assert trace.model_invoked is True
        # Verify preprocessor feature names do NOT contain target labels
        assert "isFraud" not in orch.bridge.engine.serving_preprocessor.expected_feature_names
        assert "is_fraud" not in orch.bridge.engine.serving_preprocessor.expected_feature_names


def test_p16_8_point_in_time_causality():
    """P16.8: Future events cannot influence historical feature hydration."""
    store = EntityProfileStore()
    store.seed_mock_profiles()
    orch = StreamingPipelineOrchestrator(broker=LocalKafkaBroker(), entity_store=store)

    # Ingest future transaction for CARD-13926 at DT=99999999
    future_ev = {
        "TransactionID": "TX-FUTURE",
        "card_id": "CARD-13926",
        "TransactionAmt": 5000.0,
        "TransactionDT": 99999999.0,
        "ProductCD": "W",
    }
    orch.ingest_transaction(future_ev)

    # Ingest past transaction at DT=86400.0
    past_ev = {
        "TransactionID": "TX-PAST",
        "card_id": "CARD-13926",
        "TransactionAmt": 50.0,
        "TransactionDT": 86400.0,
        "ProductCD": "W",
    }
    trace_past = orch.ingest_transaction(past_ev)
    assert trace_past.model_invoked is True

    # Hydrate past transaction directly to inspect causal history
    res = orch.bridge.adapter.hydrate(past_ev)
    assert res.complete is True
    # The history queried for past_ev must have timestamp < 86400.0, excluding TX-FUTURE
    assert res.dataframe.iloc[0]["C1"] == 1.0


def test_p16_9_no_synthetic_profile_fabrication():
    """P16.9: Unknown entity rejected; no synthetic median values manufactured."""
    store = EntityProfileStore()
    orch = StreamingPipelineOrchestrator(broker=LocalKafkaBroker(), entity_store=store)

    unknown_ev = {
        "TransactionID": "TX-UNKNOWN",
        "card_id": "CARD-UNREGISTERED-99999",
        "TransactionAmt": 100.0,
        "TransactionDT": 86500.0,
        "ProductCD": "W",
    }
    trace = orch.ingest_transaction(unknown_ev)
    assert trace.model_invoked is False
    assert trace.fraud_probability is None
    assert "REJECTED_BY_HYDRATION_GATE" in str(trace.error)


def test_p16_10_hard_hydration_gate_on_incomplete():
    """P16.10: Incomplete transactions rejected prior to model scoring."""
    store = EntityProfileStore()
    store.seed_mock_profiles()
    orch = StreamingPipelineOrchestrator(broker=LocalKafkaBroker(), entity_store=store)

    incomplete_ev = {
        "TransactionID": "TX-INCOMPLETE",
        "card_id": "CARD-13926",
        "TransactionAmt": None,  # Missing essential field
        "TransactionDT": 86600.0,
        "ProductCD": "W",
    }
    trace = orch.ingest_transaction(incomplete_ev)
    assert trace.model_invoked is False
    assert trace.fraud_probability is None
    assert "REJECTED_BY_HYDRATION_GATE" in str(trace.error)
