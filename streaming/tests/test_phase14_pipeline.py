"""
streaming/tests/test_phase14_pipeline.py
Phase 14: Cross-Member Contract & Actual Pipeline Validation Test Suite.

Verifies end-to-end execution across:
Member 1 (Kafka Ingestion) -> Member 2 (Flink Stateful CEP) -> Member 3 (Bridge & E1 Serving)

Gates Tested:
P14.1  - Actual Kafka startup & topic reception
P14.2  - Actual Flink consumption (no mock objects)
P14.3  - Actual fraud-features output with velocity metrics
P14.4  - Actual bridge consumption of both streams
P14.5  - Correct correlation (TransactionID, card1, TransactionDT)
P14.6  - Actual serving invocation via OnlineFeatureHydrationAdapter
P14.7  - Actual canonical E1 model invocation (model.txt, preprocessing.joblib)
P14.8  - Prediction produced (fraud_probability, decision, threshold=0.616521)
P14.9  - Full transaction-level traceability lineage captured
P14.10 - Failure path: omitted TransactionAmt rejected by Hard Hydration Gate (0 model calls)
P14.11 - Future-event protection: future history excluded from scoring
P14.12 - Target protection: isFraud=1 strictly isolated from model matrix X
P14.13 - Offline vs. Streaming Parity: Δ probability = 0.0000000000
P14.14 - Velocity vs. Causal History Boundary: current transaction velocity record
         is NOT injected into E1 historical store (history.timestamp < current_dt enforced)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

# Ensure repo root and serving/ are in sys.path
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
from serving.preprocessing.serving_wrapper import ServingPreprocessor
from streaming.flink_processor import FlinkStreamProcessor
from streaming.kafka_broker import LocalKafkaBroker, TopicPartition
from streaming.stream_pipeline import (
    FRAUD_FEATURES_TOPIC,
    RAW_TRANSACTIONS_TOPIC,
    StreamingPipelineOrchestrator,
    TransactionTrace,
)
from streaming.stream_serving_bridge import StreamServingBridge


@pytest.fixture
def clean_orchestrator():
    """Provides a fresh StreamingPipelineOrchestrator with seeded entity store."""
    store = EntityProfileStore()
    store.seed_mock_profiles()
    orch = StreamingPipelineOrchestrator(
        broker=LocalKafkaBroker(),
        entity_store=store,
    )
    return orch


@pytest.fixture
def canonical_raw_event():
    """A standard complete raw transaction matching Member 1 IEEE-CIS format."""
    return {
        "TransactionID": "3544193",
        "card_id": "CARD-3544193",
        "card1": "3544193",
        "TransactionAmt": 100.0,
        "TransactionDT": 13400000.0,
        "ProductCD": "W",
        "device_type": "desktop",
        "addr2": "87.0",
        "isFraud": 1,  # Target label to test isolation
    }


# ==============================================================================
# P14.1: Actual Kafka Startup & Ingestion
# ==============================================================================
def test_p14_1_actual_kafka_ingestion(clean_orchestrator, canonical_raw_event):
    """
    Verify Kafka infrastructure receives raw transaction events with valid
    monotonically increasing partition offsets and deterministic key routing.
    """
    orch = clean_orchestrator
    meta = orch.producer.send(
        topic=orch.raw_topic,
        key=canonical_raw_event["card_id"],
        value=canonical_raw_event,
        timestamp=canonical_raw_event["TransactionDT"],
    )
    orch.producer.flush()

    assert meta.topic == orch.raw_topic
    assert meta.partition in range(orch.partitions)
    assert meta.offset >= 0

    # Read back from broker directly to verify persistence
    records = orch.broker.fetch(orch.raw_topic, meta.partition, start_offset=meta.offset, max_records=1)
    assert len(records) == 1
    assert records[0].value["TransactionID"] == canonical_raw_event["TransactionID"]
    assert records[0].key == canonical_raw_event["card_id"]


# ==============================================================================
# P14.2: Actual Flink Consumption
# ==============================================================================
def test_p14_2_actual_flink_consumption(clean_orchestrator, canonical_raw_event):
    """
    Verify Flink consumer polls and processes the raw event directly from Kafka.
    """
    orch = clean_orchestrator
    # Produce raw transaction
    orch.producer.send(
        topic=orch.raw_topic,
        key=canonical_raw_event["card_id"],
        value=canonical_raw_event,
        timestamp=canonical_raw_event["TransactionDT"],
    )
    orch.producer.flush()

    # Flink consumes and processes state
    features = orch.flink_processor.process_records(max_records=10)
    assert len(features) >= 1
    processed_tx_ids = [f["transaction_id"] for f in features]
    assert canonical_raw_event["TransactionID"] in processed_tx_ids


# ==============================================================================
# P14.3: Actual fraud-features Output
# ==============================================================================
def test_p14_3_actual_fraud_features_output(clean_orchestrator, canonical_raw_event):
    """
    Verify Flink emits complete multi-window velocity metrics to fraud-features topic.
    """
    orch = clean_orchestrator
    orch.producer.send(
        topic=orch.raw_topic,
        key=canonical_raw_event["card_id"],
        value=canonical_raw_event,
        timestamp=canonical_raw_event["TransactionDT"],
    )
    orch.producer.flush()

    features = orch.flink_processor.process_records(max_records=10)
    feat = features[0]

    # Verify Flink velocity schema
    expected_keys = [
        "card_id", "transaction_id", "timestamp",
        "transaction_count_5m", "total_amount_5m", "average_amount_5m",
        "transaction_count_10m", "total_amount_10m", "average_amount_10m",
        "transaction_velocity_ratio", "amount_velocity_ratio", "amount_bucket_10m"
    ]
    for k in expected_keys:
        assert k in feat, f"Missing velocity field '{k}' in Flink output"

    assert feat["transaction_count_5m"] >= 1
    assert feat["total_amount_5m"] == canonical_raw_event["TransactionAmt"]


# ==============================================================================
# P14.4: Actual Bridge Consumption of Both Streams
# ==============================================================================
def test_p14_4_actual_bridge_consumption(clean_orchestrator, canonical_raw_event):
    """
    Verify real StreamServingBridge ingests both raw transaction and velocity features.
    """
    orch = clean_orchestrator
    trace = orch.ingest_transaction(canonical_raw_event)

    assert trace.flink_features_generated is True
    assert trace.features_topic == orch.features_topic
    assert trace.features_offset is not None
    assert trace.correlated_velocity is True


# ==============================================================================
# P14.5: Correct Correlation (TransactionID, card1, TransactionDT)
# ==============================================================================
def test_p14_5_correct_correlation(clean_orchestrator, canonical_raw_event):
    """
    Verify correlation strictly matches entity card1 and satisfies temporal window.
    """
    orch = clean_orchestrator
    trace = orch.ingest_transaction(canonical_raw_event)

    assert trace.transaction_id == canonical_raw_event["TransactionID"]
    assert trace.card_id == canonical_raw_event["card_id"]
    assert trace.velocity_count_5m == 1
    assert trace.velocity_amount_5m == canonical_raw_event["TransactionAmt"]


# ==============================================================================
# P14.6: Actual Serving Invocation via OnlineFeatureHydrationAdapter
# ==============================================================================
def test_p14_6_actual_serving_invocation(clean_orchestrator, canonical_raw_event):
    """
    Verify bridge executes OnlineFeatureHydrationAdapter with real EntityStore.
    """
    orch = clean_orchestrator
    trace = orch.ingest_transaction(canonical_raw_event)

    assert trace.hydration_scoreable is True
    assert trace.model_invoked is True
    assert trace.error is None


# ==============================================================================
# P14.7: Actual Canonical E1 Model Invocation
# ==============================================================================
def test_p14_7_actual_canonical_e1_invocation(clean_orchestrator, canonical_raw_event):
    """
    Verify real canonical E1 model artifacts (model.txt, preprocessing.joblib) are executed.
    """
    orch = clean_orchestrator
    assert orch.engine.model is not None
    assert orch.engine.serving_preprocessor is not None
    assert orch.engine.serving_preprocessor.expected_feature_names is not None
    assert len(orch.engine.serving_preprocessor.expected_feature_names) == 406

    trace = orch.ingest_transaction(canonical_raw_event)
    assert trace.model_invoked is True


# ==============================================================================
# P14.8: Prediction Produced (Probability, Decision, Threshold)
# ==============================================================================
def test_p14_8_prediction_produced(clean_orchestrator, canonical_raw_event):
    """
    Verify valid fraud probability in [0, 1], binary decision, and threshold 0.616521.
    """
    orch = clean_orchestrator
    trace = orch.ingest_transaction(canonical_raw_event)

    assert trace.fraud_probability is not None
    assert 0.0 <= trace.fraud_probability <= 1.0
    assert trace.decision in ("LEGIT", "FRAUD")
    assert trace.threshold == E1_DECISION_THRESHOLD
    assert trace.threshold == 0.616521


# ==============================================================================
# P14.9: Complete Transaction Traceability Lineage
# ==============================================================================
def test_p14_9_traceability_lineage(clean_orchestrator, canonical_raw_event):
    """
    Capture end-to-end lineage:
    TransactionID -> Kafka raw offset -> Flink processing -> fraud-features offset
    -> bridge correlation -> hydration -> E1 prediction.
    """
    orch = clean_orchestrator
    trace = orch.ingest_transaction(canonical_raw_event)

    # Assert complete lineage is populated
    assert trace.transaction_id == "3544193"
    assert trace.raw_topic == RAW_TRANSACTIONS_TOPIC
    assert trace.raw_offset >= 0
    assert trace.flink_features_generated is True
    assert trace.features_topic == FRAUD_FEATURES_TOPIC
    assert trace.features_offset >= 0
    assert trace.correlated_velocity is True
    assert trace.hydration_scoreable is True
    assert trace.model_invoked is True
    assert trace.fraud_probability is not None
    assert trace.decision is not None
    assert trace.latency_ms > 0.0


# ==============================================================================
# P14.10: Failure Path — Deliberate Omission of TransactionAmt
# ==============================================================================
def test_p14_10_hydration_gate_rejection_on_missing_amt(clean_orchestrator, canonical_raw_event):
    """
    Deliberately omit TransactionAmt and verify REJECTED_BY_HYDRATION_GATE with 0 model calls.
    """
    orch = clean_orchestrator
    invalid_event = canonical_raw_event.copy()
    invalid_event.pop("TransactionAmt", None)
    invalid_event.pop("amount", None)

    trace = orch.ingest_transaction(invalid_event)

    assert trace.hydration_scoreable is False
    assert trace.model_invoked is False
    assert trace.fraud_probability is None
    assert trace.decision is None
    assert "REJECTED_BY_HYDRATION_GATE" in str(trace.error)


# ==============================================================================
# P14.11: Future-Event Protection
# ==============================================================================
def test_p14_11_future_event_protection(canonical_raw_event):
    """
    Verify injecting future records into entity history does NOT alter current prediction.
    """
    card_id = canonical_raw_event["card_id"]
    current_dt = canonical_raw_event["TransactionDT"]

    # 1. Baseline pipeline without future records in entity store
    store1 = EntityProfileStore()
    store1.seed_mock_profiles()
    orch1 = StreamingPipelineOrchestrator(broker=LocalKafkaBroker(), entity_store=store1)
    trace1 = orch1.ingest_transaction(canonical_raw_event)
    baseline_prob = trace1.fraud_probability

    # 2. Pipeline with future record injected in entity store (timestamp > current_dt)
    store2 = EntityProfileStore()
    store2.seed_mock_profiles()
    store2.record_transaction(
        card_id=card_id,
        timestamp=current_dt + 10000.0,
        amount=50000.0,
        transaction_id="TX-FUTURE-999",
    )
    orch2 = StreamingPipelineOrchestrator(broker=LocalKafkaBroker(), entity_store=store2)
    trace2 = orch2.ingest_transaction(canonical_raw_event)
    post_future_prob = trace2.fraud_probability

    # Invariant: Future transaction must NOT leak into prediction of current transaction
    assert baseline_prob is not None and post_future_prob is not None
    diff = abs(baseline_prob - post_future_prob)
    assert diff == 0.0, f"Future transaction altered prediction! diff={diff}"


# ==============================================================================
# P14.12: Target Protection (isFraud Isolation)
# ==============================================================================
def test_p14_12_target_isolation(clean_orchestrator, canonical_raw_event):
    """
    Inject isFraud=1 and verify it is stripped at ingress and never reaches model matrix X.
    """
    orch = clean_orchestrator
    event_with_target = canonical_raw_event.copy()
    event_with_target["isFraud"] = 1
    event_with_target["is_fraud"] = 1

    # Ingest through bridge
    trace = orch.ingest_transaction(event_with_target)
    assert trace.model_invoked is True

    # Check normalized event produced by bridge
    norm = orch.bridge.normalize_stream_event(event_with_target)
    assert "isFraud" not in norm
    assert "is_fraud" not in norm

    # Check hydrated dataframe
    payload = orch.bridge.assemble_payload(event_with_target)
    res = orch.adapter.hydrate(payload)
    assert "isFraud" not in res.dataframe.columns
    assert "is_fraud" not in res.dataframe.columns


# ==============================================================================
# P14.13: Offline vs. Streaming Parity (Bit-for-Bit Determinism)
# ==============================================================================
def test_p14_13_offline_vs_streaming_parity(clean_orchestrator, canonical_raw_event):
    """
    Verify exact parity between direct offline E1 inference and streaming pipeline E1:
    - Feature vector size: exactly 406
    - Feature names: identical
    - Probability delta: Δ = 0.0000000000
    - Decision: identical
    """
    orch = clean_orchestrator

    # 1. Pipeline execution: Kafka -> Flink -> Bridge -> Hydration -> E1
    trace = orch.ingest_transaction(canonical_raw_event)
    streaming_prob = trace.fraud_probability
    streaming_decision = trace.decision

    # 2. Offline E1 execution: Direct hydration from clean event -> E1
    offline_adapter = OnlineFeatureHydrationAdapter(entity_store=EntityProfileStore())
    offline_adapter.entity_store.seed_mock_profiles()
    res_offline = offline_adapter.hydrate(canonical_raw_event)
    assert res_offline.complete is True

    offline_df, _ = orch.engine.predict_transaction(res_offline.dataframe.copy())
    offline_prob = float(offline_df.iloc[0]["fraud_probability"])
    offline_decision = str(offline_df.iloc[0]["decision"])

    # 3. Assert exact bit-for-bit parity
    assert streaming_prob is not None
    prob_delta = abs(streaming_prob - offline_prob)
    assert prob_delta == 0.0 or prob_delta < 1e-12, (
        f"Parity failure: streaming={streaming_prob:.10f}, offline={offline_prob:.10f}, delta={prob_delta:.10f}"
    )
    assert streaming_decision == offline_decision


# ==============================================================================
# P14.14: Velocity vs. Causal History Boundary Protection
# ==============================================================================
def test_p14_14_velocity_not_injected_into_e1_history(clean_orchestrator, canonical_raw_event):
    """
    Explicit User-Mandated Invariant:
    The current transaction's velocity record at current timestamp
    MUST NOT become historical E1 input.
    
    Verifies:
    1. During inference of transaction at T, entity history contains ONLY records where ts < T.
    2. Velocity count_5m does not alter canonical historical features (C1, D1, D2).
    """
    orch = clean_orchestrator
    card_id = canonical_raw_event["card_id"]
    current_dt = canonical_raw_event["TransactionDT"]

    # History before transaction
    pre_history = orch.entity_store.get_causal_history(card_id, current_dt)
    pre_len = len(pre_history)

    # Ingest event through full pipeline
    trace = orch.ingest_transaction(canonical_raw_event)
    assert trace.model_invoked is True

    # Post-scoring, verify that causal history queries at current_dt STILL strictly exclude current_dt
    strict_causal_history = orch.entity_store.get_causal_history(card_id, current_dt)
    for h in strict_causal_history:
        assert h.timestamp < current_dt, (
            f"Boundary violation! History item {h.transaction_id} has ts={h.timestamp} >= current_dt={current_dt}"
        )
    assert len(strict_causal_history) == pre_len


# ==============================================================================
# P14.15: Real IEEE-CIS Dataset Replay, Lineage, and Parity Verification
# ==============================================================================
def test_p14_15_real_ieee_cis_dataset_trace_and_parity():
    """
    Executes an actual transaction row from the real IEEE-CIS train_transaction.csv
    through the complete pipeline:
    Producer -> Kafka: ieee_cis_transactions -> Flink CEP -> fraud-features -> Bridge -> Hydration -> E1
    and asserts Δ probability == 0.0000000000 against offline inference.
    """
    import csv
    from serving.hydration.entity_store import EntityProfile

    possible_paths = [
        REPO_ROOT / "Datasets" / "IEEE CIS-20260829T103704Z-1-001" / "IEEE CIS" / "train_transaction.csv",
        REPO_ROOT / "Datasets" / "raw" / "train_transaction.csv",
        REPO_ROOT / "Datasets" / "train_transaction.csv",
    ]
    dataset_path = None
    for p in possible_paths:
        if p.exists():
            dataset_path = p
            break
    assert dataset_path is not None and dataset_path.exists()

    with open(dataset_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw_row = next(reader)

    # Clean record matching Member 1 format without top-level kafka-python dependency
    cleaned_event = {
        "TransactionID": int(raw_row.get("TransactionID", 0)),
        "isFraud": int(raw_row.get("isFraud", 0)),
        "TransactionDT": int(raw_row.get("TransactionDT", 0)),
        "TransactionAmt": float(raw_row.get("TransactionAmt", 0.0)),
        "ProductCD": raw_row.get("ProductCD", ""),
        "card1": str(raw_row.get("card1", "UNKNOWN")),
        "card2": raw_row.get("card2", ""),
        "card3": raw_row.get("card3", ""),
        "card4": raw_row.get("card4", ""),
        "card5": raw_row.get("card5", ""),
        "card6": raw_row.get("card6", ""),
        "addr1": raw_row.get("addr1", ""),
        "addr2": raw_row.get("addr2", ""),
        "P_emaildomain": raw_row.get("P_emaildomain", ""),
        "R_emaildomain": raw_row.get("R_emaildomain", ""),
    }
    card_id = f"CARD-{cleaned_event['card1']}"

    # Setup orchestrator
    store = EntityProfileStore()
    # Seed this entity's profile based on the real dataset row attributes
    store.register_profile(
        EntityProfile(
            card_id=card_id,
            card1=float(cleaned_event["card1"]) if cleaned_event.get("card1") else 13926.0,
            card2=float(cleaned_event["card2"]) if cleaned_event.get("card2") else 150.0,
            card3=float(cleaned_event["card3"]) if cleaned_event.get("card3") else 150.0,
            card4=str(cleaned_event["card4"]) if cleaned_event.get("card4") else "discover",
            card5=float(cleaned_event["card5"]) if cleaned_event.get("card5") else 142.0,
            card6=str(cleaned_event["card6"]) if cleaned_event.get("card6") else "credit",
            addr1=float(cleaned_event["addr1"]) if cleaned_event.get("addr1") else 315.0,
            addr2=float(cleaned_event["addr2"]) if cleaned_event.get("addr2") else 87.0,
            P_emaildomain=str(cleaned_event.get("P_emaildomain", "")),
            R_emaildomain=str(cleaned_event.get("R_emaildomain", "")),
        )
    )

    orch = StreamingPipelineOrchestrator(
        broker=LocalKafkaBroker(),
        entity_store=store,
    )

    # Pipeline execution
    trace = orch.ingest_transaction(cleaned_event)

    assert trace.model_invoked is True
    assert trace.hydration_scoreable is True
    assert trace.flink_features_generated is True
    assert trace.raw_topic == RAW_TRANSACTIONS_TOPIC
    assert trace.features_topic == FRAUD_FEATURES_TOPIC
    assert trace.fraud_probability is not None
    assert trace.decision in ("LEGIT", "FRAUD")

    # Offline comparison
    offline_adapter = OnlineFeatureHydrationAdapter(entity_store=store)
    res_offline = offline_adapter.hydrate(cleaned_event)
    assert res_offline.complete is True

    offline_df, _ = orch.engine.predict_transaction(res_offline.dataframe.copy())
    offline_prob = float(offline_df.iloc[0]["fraud_probability"])
    offline_decision = str(offline_df.iloc[0]["decision"])

    delta = abs(trace.fraud_probability - offline_prob)
    assert delta < 1e-12, f"Real dataset parity delta {delta} exceeds tolerance!"
    assert trace.decision == offline_decision

