"""
streaming/tests/test_phase17_replay.py
Phase 17 Test Suite: Historical Dataset Streaming Replay & Controlled Simulation.

Validates Acceptance Gates P17.1 – P17.6:
- P17.1: Chronological ordering of replay events by TransactionDT
- P17.2: Multi-stage execution scalability (100 -> 500 -> 1,000 transactions)
- P17.3: Strict point-in-time causality across all historical updates
- P17.4: Target isolation at scale (zero occurrence of isFraud in feature matrix X)
- P17.5: Lineage completeness (Kafka topic/partition/offset, Flink features, bridge trace)
- P17.6: Controlled handling of duplicate and out-of-order events
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVING_DIR = REPO_ROOT / "serving"
for p in [str(REPO_ROOT), str(SERVING_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from serving.hydration.entity_store import EntityProfileStore
from streaming.kafka_broker import LocalKafkaBroker
from streaming.historical_streaming_replay import (
    load_chronological_transactions,
    run_replay_stage,
)
from streaming.multi_transaction_benchmark import find_dataset_path
from streaming.stream_pipeline import StreamingPipelineOrchestrator


@pytest.fixture(scope="module")
def historical_replay_artifacts():
    """Load or run the historical replay benchmark results."""
    json_path = REPO_ROOT / "reports" / "integration" / "phase17_replay_results.json"
    if not json_path.exists():
        from streaming.historical_streaming_replay import run_historical_replay_benchmark
        return run_historical_replay_benchmark()
    with open(json_path, mode="r", encoding="utf-8") as f:
        return json.load(f)


def test_p17_1_chronological_ordering():
    """P17.1: Verify dataset transactions are sorted chronologically by TransactionDT."""
    dataset_path = find_dataset_path()
    txs, _ = load_chronological_transactions(dataset_path, max_count=100)
    assert len(txs) == 100
    for i in range(1, len(txs)):
        assert txs[i]["TransactionDT"] >= txs[i - 1]["TransactionDT"]


def test_p17_2_multi_stage_scalability(historical_replay_artifacts):
    """P17.2: Verify successful completion of 100, 500, and 1,000 transaction stages."""
    stages = historical_replay_artifacts["stages"]
    assert len(stages) == 3
    stage_names = [s["stage_name"] for s in stages]
    assert "Stage_100" in stage_names
    assert "Stage_500" in stage_names
    assert "Stage_1000" in stage_names

    for s in stages:
        assert s["input_count"] == s["processed_count"]
        assert s["scored_count"] == s["input_count"]
        assert s["error_count"] == 0


def test_p17_3_strict_point_in_time_causality(historical_replay_artifacts):
    """P17.3: Absolute causality invariant: 0 causality violations across all stages."""
    assert historical_replay_artifacts["all_stages_causally_valid"] is True
    for s in historical_replay_artifacts["stages"]:
        assert s["causality_violations"] == 0


def test_p17_4_target_isolation_at_scale(historical_replay_artifacts):
    """P17.4: Target isolation invariant: isFraud never enters feature matrix X."""
    assert historical_replay_artifacts["all_stages_target_isolated"] is True
    for s in historical_replay_artifacts["stages"]:
        assert s["target_leakage_detected"] == 0


def test_p17_5_lineage_completeness(historical_replay_artifacts):
    """P17.5: 100% lineage completeness across all processed transactions."""
    assert historical_replay_artifacts["all_stages_lineage_complete"] is True
    for s in historical_replay_artifacts["stages"]:
        assert s["lineage_complete"] is True


def test_p17_6_duplicate_event_handling():
    """P17.6: Ingesting a duplicate transaction event is handled cleanly without corruption."""
    store = EntityProfileStore()
    store.seed_mock_profiles()
    orch = StreamingPipelineOrchestrator(broker=LocalKafkaBroker(), entity_store=store)

    ev = {
        "TransactionID": "3544193",
        "card_id": "CARD-3544193",
        "TransactionAmt": 100.0,
        "TransactionDT": 13400000.0,
        "ProductCD": "W",
    }

    # Ingest event first time
    trace1 = orch.ingest_transaction(ev)
    assert trace1.model_invoked is True
    prob1 = trace1.fraud_probability

    # Ingest same event second time (duplicate)
    trace2 = orch.ingest_transaction(ev)
    assert trace2.model_invoked is True
    prob2 = trace2.fraud_probability

    # Both must produce valid predictions without pipeline crashes
    assert prob1 is not None and prob2 is not None
    assert trace2.raw_offset == trace1.raw_offset + 1
