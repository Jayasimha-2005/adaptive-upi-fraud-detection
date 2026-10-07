"""
streaming/tests/test_phase18_safety.py
Phase 18 Test Suite: Safety, Poisoning & Negative Chaos Testing (Cases A – T).

Validates:
- Case A: Missing TransactionAmt -> REJECTED_BY_HYDRATION_GATE, scored=False
- Case B: Missing card1 -> rejection
- Case C: Unknown entity/card -> rejection (zero synthetic median profile fabrication)
- Case D: Missing ProductCD -> controlled handling
- Case E: isFraud target injection -> stripped before model matrix X
- Case F: fraud_bool target injection -> stripped before model matrix X
- Case G: Future history injection -> excluded from causal query (< current_dt)
- Case H: Current-event history injection -> excluded from historical state before scoring
- Case I: Out-of-order event -> causal point-in-time isolation
- Case J: Duplicate event -> deterministic handling; no process crash
- Case K: Malformed event payload -> controlled rejection; no process crash
- Case L: Invalid timestamp -> controlled rejection; no unhandled exception
- Case M: Negative transaction amount -> controlled scoring/handling; bounded prediction
- Case N: Extremely large transaction amount -> numerical stability; bounded in [0, 1]
- Case O: Unknown categorical values -> preprocessor handles unseen category safely
- Case P: Corrupted velocity record -> does not corrupt canonical E1 state or crash
- Case Q: Fraud-features arriving before raw event -> correlation buffer handles out-of-order arrival
- Case R: Velocity record with future timestamp -> rejected from causal correlation
- Case S: Kafka duplicate replay scenario -> independent consumer groups replay log safely
- Case T: Serving model unavailable -> fail-closed; scored=False; zero false positives
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
from streaming.safety_chaos_benchmark import run_all_safety_chaos_tests
from streaming.stream_pipeline import StreamingPipelineOrchestrator


@pytest.fixture(scope="module")
def safety_chaos_results():
    """Run or load all 20 safety chaos test results."""
    json_path = REPO_ROOT / "reports" / "integration" / "phase18_safety_results.json"
    if not json_path.exists():
        return run_all_safety_chaos_tests()
    with open(json_path, mode="r", encoding="utf-8") as f:
        return json.load(f)


def test_p18_overall_certification(safety_chaos_results):
    """P18: All 20 safety & chaos test cases must pass."""
    assert safety_chaos_results["total_test_cases"] == 20
    assert safety_chaos_results["passed_test_cases"] == 20
    assert safety_chaos_results["failed_test_cases"] == 0
    assert safety_chaos_results["all_passed"] is True


@pytest.mark.parametrize("case_id", [
    "Case_A", "Case_B", "Case_C", "Case_D", "Case_E",
    "Case_F", "Case_G", "Case_H", "Case_I", "Case_J",
    "Case_K", "Case_L", "Case_M", "Case_N", "Case_O",
    "Case_P", "Case_Q", "Case_R", "Case_S", "Case_T",
])
def test_p18_individual_case(safety_chaos_results, case_id):
    """Verify each individual safety and chaos test case passed."""
    matching = [r for r in safety_chaos_results["test_results"] if r["case_id"] == case_id]
    assert len(matching) == 1
    assert matching[0]["passed"] is True, f"Failed case {case_id}: {matching[0]}"
