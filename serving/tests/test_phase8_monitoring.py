"""
tests/test_phase8_monitoring.py
Automated unit test suite for Phase 8 — Monitoring & Model Versioning.

Tests:
 1. MonitoringState counts total requests correctly
 2. MonitoringState tracks successful (2xx) and failed (4xx/5xx) requests
 3. MonitoringState accumulates latency values per prediction
 4. MonitoringState.get_summary() returns all expected keys with correct types
 5. MonitoringState latency percentiles are computed correctly
 6. MODEL_METADATA has correct E1 model metadata values
 7. GET /health returns model_version field with correct value
 8. GET /health still returns all Phase 3/4/5/6/7 fields (regression)
 9. POST /predict still returns all expected response fields (regression)
10. POST /predict/batch still returns all expected response fields (regression)
11. A successful /predict call increments the monitor counter
12. A failed (validation error) request increments monitor.failed_requests
13. inference_logger emits a log line with required fields (no raw payload)
14. inference_logger does NOT include raw transaction payload in log output
15. model_version in inference log matches MODEL_METADATA["model_version"]
"""
from __future__ import annotations

import logging
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

# Add project root and serving directory to sys.path
SERVING_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SERVING_DIR.parent
for p in [str(REPO_ROOT), str(SERVING_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from api.main import app
from inference.offline_inference import OfflineInferenceEngine, E1_DECISION_THRESHOLD
from monitoring.api_monitor import MonitoringState
from monitoring.model_metadata import MODEL_METADATA
from monitoring.inference_logger import log_inference_event, log_inference_error
from monitoring.api_monitor import monitor


def ensure_engine_loaded():
    """Ensure app.state.engine holds a valid OfflineInferenceEngine instance."""
    if not isinstance(getattr(app.state, "engine", None), OfflineInferenceEngine):
        app.state.engine = OfflineInferenceEngine(
            model_path="models/E1/model.txt",
            preprocessor_path="models/E1/preprocessing.joblib",
            feature_names_path="models/E1/feature_names.json",
            threshold=E1_DECISION_THRESHOLD,
        )


# ── Test 1: Total request counting ───────────────────────────────────────────

def test_monitor_counts_requests():
    """MonitoringState increments total_requests on every record() call."""
    m = MonitoringState()
    assert m.total_requests == 0
    m.record(http_status=200, latency_ms=85.0)
    m.record(http_status=200, latency_ms=90.0)
    m.record(http_status=422)
    assert m.total_requests == 3
    print("[PASS] test_monitor_counts_requests — total_requests == 3")


# ── Test 2: Success/error tracking ───────────────────────────────────────────

def test_monitor_success_error_tracking():
    """MonitoringState correctly classifies 2xx, 4xx, 5xx responses."""
    m = MonitoringState()
    m.record(http_status=200, latency_ms=80.0)
    m.record(http_status=200, latency_ms=90.0)
    m.record(http_status=422)
    m.record(http_status=500)

    assert m.successful_requests == 2,  f"Expected 2, got {m.successful_requests}"
    assert m.failed_requests == 2,      f"Expected 2, got {m.failed_requests}"
    assert m.http_2xx_count == 2,       f"Expected 2, got {m.http_2xx_count}"
    assert m.http_4xx_count == 1,       f"Expected 1, got {m.http_4xx_count}"
    assert m.http_5xx_count == 1,       f"Expected 1, got {m.http_5xx_count}"
    print("[PASS] test_monitor_success_error_tracking")


# ── Test 3: Latency accumulation ─────────────────────────────────────────────

def test_monitor_latency_recorded():
    """MonitoringState accumulates latency values from successful predictions."""
    m = MonitoringState()
    m.record(http_status=200, latency_ms=70.0)
    m.record(http_status=200, latency_ms=90.0)
    m.record(http_status=422)          # no latency for error

    assert len(m.latency_ms_list) == 2, f"Expected 2 latency records, got {len(m.latency_ms_list)}"
    assert 70.0 in m.latency_ms_list
    assert 90.0 in m.latency_ms_list
    print("[PASS] test_monitor_latency_recorded")


# ── Test 4: get_summary() structure ──────────────────────────────────────────

def test_monitor_get_summary_structure():
    """get_summary() returns a dict with all expected keys and correct types."""
    m = MonitoringState()
    m.record(http_status=200, latency_ms=85.0)
    m.record(http_status=422)

    summary = m.get_summary()
    expected_keys = [
        "total_requests", "successful_requests", "failed_requests",
        "http_2xx_count", "http_4xx_count", "http_5xx_count",
        "latency_sample_count", "latency_min_ms", "latency_mean_ms",
        "latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "latency_max_ms",
    ]
    for key in expected_keys:
        assert key in summary, f"Missing key in get_summary(): {key}"

    assert summary["total_requests"] == 2
    assert summary["successful_requests"] == 1
    assert summary["failed_requests"] == 1
    assert summary["latency_sample_count"] == 1
    assert summary["latency_min_ms"] == 85.0
    print("[PASS] test_monitor_get_summary_structure")


# ── Test 5: Latency percentiles ───────────────────────────────────────────────

def test_monitor_latency_percentiles():
    """Latency percentiles are computed correctly from a known distribution."""
    m = MonitoringState()
    # Record 10 known values: 10, 20, 30, ..., 100 ms
    for v in range(10, 110, 10):
        m.record(http_status=200, latency_ms=float(v))

    summary = m.get_summary()
    assert summary["latency_min_ms"] == 10.0,  f"Expected 10.0, got {summary['latency_min_ms']}"
    assert summary["latency_max_ms"] == 100.0, f"Expected 100.0, got {summary['latency_max_ms']}"
    assert summary["latency_mean_ms"] == 55.0, f"Expected 55.0, got {summary['latency_mean_ms']}"
    assert summary["latency_p50_ms"] is not None
    assert summary["latency_p95_ms"] is not None
    assert summary["latency_p99_ms"] is not None
    assert summary["latency_p95_ms"] >= summary["latency_p50_ms"]
    assert summary["latency_p99_ms"] >= summary["latency_p95_ms"]
    print("[PASS] test_monitor_latency_percentiles")


# ── Test 6: MODEL_METADATA correctness ───────────────────────────────────────

def test_model_metadata_correct():
    """MODEL_METADATA contains the correct E1 model identity values."""
    assert MODEL_METADATA["model_name"] == "E1_LightGBM",       f"model_name mismatch: {MODEL_METADATA['model_name']}"
    assert MODEL_METADATA["model_version"] == "E1",             f"model_version mismatch: {MODEL_METADATA['model_version']}"
    assert MODEL_METADATA["feature_count"] == 406,              f"feature_count mismatch: {MODEL_METADATA['feature_count']}"
    assert "model.txt" in MODEL_METADATA["artifact_model"]
    print("[PASS] test_model_metadata_correct")


# ── Test 7: /health includes model_version ────────────────────────────────────

def test_health_includes_model_version(client: TestClient):
    """GET /health response now includes model_version field set to 'E1'."""
    ensure_engine_loaded()
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "model_version" in data, "model_version field missing from /health response"
    assert data["model_version"] == "E1", f"Expected 'E1', got '{data['model_version']}'"
    print(f"[PASS] test_health_includes_model_version — model_version={data['model_version']}")


# ── Test 8: /health regression (all existing fields still present) ────────────

def test_health_regression(client: TestClient):
    """GET /health still returns all Phase 3 expected fields after Phase 8 changes."""
    ensure_engine_loaded()
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_name"] == "E1_LightGBM"
    assert data["feature_count"] == 406
    assert abs(data["decision_threshold"] - 0.616521) < 1e-6
    print("[PASS] test_health_regression")


# ── Test 9: /predict response unchanged ──────────────────────────────────────

def test_predict_response_unchanged(client: TestClient):
    """POST /predict still returns all Phase 3 expected response fields."""
    ensure_engine_loaded()
    resp = client.post("/predict", json={
        "TransactionID": 3544193, "TransactionDT": 14757390,
        "TransactionAmt": 100.0, "ProductCD": "W",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "transaction_id" in data
    assert "fraud_probability" in data
    assert "decision" in data
    assert "actual_label" in data
    assert "preprocessing_time_ms" in data
    assert "model_prediction_time_ms" in data
    assert "total_inference_time_ms" in data
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["decision"] in ("FRAUD", "LEGIT")
    print("[PASS] test_predict_response_unchanged")


# ── Test 10: /predict/batch response unchanged ────────────────────────────────

def test_batch_response_unchanged(client: TestClient):
    """POST /predict/batch still returns all Phase 3 expected response fields."""
    ensure_engine_loaded()
    resp = client.post("/predict/batch", json={"transactions": [
        {"TransactionID": 888001, "TransactionAmt": 50.0, "ProductCD": "W"},
        {"TransactionID": 888002, "TransactionAmt": 500.0, "ProductCD": "C"},
    ]})
    assert resp.status_code == 200
    data = resp.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2
    for pred in data["predictions"]:
        assert "transaction_id" in pred
        assert "fraud_probability" in pred
        assert "decision" in pred
    print("[PASS] test_batch_response_unchanged")


# ── Test 11: monitor increments after /predict ────────────────────────────────

def test_monitor_increments_on_predict(client: TestClient):
    """A successful /predict call increments monitor.total_requests and successful_requests."""
    ensure_engine_loaded()
    monitor.reset()

    client.post("/predict", json={
        "TransactionID": 777001, "TransactionAmt": 75.0, "ProductCD": "W",
    })

    assert monitor.total_requests == 1,      f"Expected 1, got {monitor.total_requests}"
    assert monitor.successful_requests == 1, f"Expected 1, got {monitor.successful_requests}"
    assert monitor.failed_requests == 0,     f"Expected 0, got {monitor.failed_requests}"
    assert len(monitor.latency_ms_list) == 1
    print(f"[PASS] test_monitor_increments_on_predict — latency={monitor.latency_ms_list[0]:.3f}ms")


# ── Test 12: monitor tracks failed requests ───────────────────────────────────

def test_monitor_tracks_errors(client: TestClient):
    """A validation error (422) increments monitor.failed_requests."""
    ensure_engine_loaded()
    monitor.reset()

    client.post("/predict/batch", json={"transactions": []})     # → 422

    assert monitor.failed_requests >= 1, f"Expected >=1, got {monitor.failed_requests}"
    assert monitor.http_4xx_count >= 1,  f"Expected >=1, got {monitor.http_4xx_count}"
    print(f"[PASS] test_monitor_tracks_errors — failed={monitor.failed_requests}, 4xx={monitor.http_4xx_count}")


# ── Test 13: Inference logger emits required fields ───────────────────────────

def test_inference_log_event_fields():
    """log_inference_event emits a structured log line with all required fields."""
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.INFO)
    inf_logger = logging.getLogger("inference_monitor")
    inf_logger.addHandler(handler)
    inf_logger.setLevel(logging.INFO)

    log_inference_event(
        endpoint="/predict",
        transaction_id="3544193",
        model_version="E1",
        fraud_probability=0.003993,
        decision="LEGIT",
        preprocessing_ms=53.5,
        model_ms=2.4,
        total_ms=55.9,
        http_status=200,
    )

    log_output = log_stream.getvalue()
    inf_logger.removeHandler(handler)

    assert "endpoint=/predict" in log_output,            "Missing endpoint field"
    assert "transaction_id=3544193" in log_output,       "Missing transaction_id field"
    assert "model_version=E1" in log_output,             "Missing model_version field"
    assert "fraud_probability=0.003993" in log_output,   "Missing fraud_probability field"
    assert "decision=LEGIT" in log_output,               "Missing decision field"
    assert "preprocessing_ms=53.500" in log_output,      "Missing preprocessing_ms field"
    assert "http_status=200" in log_output,              "Missing http_status field"
    print("[PASS] test_inference_log_event_fields")


# ── Test 14: Inference logger does NOT contain raw payload ────────────────────

def test_inference_log_no_raw_payload():
    """log_inference_event does NOT include raw transaction payload fields in output."""
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    inf_logger = logging.getLogger("inference_monitor")
    inf_logger.addHandler(handler)
    inf_logger.setLevel(logging.INFO)

    log_inference_event(
        endpoint="/predict",
        transaction_id="9999",
        model_version="E1",
        fraud_probability=0.12,
        decision="LEGIT",
        preprocessing_ms=70.0,
        model_ms=3.0,
        total_ms=73.0,
        http_status=200,
    )

    log_output = log_stream.getvalue()
    inf_logger.removeHandler(handler)

    # These raw payload fields must NOT appear in the log
    assert "TransactionAmt" not in log_output,      "Raw field TransactionAmt found in log — privacy violation"
    assert "TransactionDT" not in log_output,       "Raw field TransactionDT found in log — privacy violation"
    assert "card1" not in log_output,               "Raw field card1 found in log — privacy violation"
    assert "ProductCD" not in log_output,           "Raw field ProductCD found in log — privacy violation"
    assert "addr1" not in log_output,               "Raw field addr1 found in log — privacy violation"
    print("[PASS] test_inference_log_no_raw_payload")


# ── Test 15: model_version in log matches MODEL_METADATA ─────────────────────

def test_model_version_in_log_matches_metadata():
    """model_version logged by inference_logger matches MODEL_METADATA['model_version']."""
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    inf_logger = logging.getLogger("inference_monitor")
    inf_logger.addHandler(handler)
    inf_logger.setLevel(logging.INFO)

    log_inference_event(
        endpoint="/predict",
        transaction_id="123",
        model_version=MODEL_METADATA["model_version"],
        fraud_probability=0.05,
        decision="LEGIT",
        preprocessing_ms=60.0,
        model_ms=3.0,
        total_ms=63.0,
        http_status=200,
    )

    log_output = log_stream.getvalue()
    inf_logger.removeHandler(handler)

    assert f"model_version={MODEL_METADATA['model_version']}" in log_output, \
        f"model_version in log does not match MODEL_METADATA: got log='{log_output}'"
    print(f"[PASS] test_model_version_in_log_matches_metadata — model_version={MODEL_METADATA['model_version']}")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running Phase 8 Monitoring & Model Versioning Unit Tests...")
    print()

    # Unit tests (no client needed)
    test_monitor_counts_requests()
    test_monitor_success_error_tracking()
    test_monitor_latency_recorded()
    test_monitor_get_summary_structure()
    test_monitor_latency_percentiles()
    test_model_metadata_correct()
    test_inference_log_event_fields()
    test_inference_log_no_raw_payload()
    test_model_version_in_log_matches_metadata()

    # API integration tests (require TestClient)
    with TestClient(app) as c:
        test_health_includes_model_version(c)
        test_health_regression(c)
        test_predict_response_unchanged(c)
        test_batch_response_unchanged(c)
        test_monitor_increments_on_predict(c)
        test_monitor_tracks_errors(c)

    print()
    print("ALL PHASE 8 MONITORING & MODEL VERSIONING UNIT TESTS PASSED SUCCESSFULLY!")
