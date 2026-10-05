"""
tests/test_phase6_error_handling.py
Automated unit test suite for Phase 6 — Failure & Error Handling.

Tests:
1. Missing Required Request Fields (HTTP 422)
2. Wrong Data Types in Request Payload (HTTP 422)
3. Malformed JSON Input (HTTP 422/400)
4. Invalid/Missing Feature Data Handling (HTTP 200)
5. Invalid Batch Requests (empty list, missing transactions key, invalid items) (HTTP 422)
6. Model Unavailable / Startup Loading Failure (HTTP 503)
7. Preprocessor Loading Failure Handling (FileNotFoundError)
8. Controlled Runtime Inference Exception (HTTP 500)
9. Health Check Verification After Error Scenarios (HTTP 200)
10. Error Response JSON Structure Validation (HTTP 422)
11. Zero Internal Path / Traceback Leakage Verification (HTTP 500)
12. Service Resilience Verification (HTTP 200)
"""
from __future__ import annotations

import sys
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
from inference.offline_inference import OfflineInferenceEngine


def ensure_engine_loaded():
    """Ensure app.state.engine holds a valid real OfflineInferenceEngine instance."""
    if not isinstance(getattr(app.state, "engine", None), OfflineInferenceEngine):
        app.state.engine = OfflineInferenceEngine(
            model_path="models/E1/model.txt",
            preprocessor_path="models/E1/preprocessing.joblib",
            feature_names_path="models/E1/feature_names.json",
            threshold=0.616521,
        )


def test_missing_required_request_fields(client: TestClient):
    """Test sending malformed/empty payload missing expected fields."""
    ensure_engine_loaded()
    response = client.post("/predict", json={"features": {}})
    assert response.status_code == 200  # Dynamic extra fields allowed
    data = response.json()
    assert "fraud_probability" in data
    assert "decision" in data

    # Batch endpoint missing required 'transactions' field
    response_batch = client.post("/predict/batch", json={})
    assert response_batch.status_code == 422
    body = response_batch.json()
    assert "error" in body


def test_wrong_data_types(client: TestClient):
    """Test sending wrong data types in batch requests."""
    ensure_engine_loaded()
    response = client.post("/predict/batch", json={"transactions": "invalid_string_not_list"})
    assert response.status_code == 422
    body = response.json()
    assert "error" in body
    assert body["error"] == "Request Validation Error"


def test_malformed_json_input(client: TestClient):
    """Test sending malformed/corrupt JSON syntax."""
    ensure_engine_loaded()
    response = client.post(
        "/predict",
        content='{"TransactionID": 3544193, "TransactionAmt": ',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code in (400, 422)
    body = response.json()
    assert "error" in body


def test_invalid_feature_data_handling(client: TestClient):
    """Test handling of null or NaN feature inputs."""
    ensure_engine_loaded()
    payload = {
        "TransactionID": 999999,
        "TransactionDT": None,
        "TransactionAmt": -100.0,  # Negative transaction amount
        "ProductCD": None,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data


def test_invalid_batch_requests(client: TestClient):
    """Test /predict/batch with empty list, missing field, wrong structure."""
    ensure_engine_loaded()
    # 5a. Empty batch list (min_items=1 constraint check)
    resp_empty = client.post("/predict/batch", json={"transactions": []})
    assert resp_empty.status_code == 422

    # 5b. Invalid item in batch
    resp_bad_item = client.post("/predict/batch", json={"transactions": [12345]})
    assert resp_bad_item.status_code == 422


def test_model_unavailable_handling(client: TestClient):
    """Test API when E1 engine is not loaded in memory (HTTP 503)."""
    ensure_engine_loaded()
    real_engine = app.state.engine
    try:
        app.state.engine = None
        response = client.post(
            "/predict",
            json={"TransactionID": 3544193, "TransactionAmt": 100.0},
        )
        assert response.status_code == 503
        body = response.json()
        assert "error" in body
        assert "not loaded" in body["detail"].lower()
    finally:
        app.state.engine = real_engine


def test_preprocessor_loading_failure():
    """Test exception handling during preprocessor artifact initialization."""
    with patch("preprocessing.serving_wrapper.joblib.load", side_effect=FileNotFoundError("Mock preprocessor missing")):
        try:
            from preprocessing.serving_wrapper import ServingPreprocessor
            ServingPreprocessor("invalid/path.joblib", "models/E1/feature_names.json")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass


def test_runtime_inference_failure(client: TestClient):
    """Test 500 internal server error when inference engine throws an unexpected exception."""
    ensure_engine_loaded()
    with patch.object(app.state.engine, "predict_transaction", side_effect=RuntimeError("Simulated internal CUDA/LGBM failure")):
        response = client.post(
            "/predict",
            json={"TransactionID": 3544193, "TransactionAmt": 100.0},
        )
        assert response.status_code == 500
        body = response.json()
        assert "error" in body
        assert body["error"] == "HTTP Error"
        assert "internal error" in body["detail"].lower()


def test_health_check_after_errors(client: TestClient):
    """Test GET /health after intentionally sending invalid and failed requests."""
    ensure_engine_loaded()
    # Trigger validation error
    client.post("/predict/batch", json={"transactions": []})
    # Trigger malformed JSON error
    client.post("/predict", content="{bad_json", headers={"Content-Type": "application/json"})

    # Health check must still be 200 OK
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"


def test_error_response_structure(client: TestClient):
    """Test that error responses strictly match expected JSON keys."""
    ensure_engine_loaded()
    resp = client.post("/predict/batch", json={})
    assert resp.status_code == 422
    data = resp.json()
    assert "error" in data
    assert "detail" in data
    assert isinstance(data["error"], str)


def test_no_traceback_or_path_leakage(client: TestClient):
    """Assert no local filesystem paths or stack traces are leaked in HTTP response bodies."""
    ensure_engine_loaded()
    with patch.object(app.state.engine, "predict_transaction", side_effect=Exception("Critical Error in C:\\Users\\DEV_WORKSPACE\\secret_script.py line 42")):
        resp = client.post("/predict", json={"TransactionID": 100})
        assert resp.status_code == 500
        content_str = resp.text
        # Must NOT contain filesystem paths or stack traces
        assert "C:\\Users" not in content_str
        assert "secret_script.py" not in content_str
        assert "Traceback (most recent call last)" not in content_str


def test_service_resilience(client: TestClient):
    """Verify normal predictions succeed after an error has occurred."""
    ensure_engine_loaded()
    # Force an error
    client.post("/predict/batch", json={"transactions": "bad_type"})

    # Valid prediction request must still succeed
    resp = client.post(
        "/predict",
        json={"TransactionID": 3544193, "TransactionDT": 14757390, "TransactionAmt": 100.0, "ProductCD": "W"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["transaction_id"] == "3544193"


if __name__ == "__main__":
    print("Running Phase 6 Failure & Error Handling Unit Tests...")
    with TestClient(app) as c:
        test_missing_required_request_fields(c)
        print("[PASS] test_missing_required_request_fields")
        test_wrong_data_types(c)
        print("[PASS] test_wrong_data_types")
        test_malformed_json_input(c)
        print("[PASS] test_malformed_json_input")
        test_invalid_feature_data_handling(c)
        print("[PASS] test_invalid_feature_data_handling")
        test_invalid_batch_requests(c)
        print("[PASS] test_invalid_batch_requests")
        test_model_unavailable_handling(c)
        print("[PASS] test_model_unavailable_handling")
        test_preprocessor_loading_failure()
        print("[PASS] test_preprocessor_loading_failure")
        test_runtime_inference_failure(c)
        print("[PASS] test_runtime_inference_failure")
        test_health_check_after_errors(c)
        print("[PASS] test_health_check_after_errors")
        test_error_response_structure(c)
        print("[PASS] test_error_response_structure")
        test_no_traceback_or_path_leakage(c)
        print("[PASS] test_no_traceback_or_path_leakage")
        test_service_resilience(c)
        print("[PASS] test_service_resilience")
    print("\nALL PHASE 6 ERROR HANDLING UNIT TESTS PASSED SUCCESSFULLY!")
