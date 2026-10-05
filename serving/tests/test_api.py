"""
tests/test_api.py
Automated API tests for Phase 3 FastAPI Model Serving in fraud-model-serving.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root and serving directory to sys.path
SERVING_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SERVING_DIR.parent
for p in [str(REPO_ROOT), str(SERVING_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from api.main import app
from inference.offline_inference import (
    E1_DECISION_THRESHOLD,
    OfflineInferenceEngine,
    load_e1_test_transactions,
)


@pytest.fixture(scope="module")
def client():
    """TestClient fixture with app lifespan initialized."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient):
    """Verify GET /health endpoint status and model loading."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["feature_count"] == 406
    assert data["decision_threshold"] == E1_DECISION_THRESHOLD


def test_predict_valid_transaction(client: TestClient):
    """Verify POST /predict with a valid dummy transaction payload."""
    payload = {
        "TransactionID": 999901,
        "TransactionDT": 14000000,
        "TransactionAmt": 150.0,
        "ProductCD": "W",
        "card1": 1000,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert str(data["transaction_id"]) == "999901"
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["decision"] in ("FRAUD", "LEGIT")
    assert data["preprocessing_time_ms"] >= 0.0
    assert data["model_prediction_time_ms"] >= 0.0
    assert data["total_inference_time_ms"] >= 0.0


def test_predict_consistency_with_offline(client: TestClient):
    """
    Verify POST /predict produces identical probability and decision
    as the offline inference pipeline for real E1 test transaction 3544193.
    """
    # 1. Get offline prediction
    engine = OfflineInferenceEngine()
    df_test = load_e1_test_transactions(sample_size=1, random_seed=42)
    offline_res, _ = engine.predict_transaction(df_test)
    offline_row = offline_res.iloc[0]

    # 2. Extract raw transaction dict from test row
    tx_dict = df_test.iloc[0].to_dict()
    # Clean NaN values for JSON serialization
    tx_dict_clean = {k: v for k, v in tx_dict.items() if pd.notna(v)}

    # Remove isFraud to test online serving edge case
    tx_dict_clean.pop("isFraud", None)

    # 3. Post to API
    response = client.post("/predict", json=tx_dict_clean)
    assert response.status_code == 200
    api_data = response.json()

    # 4. Assert predictions match
    assert str(api_data["transaction_id"]) == str(offline_row["transaction_id"])
    assert abs(api_data["fraud_probability"] - offline_row["fraud_probability"]) < 1e-5
    assert api_data["decision"] == offline_row["decision"]
    assert api_data["decision"] == "LEGIT"
    assert abs(api_data["fraud_probability"] - 0.003993) < 1e-4


def test_predict_without_is_fraud(client: TestClient):
    """Verify POST /predict handles transactions without target label isFraud."""
    payload = {
        "TransactionID": 999902,
        "TransactionDT": 15000000,
        "TransactionAmt": 250.0,
        "ProductCD": "C",
    }
    # Explicitly ensure isFraud is not in payload
    assert "isFraud" not in payload

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert str(data["transaction_id"]) == "999902"
    assert data["decision"] in ("FRAUD", "LEGIT")


def test_predict_invalid_request(client: TestClient):
    """Verify POST /predict rejects malformed / invalid requests cleanly."""
    # Malformed data (e.g. invalid array instead of object)
    response = client.post("/predict", json=["invalid_array"])
    assert response.status_code in (400, 422)
    data = response.json()
    assert "error" in data


def test_batch_predict_endpoint(client: TestClient):
    """Verify POST /predict/batch endpoint with multi-transaction batch."""
    payload = {
        "transactions": [
            {"TransactionID": 888001, "TransactionDT": 14000000, "TransactionAmt": 50.0, "ProductCD": "W"},
            {"TransactionID": 888002, "TransactionDT": 14001000, "TransactionAmt": 500.0, "ProductCD": "C"},
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "predictions" in data
    assert len(data["predictions"]) == 2
    assert str(data["predictions"][0]["transaction_id"]) == "888001"
    assert str(data["predictions"][1]["transaction_id"]) == "888002"


if __name__ == "__main__":
    print("Running Phase 3 FastAPI Serving Unit Tests...")
    # Initialize test client directly
    with TestClient(app) as c:
        test_health_endpoint(c)
        print("[PASS] test_health_endpoint")
        test_predict_valid_transaction(c)
        print("[PASS] test_predict_valid_transaction")
        test_predict_consistency_with_offline(c)
        print("[PASS] test_predict_consistency_with_offline")
        test_predict_without_is_fraud(c)
        print("[PASS] test_predict_without_is_fraud")
        test_predict_invalid_request(c)
        print("[PASS] test_predict_invalid_request")
        test_batch_predict_endpoint(c)
        print("[PASS] test_batch_predict_endpoint")
    print("\nALL PHASE 3 API UNIT TESTS PASSED SUCCESSFULLY!")
