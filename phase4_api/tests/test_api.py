"""
phase4_api/tests/test_api.py
============================
Comprehensive Automated Test Suite for Phase 7 FastAPI Serving Layer.

Covers all 49 required test scenarios across 9 categories:
  A. Application Tests (1-5)
  B. Single Endpoint Tests (6-14)
  C. Batch Endpoint Tests (15-26)
  D. Health Endpoint Tests (27-30)
  E. Model Information Tests (31-35)
  F. Centralized Error Handling Tests (36-39)
  G. OpenAPI Documentation Tests (40-44)
  H. Internal Integration Tests (45-48)
  I. Real-Model Smoke Test (49)
"""
from __future__ import annotations

import json
import pathlib
import sys
from typing import Any, Dict, List
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

# Ensure workspace root is in sys.path
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase4_api.src.batch_inference_engine import (
    BatchPredictionResponse,
    BatchRecordResult,
    BatchSizeLimitExceededError,
    BatchTransactionInferenceEngine,
)
from phase4_api.src.dependencies import (
    get_active_bundle,
    get_batch_engine,
    get_single_engine,
)
from phase4_api.src.inference_engine import (
    InferenceEngineError,
    ModelNotReadyError,
    ModelPredictionError,
    PredictionResponse,
    SingleTransactionInferenceEngine,
)
from phase4_api.src.main import app, create_app
from phase4_api.src.model_loader import (
    APPROVED_DEFAULT_MODEL_VERSION,
    EXPECTED_FEATURE_COUNT,
    LoadedModelBundle,
    ModelInfo,
    ModelLoaderError,
    get_model_bundle,
)
from phase4_api.src.risk_classifier import (
    ReasonCode,
    RiskClassifier,
    RiskLevel,
    RiskPolicy,
    get_risk_classifier,
)
from phase4_api.tests.test_transaction_preprocessor import get_sample_valid_transaction

FROZEN_THRESHOLD: float = 0.8964321608040201


class MockBooster:
    """Mock LightGBM Booster returning controlled predictions."""

    def __init__(self, return_val: Any = 0.5) -> None:
        self.return_val = return_val
        self.call_count = 0

    def predict(self, df: pd.DataFrame) -> Any:
        self.call_count += 1
        if isinstance(self.return_val, Exception):
            raise self.return_val
        if isinstance(self.return_val, (int, float)):
            if len(df) == 1:
                return np.array([float(self.return_val)], dtype=np.float64)
            return np.full(len(df), float(self.return_val), dtype=np.float64)
        return self.return_val


def create_mock_bundle(
    booster: Any,
    threshold: float = FROZEN_THRESHOLD,
    version: str = "v3",
) -> LoadedModelBundle:
    info = ModelInfo(
        model_version=version,
        model_artifact_path=f"artifacts/models/lgbm_{version}.txt",
        num_features=EXPECTED_FEATURE_COUNT,
        feature_names=["col"] * EXPECTED_FEATURE_COUNT,
        num_trees=500,
        frozen_threshold=threshold,
        model_loading_status="ACTIVE_LOADED",
        training_months=[0, 1, 2, 3, 4, 5, 6],
        model_hash="mock_hash_12345",
    )
    return LoadedModelBundle(
        booster=booster,
        info=info,
        metadata={"version_label": version},
        frozen_threshold=threshold,
    )


class TestFastAPIServing(unittest.TestCase):
    """Test suite for Phase 7 FastAPI serving layer."""

    def setUp(self) -> None:
        self.app = create_app()
        self.client = TestClient(self.app)
        self.sample_tx = get_sample_valid_transaction()
        self.sample_payload = dict(self.sample_tx, transaction_id="TEST-TX-001")

    def tearDown(self) -> None:
        self.app.dependency_overrides.clear()

    # -----------------------------------------------------------------------
    # A. Application Tests (1 - 5)
    # -----------------------------------------------------------------------
    def test_01_application_imports_successfully(self) -> None:
        """1. Application module imports successfully."""
        from phase4_api.src.main import app as main_app
        self.assertIsNotNone(main_app)

    def test_02_application_instance_is_created(self) -> None:
        """2. FastAPI application instance is properly configured."""
        self.assertEqual(self.app.title, "Adaptive Financial Fraud Detection API")
        self.assertEqual(self.app.version, "1.0.0")

    def test_03_lifespan_initializes_dependencies(self) -> None:
        """3. Lifespan context manager starts up cleanly."""
        with TestClient(self.app) as c:
            resp = c.get("/health")
            self.assertEqual(resp.status_code, 200)

    def test_04_startup_model_failure_handled_safely(self) -> None:
        """4. Startup model pre-warm failure reports degraded in health without crash."""
        with patch("phase4_api.src.main.get_model_bundle", side_effect=Exception("Disk error")):
            # Application still creates and handles degraded status gracefully
            test_app = create_app()
            with TestClient(test_app) as client:
                resp = client.get("/health")
                # Health endpoint reports degraded or returns valid JSON
                self.assertIn(resp.status_code, (200, 503))
                self.assertIn("status", resp.json())

    def test_05_router_registration_works(self) -> None:
        """5. All required routes are registered."""
        routes = list(self.app.openapi()["paths"].keys())
        self.assertIn("/health", routes)
        self.assertIn("/api/v1/model/info", routes)
        self.assertIn("/api/v1/predict/single", routes)
        self.assertIn("/api/v1/predict/batch", routes)

    # -----------------------------------------------------------------------
    # B. Single Prediction Endpoint Tests (6 - 14)
    # -----------------------------------------------------------------------
    def test_06_valid_single_prediction(self) -> None:
        """6. POST /api/v1/predict/single with valid transaction."""
        mock_booster = MockBooster(0.12)
        mock_bundle = create_mock_bundle(mock_booster)
        engine = SingleTransactionInferenceEngine(model_bundle=mock_bundle)
        self.app.dependency_overrides[get_single_engine] = lambda: engine

        resp = self.client.post("/api/v1/predict/single", json=self.sample_payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["transaction_id"], "TEST-TX-001")
        self.assertEqual(data["prediction"], "LEGIT")
        self.assertEqual(data["risk_level"], "LOW")
        self.assertAlmostEqual(data["fraud_probability"], 0.12, places=4)
        self.assertEqual(data["threshold"], FROZEN_THRESHOLD)
        self.assertGreater(data["execution_time_ms"], 0.0)

    def test_07_missing_required_feature(self) -> None:
        """7. Missing required feature rejected with HTTP 422."""
        bad_payload = dict(self.sample_payload)
        del bad_payload["income"]
        resp = self.client.post("/api/v1/predict/single", json=bad_payload)
        self.assertEqual(resp.status_code, 422)
        data = resp.json()
        self.assertIn("error", data)

    def test_08_invalid_numerical_input(self) -> None:
        """8. Non-numeric value for numerical feature rejected with HTTP 422."""
        bad_payload = dict(self.sample_payload, customer_age="not-a-number")
        resp = self.client.post("/api/v1/predict/single", json=bad_payload)
        self.assertEqual(resp.status_code, 422)
        data = resp.json()
        self.assertIn("error", data)

    def test_09_invalid_categorical_value(self) -> None:
        """9. Unseen / invalid categorical level rejected with HTTP 422."""
        bad_payload = dict(self.sample_payload, payment_type="UNKNOWN_CAT")
        resp = self.client.post("/api/v1/predict/single", json=bad_payload)
        self.assertEqual(resp.status_code, 422)
        data = resp.json()
        self.assertIn("error", data)

    def test_10_invalid_transaction_id(self) -> None:
        """10. Whitespace-only transaction_id rejected with HTTP 422."""
        bad_payload = dict(self.sample_payload, transaction_id="   ")
        resp = self.client.post("/api/v1/predict/single", json=bad_payload)
        self.assertEqual(resp.status_code, 422)

    def test_11_correct_response_schema(self) -> None:
        """11. Single prediction response contains all standardized fields."""
        mock_booster = MockBooster(0.92)
        mock_bundle = create_mock_bundle(mock_booster)
        engine = SingleTransactionInferenceEngine(model_bundle=mock_bundle)
        self.app.dependency_overrides[get_single_engine] = lambda: engine

        resp = self.client.post("/api/v1/predict/single", json=self.sample_payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        required_keys = {
            "status",
            "transaction_id",
            "prediction",
            "fraud_probability",
            "threshold",
            "risk_level",
            "model_version",
            "model_status",
            "feature_count",
            "reason_codes",
            "explanation_metadata",
            "execution_time_ms",
        }
        self.assertTrue(required_keys.issubset(data.keys()))
        self.assertEqual(data["prediction"], "FRAUD")
        self.assertEqual(data["risk_level"], "CRITICAL")

    def test_12_correct_json_serialization(self) -> None:
        """12. Response is strictly JSON-serializable."""
        mock_booster = MockBooster(0.3)
        mock_bundle = create_mock_bundle(mock_booster)
        engine = SingleTransactionInferenceEngine(model_bundle=mock_bundle)
        self.app.dependency_overrides[get_single_engine] = lambda: engine

        resp = self.client.post("/api/v1/predict/single", json=self.sample_payload)
        self.assertEqual(resp.status_code, 200)
        # Re-serializing verifies cleanliness
        re_serialized = json.dumps(resp.json())
        self.assertIn("RISK_LEVEL_MEDIUM", re_serialized)

    def test_13_inference_engine_exception(self) -> None:
        """13. Internal prediction exception returns HTTP 500 without leaking stack trace."""
        mock_booster = MockBooster(RuntimeError("C++ booster out of memory"))
        mock_bundle = create_mock_bundle(mock_booster)
        engine = SingleTransactionInferenceEngine(model_bundle=mock_bundle)
        self.app.dependency_overrides[get_single_engine] = lambda: engine

        resp = self.client.post("/api/v1/predict/single", json=self.sample_payload)
        self.assertEqual(resp.status_code, 500)
        data = resp.json()
        self.assertIn("error", data)
        self.assertNotIn("Traceback", json.dumps(data))

    def test_14_model_unavailable(self) -> None:
        """14. Model loading failure returns HTTP 503."""
        def broken_engine() -> SingleTransactionInferenceEngine:
            raise ModelNotReadyError("Model artifact corrupted or missing")

        self.app.dependency_overrides[get_single_engine] = broken_engine
        resp = self.client.post("/api/v1/predict/single", json=self.sample_payload)
        self.assertEqual(resp.status_code, 503)
        data = resp.json()
        self.assertEqual(data["error"]["code"], "SERVICE_UNAVAILABLE")

    # -----------------------------------------------------------------------
    # C. Batch Prediction Endpoint Tests (15 - 26)
    # -----------------------------------------------------------------------
    def test_15_valid_batch_request(self) -> None:
        """15. Valid batch request returns HTTP 200."""
        mock_booster = MockBooster(0.15)
        mock_bundle = create_mock_bundle(mock_booster)
        engine = BatchTransactionInferenceEngine(model_bundle=mock_bundle)
        self.app.dependency_overrides[get_batch_engine] = lambda: engine

        batch_payload = {"transactions": [self.sample_payload]}
        resp = self.client.post("/api/v1/predict/batch", json=batch_payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["batch_status"], "all_success")
        self.assertEqual(data["total_transactions"], 1)

    def test_16_multiple_transactions_batch(self) -> None:
        """16. Multiple transactions in batch processed accurately."""
        mock_booster = MockBooster(0.15)
        mock_bundle = create_mock_bundle(mock_booster)
        engine = BatchTransactionInferenceEngine(model_bundle=mock_bundle)
        self.app.dependency_overrides[get_batch_engine] = lambda: engine

        records = [dict(self.sample_payload, transaction_id=f"TX-{i}") for i in range(5)]
        resp = self.client.post("/api/v1/predict/batch", json={"transactions": records})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total_transactions"], 5)
        self.assertEqual(data["successful_transactions"], 5)
        self.assertEqual(len(data["results"]), 5)

    def test_17_empty_batch_rejected(self) -> None:
        """17. Empty transactions list rejected with HTTP 422."""
        resp = self.client.post("/api/v1/predict/batch", json={"transactions": []})
        self.assertEqual(resp.status_code, 422)

    def test_18_invalid_batch_structure(self) -> None:
        """18. Non-list transactions rejected with HTTP 422."""
        resp = self.client.post("/api/v1/predict/batch", json={"transactions": "not-a-list"})
        self.assertEqual(resp.status_code, 422)

    def test_19_oversized_batch_rejected(self) -> None:
        """19. Batch exceeding max_batch_size rejected with HTTP 413."""
        mock_booster = MockBooster(0.2)
        mock_bundle = create_mock_bundle(mock_booster)
        engine = BatchTransactionInferenceEngine(model_bundle=mock_bundle, max_batch_size=3)
        self.app.dependency_overrides[get_batch_engine] = lambda: engine

        records = [dict(self.sample_payload, transaction_id=f"TX-{i}") for i in range(5)]
        resp = self.client.post("/api/v1/predict/batch", json={"transactions": records})
        self.assertEqual(resp.status_code, 413)
        data = resp.json()
        self.assertEqual(data["error"]["code"], "BATCH_SIZE_LIMIT_EXCEEDED")

    def test_20_partial_success_response(self) -> None:
        """20. Partial success: valid records predicted, invalid records hold errors."""
        mock_booster = MockBooster(0.7)
        mock_bundle = create_mock_bundle(mock_booster)
        engine = BatchTransactionInferenceEngine(model_bundle=mock_bundle)
        self.app.dependency_overrides[get_batch_engine] = lambda: engine

        good_rec = dict(self.sample_payload, transaction_id="GOOD")
        bad_rec = {"transaction_id": "BAD", "income": 100}  # Missing 28 features

        resp = self.client.post("/api/v1/predict/batch", json={"transactions": [good_rec, bad_rec]})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["batch_status"], "partial_success")
        self.assertEqual(data["successful_transactions"], 1)
        self.assertEqual(data["failed_transactions"], 1)

        # Index 0: success -> HIGH risk
        self.assertEqual(data["results"][0]["status"], "success")
        self.assertEqual(data["results"][0]["risk_level"], "HIGH")

        # Index 1: error -> no risk level
        self.assertEqual(data["results"][1]["status"], "error")
        self.assertIsNone(data["results"][1]["risk_level"])

    def test_21_all_invalid_batch_behavior(self) -> None:
        """21. All-invalid batch returns HTTP 200 with batch_status='all_failed'."""
        mock_booster = MockBooster(0.5)
        mock_bundle = create_mock_bundle(mock_booster)
        engine = BatchTransactionInferenceEngine(model_bundle=mock_bundle)
        self.app.dependency_overrides[get_batch_engine] = lambda: engine

        bad_records = [{"bad": 1}, {"bad": 2}]
        resp = self.client.post("/api/v1/predict/batch", json={"transactions": bad_records})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["batch_status"], "all_failed")
        self.assertEqual(data["successful_transactions"], 0)
        self.assertEqual(data["failed_transactions"], 2)

    def test_22_input_order_preservation(self) -> None:
        """22. Original input order and input_index strictly preserved."""
        mock_booster = MockBooster(0.2)
        mock_bundle = create_mock_bundle(mock_booster)
        engine = BatchTransactionInferenceEngine(model_bundle=mock_bundle)
        self.app.dependency_overrides[get_batch_engine] = lambda: engine

        records = [
            dict(self.sample_payload, transaction_id="REC-0"),
            {"invalid": True},
            dict(self.sample_payload, transaction_id="REC-2"),
        ]
        resp = self.client.post("/api/v1/predict/batch", json={"transactions": records})
        self.assertEqual(resp.status_code, 200)
        results = resp.json()["results"]
        self.assertEqual(results[0]["input_index"], 0)
        self.assertEqual(results[0]["transaction_id"], "REC-0")
        self.assertEqual(results[1]["input_index"], 1)
        self.assertEqual(results[1]["status"], "error")
        self.assertEqual(results[2]["input_index"], 2)
        self.assertEqual(results[2]["transaction_id"], "REC-2")

    def test_23_transaction_id_preservation(self) -> None:
        """23. Transaction IDs preserved accurately."""
        mock_booster = MockBooster(0.3)
        mock_bundle = create_mock_bundle(mock_booster)
        engine = BatchTransactionInferenceEngine(model_bundle=mock_bundle)
        self.app.dependency_overrides[get_batch_engine] = lambda: engine

        records = [dict(self.sample_payload, transaction_id="CUSTOM-TID-999")]
        resp = self.client.post("/api/v1/predict/batch", json={"transactions": records})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["results"][0]["transaction_id"], "CUSTOM-TID-999")

    def test_24_correct_batch_counts(self) -> None:
        """24. Aggregate counts match arithmetic sum of results."""
        mock_booster = MockBooster(0.4)
        mock_bundle = create_mock_bundle(mock_booster)
        engine = BatchTransactionInferenceEngine(model_bundle=mock_bundle)
        self.app.dependency_overrides[get_batch_engine] = lambda: engine

        records = [self.sample_payload, {"bad": 1}, self.sample_payload, {"bad": 2}]
        resp = self.client.post("/api/v1/predict/batch", json={"transactions": records})
        data = resp.json()
        self.assertEqual(data["total_transactions"], 4)
        self.assertEqual(data["successful_transactions"], 2)
        self.assertEqual(data["failed_transactions"], 2)

    def test_25_and_26_vectorized_booster_called_once(self) -> None:
        """25 & 26. Vectorized booster is called exactly once for batch, not in a loop."""
        mock_booster = MockBooster(0.2)
        mock_bundle = create_mock_bundle(mock_booster)
        engine = BatchTransactionInferenceEngine(model_bundle=mock_bundle)
        self.app.dependency_overrides[get_batch_engine] = lambda: engine

        records = [dict(self.sample_payload, transaction_id=f"TX-{i}") for i in range(10)]
        resp = self.client.post("/api/v1/predict/batch", json={"transactions": records})
        self.assertEqual(resp.status_code, 200)
        # Booster predict called exactly once for all 10 records
        self.assertEqual(mock_booster.call_count, 1)

    # -----------------------------------------------------------------------
    # D. Health Endpoint Tests (27 - 30)
    # -----------------------------------------------------------------------
    def test_27_health_api_running(self) -> None:
        """27. GET /health returns HTTP 200."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("status", data)
        self.assertIn("uptime_seconds", data)

    def test_28_health_model_loaded(self) -> None:
        """28. GET /health reports model_loaded=True when booster active."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["model_loaded"])
        self.assertTrue(data["ready"])

    def test_29_health_model_unavailable(self) -> None:
        """29. GET /health reports degraded when booster is unavailable."""
        broken_engine = MagicMock(spec=SingleTransactionInferenceEngine)
        broken_engine.booster = None
        broken_engine.threshold = 0.0
        broken_engine.model_version = "v3"
        self.app.dependency_overrides[get_single_engine] = lambda: broken_engine

        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "degraded")
        self.assertFalse(data["ready"])

    def test_30_correct_readiness_behavior(self) -> None:
        """30. Readiness requires both model residency and valid threshold."""
        resp = self.client.get("/health")
        data = resp.json()
        self.assertEqual(data["ready"], data["model_loaded"] and data["frozen_threshold"] > 0)

    # -----------------------------------------------------------------------
    # E. Model Information Tests (31 - 35)
    # -----------------------------------------------------------------------
    def test_31_model_info_version(self) -> None:
        """31. GET /api/v1/model/info returns correct model version ('v3')."""
        resp = self.client.get("/api/v1/model/info")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["model_version"], "v3")

    def test_32_model_info_feature_count(self) -> None:
        """32. Feature count is exactly 29."""
        resp = self.client.get("/api/v1/model/info")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["feature_count"], 29)

    def test_33_model_info_frozen_threshold(self) -> None:
        """33. Frozen threshold is exactly 0.8964321608040201."""
        resp = self.client.get("/api/v1/model/info")
        self.assertEqual(resp.status_code, 200)
        self.assertAlmostEqual(resp.json()["frozen_threshold"], FROZEN_THRESHOLD, places=5)

    def test_34_model_info_supported_modes(self) -> None:
        """34. Supported inference modes include single and batch."""
        resp = self.client.get("/api/v1/model/info")
        self.assertEqual(resp.status_code, 200)
        modes = resp.json()["supported_inference_modes"]
        self.assertIn("single", modes)
        self.assertIn("batch", modes)

    def test_35_model_info_no_sensitive_paths_exposed(self) -> None:
        """35. No absolute local filesystem paths or secrets exposed in info payload."""
        resp = self.client.get("/api/v1/model/info")
        self.assertEqual(resp.status_code, 200)
        raw_text = resp.text
        self.assertNotIn("D:\\", raw_text)
        self.assertNotIn("C:\\", raw_text)
        self.assertNotIn("/home/", raw_text)
        self.assertNotIn("password", raw_text.lower())
        self.assertNotIn("secret", raw_text.lower())

    # -----------------------------------------------------------------------
    # F. Centralized Error Handling Tests (36 - 39)
    # -----------------------------------------------------------------------
    def test_36_structured_validation_errors(self) -> None:
        """36. Structured error envelope contains code and message."""
        resp = self.client.post("/api/v1/predict/single", json={})
        self.assertEqual(resp.status_code, 422)
        data = resp.json()
        self.assertIn("error", data)
        self.assertIn("code", data["error"])
        self.assertIn("message", data["error"])

    def test_37_structured_model_errors(self) -> None:
        """37. Model loading exceptions return structured 503."""
        def raise_loader_error() -> SingleTransactionInferenceEngine:
            raise ModelLoaderError("Artifact corrupted")

        self.app.dependency_overrides[get_single_engine] = raise_loader_error
        resp = self.client.post("/api/v1/predict/single", json=self.sample_payload)
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.json()["error"]["code"], "SERVICE_UNAVAILABLE")

    def test_38_internal_errors_do_not_leak_stack_traces(self) -> None:
        """38. Unhandled exceptions return 500 without stack traces."""
        def unhandled_error() -> SingleTransactionInferenceEngine:
            raise ArithmeticError("Unexpected division by zero in module XYZ")

        self.app.dependency_overrides[get_single_engine] = unhandled_error
        unhandled_client = TestClient(self.app, raise_server_exceptions=False)
        resp = unhandled_client.post("/api/v1/predict/single", json=self.sample_payload)
        self.assertEqual(resp.status_code, 500)
        data = resp.json()
        self.assertEqual(data["error"]["code"], "INTERNAL_SERVER_ERROR")
        self.assertNotIn("Traceback", resp.text)
        self.assertNotIn("ArithmeticError", resp.text)

    def test_39_error_responses_are_json_serializable(self) -> None:
        """39. Error responses are valid JSON."""
        resp = self.client.post("/api/v1/predict/single", json={"invalid": "payload"})
        parsed = json.loads(resp.text)
        self.assertIsInstance(parsed, dict)

    # -----------------------------------------------------------------------
    # G. OpenAPI Documentation Tests (40 - 44)
    # -----------------------------------------------------------------------
    def test_40_swagger_docs_accessible(self) -> None:
        """40. GET /docs returns HTTP 200."""
        resp = self.client.get("/docs")
        self.assertEqual(resp.status_code, 200)

    def test_41_redoc_accessible(self) -> None:
        """41. GET /redoc returns HTTP 200."""
        resp = self.client.get("/redoc")
        self.assertEqual(resp.status_code, 200)

    def test_42_openapi_json_accessible(self) -> None:
        """42. GET /openapi.json returns valid schema."""
        resp = self.client.get("/openapi.json")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("openapi", data)
        self.assertIn("paths", data)

    def test_43_required_routes_appear_in_openapi(self) -> None:
        """43. Required API routes appear in OpenAPI schema."""
        resp = self.client.get("/openapi.json")
        paths = resp.json()["paths"]
        self.assertIn("/health", paths)
        self.assertIn("/api/v1/model/info", paths)
        self.assertIn("/api/v1/predict/single", paths)
        self.assertIn("/api/v1/predict/batch", paths)

    def test_44_request_schemas_appear_correctly_in_openapi(self) -> None:
        """44. Request schemas documented in OpenAPI components."""
        resp = self.client.get("/openapi.json")
        schemas = resp.json().get("components", {}).get("schemas", {})
        self.assertIn("SingleTransactionRequest", schemas)
        self.assertIn("BatchTransactionRequest", schemas)

    # -----------------------------------------------------------------------
    # H. Internal Integration Tests (45 - 48)
    # -----------------------------------------------------------------------
    def test_45_single_api_reaches_phase4_engine(self) -> None:
        """45. POST /api/v1/predict/single invokes SingleTransactionInferenceEngine."""
        mock_booster = MockBooster(0.8)
        mock_bundle = create_mock_bundle(mock_booster)
        engine = SingleTransactionInferenceEngine(model_bundle=mock_bundle)
        self.app.dependency_overrides[get_single_engine] = lambda: engine

        resp = self.client.post("/api/v1/predict/single", json=self.sample_payload)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(mock_booster.call_count, 1)

    def test_46_batch_api_reaches_phase5_engine(self) -> None:
        """46. POST /api/v1/predict/batch invokes BatchTransactionInferenceEngine."""
        mock_booster = MockBooster(0.8)
        mock_bundle = create_mock_bundle(mock_booster)
        engine = BatchTransactionInferenceEngine(model_bundle=mock_bundle)
        self.app.dependency_overrides[get_batch_engine] = lambda: engine

        resp = self.client.post("/api/v1/predict/batch", json={"transactions": [self.sample_payload]})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(mock_booster.call_count, 1)

    def test_47_phase6_risk_classification_appears_in_api(self) -> None:
        """47. Phase 6 risk classification, reason codes, and metadata appear in API results."""
        mock_booster = MockBooster(0.75)  # HIGH risk, but LEGIT
        mock_bundle = create_mock_bundle(mock_booster)
        engine = SingleTransactionInferenceEngine(model_bundle=mock_bundle)
        self.app.dependency_overrides[get_single_engine] = lambda: engine

        resp = self.client.post("/api/v1/predict/single", json=self.sample_payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["prediction"], "LEGIT")
        self.assertEqual(data["risk_level"], "HIGH")
        codes = [rc["code"] for rc in data["reason_codes"]]
        self.assertIn("RISK_LEVEL_HIGH", codes)
        self.assertIn("MODEL_DECISION_LEGIT", codes)
        self.assertIn("FEATURE_EXPLANATION_UNAVAILABLE", codes)

    def test_48_single_and_batch_api_outputs_consistent(self) -> None:
        """48. Single and batch endpoints yield identical predictions and risk classifications."""
        mock_booster = MockBooster(0.42)
        mock_bundle = create_mock_bundle(mock_booster)
        single_eng = SingleTransactionInferenceEngine(model_bundle=mock_bundle)
        batch_eng = BatchTransactionInferenceEngine(model_bundle=mock_bundle)
        self.app.dependency_overrides[get_single_engine] = lambda: single_eng
        self.app.dependency_overrides[get_batch_engine] = lambda: batch_eng

        s_resp = self.client.post("/api/v1/predict/single", json=self.sample_payload)
        b_resp = self.client.post("/api/v1/predict/batch", json={"transactions": [self.sample_payload]})

        s_data = s_resp.json()
        b_item = b_resp.json()["results"][0]

        self.assertEqual(s_data["prediction"], b_item["prediction"])
        self.assertAlmostEqual(s_data["fraud_probability"], b_item["fraud_probability"], places=5)
        self.assertEqual(s_data["risk_level"], b_item["risk_level"])
        self.assertEqual(s_data["threshold"], b_item["threshold"])

    # -----------------------------------------------------------------------
    # I. Real-Model Smoke Test (49)
    # -----------------------------------------------------------------------
    def test_49_real_model_smoke_test(self) -> None:
        """49. End-to-end live smoke test using real LightGBM v3 model via FastAPI."""
        # Use real singleton engines without dependency overrides
        with TestClient(self.app) as real_client:
            # 1. Health check
            h_resp = real_client.get("/health")
            self.assertEqual(h_resp.status_code, 200)
            h_data = h_resp.json()
            self.assertEqual(h_data["status"], "healthy")
            self.assertTrue(h_data["ready"])
            self.assertEqual(h_data["model_version"], "v3")

            # 2. Model info check
            m_resp = real_client.get("/api/v1/model/info")
            self.assertEqual(m_resp.status_code, 200)
            m_data = m_resp.json()
            self.assertEqual(m_data["model_version"], "v3")
            self.assertEqual(m_data["feature_count"], 29)
            self.assertAlmostEqual(m_data["frozen_threshold"], FROZEN_THRESHOLD, places=5)

            # 3. Single prediction check
            single_payload = dict(self.sample_tx, transaction_id="REAL-SMOKE-HTTP-1")
            s_resp = real_client.post("/api/v1/predict/single", json=single_payload)
            self.assertEqual(s_resp.status_code, 200)
            s_data = s_resp.json()
            self.assertEqual(s_data["status"], "SUCCESS")
            self.assertEqual(s_data["transaction_id"], "REAL-SMOKE-HTTP-1")
            self.assertIn(s_data["prediction"], ("FRAUD", "LEGIT"))
            self.assertIn(s_data["risk_level"], ("LOW", "MEDIUM", "HIGH", "CRITICAL"))
            self.assertAlmostEqual(s_data["threshold"], FROZEN_THRESHOLD, places=5)
            self.assertTrue(len(s_data["reason_codes"]) >= 3)

            # 4. Batch prediction check
            batch_payload = {
                "transactions": [
                    dict(self.sample_tx, transaction_id="REAL-SMOKE-BATCH-1"),
                    dict(self.sample_tx, transaction_id="REAL-SMOKE-BATCH-2"),
                ]
            }
            b_resp = real_client.post("/api/v1/predict/batch", json=batch_payload)
            self.assertEqual(b_resp.status_code, 200)
            b_data = b_resp.json()
            self.assertEqual(b_data["batch_status"], "all_success")
            self.assertEqual(b_data["total_transactions"], 2)
            self.assertEqual(b_data["successful_transactions"], 2)


if __name__ == "__main__":
    unittest.main()
