"""
phase4_api/tests/test_inference_engine.py
=========================================
Comprehensive unit test suite for SingleTransactionInferenceEngine.

Covers all 25 required test cases:
  1. Valid transaction produces a prediction.
  2. Model version is correctly returned.
  3. Fraud probability is between 0 and 1.
  4. Prediction uses authoritative frozen threshold.
  5. Transaction ID is preserved.
  6. Missing transaction ID is supported.
  7. Feature count remains 29.
  8. Model is not reloaded for every inference.
  9. Probability below threshold returns LEGIT.
  10. Probability equal to threshold returns FRAUD.
  11. Probability above threshold returns FRAUD.
  12. Missing features are rejected.
  13. Unexpected features are rejected.
  14. Invalid numerical values are rejected.
  15. Invalid categorical values are rejected.
  16. Model initialization failure is handled.
  17. Model prediction exception is handled.
  18. Empty prediction output is rejected.
  19. Multiple prediction outputs are rejected.
  20. NaN probability is rejected.
  21. Infinite probability is rejected.
  22. Probability below 0 is rejected.
  23. Probability above 1 is rejected.
  24. Invalid threshold is rejected.
  25. Preprocessor returning an incorrect shape is rejected.
"""
from __future__ import annotations

import pathlib
import sys
from typing import Any
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# Setup path
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase4_api.src.inference_engine import (
    InferenceEngineError,
    InvalidModelOutputError,
    InvalidTransactionIdError,
    ModelNotReadyError,
    ModelPredictionError,
    PredictionResponse,
    SingleTransactionInferenceEngine,
)
from phase4_api.src.model_loader import (
    EXPECTED_FEATURE_COUNT,
    LoadedModelBundle,
    ModelInfo,
    ThresholdConfigurationError,
    get_model_bundle,
)
from phase4_api.src.transaction_preprocessor import (
    ExtraFeatureError,
    InvalidCategoricalValueError,
    InvalidNumericalValueError,
    MissingFeatureError,
    PreprocessedOutputValidationError,
    TransactionPreprocessor,
)
from phase4_api.tests.test_transaction_preprocessor import get_sample_valid_transaction


class MockBooster:
    """Mock LightGBM Booster returning controlled predictions for testing."""

    def __init__(self, return_val: Any = 0.5) -> None:
        self.return_val = return_val
        self.call_count = 0

    def predict(self, df: pd.DataFrame) -> Any:
        self.call_count += 1
        if isinstance(self.return_val, Exception):
            raise self.return_val
        return self.return_val


def create_mock_bundle(
    booster: Any,
    threshold: float = 0.8964321608040201,
    version: str = "v3",
) -> LoadedModelBundle:
    """Helper creating a test LoadedModelBundle with a mock booster."""
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


class TestInferenceEngine(unittest.TestCase):
    """Unit test suite for SingleTransactionInferenceEngine."""

    def setUp(self) -> None:
        self.sample = get_sample_valid_transaction()
        self.mock_booster = MockBooster(return_val=np.array([0.45]))
        self.bundle = create_mock_bundle(self.mock_booster)
        self.engine = SingleTransactionInferenceEngine(
            model_version="v3",
            model_bundle=self.bundle,
        )

    # -------------------------------------------------------------------------
    # Test 1: Valid transaction produces a prediction
    # -------------------------------------------------------------------------
    def test_01_valid_transaction_produces_prediction(self) -> None:
        resp = self.engine.predict(self.sample, transaction_id="TX-1001")
        self.assertIsInstance(resp, PredictionResponse)
        self.assertEqual(resp.prediction, "LEGIT")
        self.assertEqual(resp.model_status, "ACTIVE_LOADED")
        self.assertGreater(resp.execution_time_ms, 0.0)

    # -------------------------------------------------------------------------
    # Test 2: Model version correctly returned
    # -------------------------------------------------------------------------
    def test_02_model_version_correctly_returned(self) -> None:
        resp = self.engine.predict(self.sample)
        self.assertEqual(resp.model_version, "v3")

    # -------------------------------------------------------------------------
    # Test 3: Fraud probability is between 0 and 1
    # -------------------------------------------------------------------------
    def test_03_fraud_probability_between_0_and_1(self) -> None:
        resp = self.engine.predict(self.sample)
        self.assertGreaterEqual(resp.fraud_probability, 0.0)
        self.assertLessEqual(resp.fraud_probability, 1.0)
        self.assertEqual(resp.fraud_probability, 0.45)

    # -------------------------------------------------------------------------
    # Test 4: Prediction uses authoritative frozen threshold
    # -------------------------------------------------------------------------
    def test_04_prediction_uses_frozen_threshold(self) -> None:
        resp = self.engine.predict(self.sample)
        self.assertEqual(resp.threshold, 0.8964321608040201)

    # -------------------------------------------------------------------------
    # Test 5: Transaction ID is preserved
    # -------------------------------------------------------------------------
    def test_05_transaction_id_preserved(self) -> None:
        resp = self.engine.predict(self.sample, transaction_id="BAF-TXN-998877")
        self.assertEqual(resp.transaction_id, "BAF-TXN-998877")

    # -------------------------------------------------------------------------
    # Test 6: Missing transaction ID is supported (None)
    # -------------------------------------------------------------------------
    def test_06_missing_transaction_id_supported(self) -> None:
        resp = self.engine.predict(self.sample)
        self.assertIsNone(resp.transaction_id)

        # Invalid transaction IDs
        with self.assertRaises(InvalidTransactionIdError):
            self.engine.predict(self.sample, transaction_id="")
        with self.assertRaises(InvalidTransactionIdError):
            self.engine.predict(self.sample, transaction_id="   ")
        with self.assertRaises(InvalidTransactionIdError):
            self.engine.predict(self.sample, transaction_id="bad\x00id")
        with self.assertRaises(InvalidTransactionIdError):
            self.engine.predict(self.sample, transaction_id="A" * 129)
        with self.assertRaises(InvalidTransactionIdError):
            self.engine.predict(self.sample, transaction_id=12345)  # type: ignore[arg-type]

    # -------------------------------------------------------------------------
    # Test 7: Feature count remains 29
    # -------------------------------------------------------------------------
    def test_07_feature_count_remains_29(self) -> None:
        resp = self.engine.predict(self.sample)
        self.assertEqual(resp.feature_count, 29)

    # -------------------------------------------------------------------------
    # Test 8: Model is not reloaded for every inference
    # -------------------------------------------------------------------------
    def test_08_model_not_reloaded_per_inference(self) -> None:
        initial_bundle = self.engine.bundle
        resp1 = self.engine.predict(self.sample)
        resp2 = self.engine.predict(self.sample)
        self.assertIs(self.engine.bundle, initial_bundle)
        self.assertEqual(self.mock_booster.call_count, 2)

    # -------------------------------------------------------------------------
    # Test 9: Probability below threshold returns LEGIT
    # -------------------------------------------------------------------------
    def test_09_probability_below_threshold_returns_legit(self) -> None:
        self.mock_booster.return_val = np.array([0.896430])  # Just below 0.896432
        resp = self.engine.predict(self.sample)
        self.assertEqual(resp.prediction, "LEGIT")

    # -------------------------------------------------------------------------
    # Test 10: Probability equal to threshold returns FRAUD
    # -------------------------------------------------------------------------
    def test_10_probability_equal_to_threshold_returns_fraud(self) -> None:
        self.mock_booster.return_val = np.array([0.8964321608040201])  # Exact threshold
        resp = self.engine.predict(self.sample)
        self.assertEqual(resp.prediction, "FRAUD")

    # -------------------------------------------------------------------------
    # Test 11: Probability above threshold returns FRAUD
    # -------------------------------------------------------------------------
    def test_11_probability_above_threshold_returns_fraud(self) -> None:
        self.mock_booster.return_val = np.array([0.950000])  # Clear fraud
        resp = self.engine.predict(self.sample)
        self.assertEqual(resp.prediction, "FRAUD")

    # -------------------------------------------------------------------------
    # Test 12: Missing features are rejected
    # -------------------------------------------------------------------------
    def test_12_missing_features_rejected(self) -> None:
        bad_sample = dict(self.sample)
        del bad_sample["income"]
        with self.assertRaises(MissingFeatureError):
            self.engine.predict(bad_sample)

    # -------------------------------------------------------------------------
    # Test 13: Unexpected features are rejected
    # -------------------------------------------------------------------------
    def test_13_unexpected_features_rejected(self) -> None:
        extra_sample = dict(self.sample)
        extra_sample["device_fraud_count"] = 0
        with self.assertRaises(ExtraFeatureError):
            self.engine.predict(extra_sample)

    # -------------------------------------------------------------------------
    # Test 14: Invalid numerical values are rejected
    # -------------------------------------------------------------------------
    def test_14_invalid_numerical_values_rejected(self) -> None:
        bad_num = dict(self.sample)
        bad_num["velocity_6h"] = "not_numeric"
        with self.assertRaises(InvalidNumericalValueError):
            self.engine.predict(bad_num)

    # -------------------------------------------------------------------------
    # Test 15: Invalid categorical values are rejected
    # -------------------------------------------------------------------------
    def test_15_invalid_categorical_values_rejected(self) -> None:
        bad_cat = dict(self.sample)
        bad_cat["housing_status"] = "INVALID_HOUSING_CAT"
        with self.assertRaises(InvalidCategoricalValueError):
            self.engine.predict(bad_cat)

    # -------------------------------------------------------------------------
    # Test 16: Model initialization failure is handled
    # -------------------------------------------------------------------------
    def test_16_model_initialization_failure_handled(self) -> None:
        with patch("phase4_api.src.inference_engine.get_model_bundle") as mock_get:
            mock_get.side_effect = RuntimeError("Disk IO failure")
            with self.assertRaises(ModelNotReadyError) as ctx:
                SingleTransactionInferenceEngine(model_version="v3")
            self.assertIn("Failed to initialize model", str(ctx.exception))

    # -------------------------------------------------------------------------
    # Test 17: Model prediction exception is handled
    # -------------------------------------------------------------------------
    def test_17_model_prediction_exception_handled(self) -> None:
        self.mock_booster.return_val = RuntimeError("C++ booster segmentation fault")
        with self.assertRaises(ModelPredictionError) as ctx:
            self.engine.predict(self.sample)
        self.assertIn("LightGBM prediction failed", str(ctx.exception))

    # -------------------------------------------------------------------------
    # Test 18: Empty prediction output is rejected
    # -------------------------------------------------------------------------
    def test_18_empty_prediction_output_rejected(self) -> None:
        self.mock_booster.return_val = np.array([])
        with self.assertRaises(InvalidModelOutputError) as ctx:
            self.engine.predict(self.sample)
        self.assertIn("empty prediction output", str(ctx.exception).lower())

    # -------------------------------------------------------------------------
    # Test 19: Multiple prediction outputs rejected
    # -------------------------------------------------------------------------
    def test_19_multiple_prediction_outputs_rejected(self) -> None:
        self.mock_booster.return_val = np.array([0.3, 0.7])
        with self.assertRaises(InvalidModelOutputError) as ctx:
            self.engine.predict(self.sample)
        self.assertIn("expected exactly 1 prediction output", str(ctx.exception).lower())

    # -------------------------------------------------------------------------
    # Test 20: NaN probability is rejected
    # -------------------------------------------------------------------------
    def test_20_nan_probability_rejected(self) -> None:
        self.mock_booster.return_val = np.array([np.nan])
        with self.assertRaises(InvalidModelOutputError) as ctx:
            self.engine.predict(self.sample)
        self.assertIn("non-finite probability value", str(ctx.exception).lower())

    # -------------------------------------------------------------------------
    # Test 21: Infinite probability is rejected
    # -------------------------------------------------------------------------
    def test_21_infinite_probability_rejected(self) -> None:
        self.mock_booster.return_val = np.array([np.inf])
        with self.assertRaises(InvalidModelOutputError) as ctx:
            self.engine.predict(self.sample)
        self.assertIn("non-finite probability value", str(ctx.exception).lower())

    # -------------------------------------------------------------------------
    # Test 22: Probability below 0 is rejected
    # -------------------------------------------------------------------------
    def test_22_probability_below_0_rejected(self) -> None:
        self.mock_booster.return_val = np.array([-0.05])
        with self.assertRaises(InvalidModelOutputError) as ctx:
            self.engine.predict(self.sample)
        self.assertIn("outside valid range [0.0, 1.0]", str(ctx.exception))

    # -------------------------------------------------------------------------
    # Test 23: Probability above 1 is rejected
    # -------------------------------------------------------------------------
    def test_23_probability_above_1_rejected(self) -> None:
        self.mock_booster.return_val = np.array([1.05])
        with self.assertRaises(InvalidModelOutputError) as ctx:
            self.engine.predict(self.sample)
        self.assertIn("outside valid range [0.0, 1.0]", str(ctx.exception))

    # -------------------------------------------------------------------------
    # Test 24: Invalid threshold is rejected at engine initialization
    # -------------------------------------------------------------------------
    def test_24_invalid_threshold_rejected(self) -> None:
        bad_bundle = create_mock_bundle(self.mock_booster, threshold=1.5)
        with self.assertRaises(ThresholdConfigurationError):
            SingleTransactionInferenceEngine(model_bundle=bad_bundle)

    # -------------------------------------------------------------------------
    # Test 25: Preprocessor returning an incorrect shape is rejected
    # -------------------------------------------------------------------------
    def test_25_incorrect_preprocessor_shape_rejected(self) -> None:
        class MockBadPreprocessor:
            def transform(self, record: Any) -> pd.DataFrame:
                # Return 2 rows instead of 1
                return pd.DataFrame([{"col": 1}, {"col": 2}])

        bad_engine = SingleTransactionInferenceEngine(
            model_bundle=self.bundle,
            preprocessor=MockBadPreprocessor(),  # type: ignore[arg-type]
        )
        with self.assertRaises(PreprocessedOutputValidationError) as ctx:
            bad_engine.predict(self.sample)
        self.assertIn("shape invalid", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
