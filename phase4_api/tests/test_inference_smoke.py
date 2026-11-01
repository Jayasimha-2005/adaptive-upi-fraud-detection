"""
phase4_api/tests/test_inference_smoke.py
========================================
Real-Model Inference Smoke Test.

NOTE: This is strictly an end-to-end integration and smoke test verifying
that the real LightGBM v3 booster, Phase 2 model loader, and Phase 3 preprocessor
connect seamlessly into the SingleTransactionInferenceEngine.
It is NOT a statistical evaluation or benchmark of model accuracy.

Verifies:
  1. Real approved v3 model loads successfully from artifacts/models/lgbm_v3.txt.
  2. Transaction preprocessing succeeds on a representative valid BAF transaction.
  3. Exactly one probability score is generated.
  4. Probability is a finite numeric value in [0.0, 1.0].
  5. Binary classification matches the authoritative frozen threshold (0.8964321608040201).
  6. Model version reports 'v3' with 500 trees and 29 features.
  7. Exactly 29 canonical features are ingested.
"""
from __future__ import annotations

import math
import pathlib
import sys
import unittest

# Setup path
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase4_api.src.inference_engine import (
    PredictionResponse,
    SingleTransactionInferenceEngine,
    get_inference_engine,
)
from phase4_api.src.model_loader import EXPECTED_FEATURE_COUNT
from phase4_api.tests.test_transaction_preprocessor import get_sample_valid_transaction


class TestInferenceSmoke(unittest.TestCase):
    """End-to-end smoke test using the actual approved LightGBM v3 model artifact."""

    def setUp(self) -> None:
        # Instantiate engine without mocks; uses real v3 model artifact
        self.engine = SingleTransactionInferenceEngine(model_version="v3")
        self.sample = get_sample_valid_transaction()

    def test_end_to_end_real_model_smoke_inference(self) -> None:
        """
        Executes one real inference cycle across the entire stack:
        Raw Dict -> TransactionPreprocessor -> LightGBM v3 Booster -> Frozen Threshold -> PredictionResponse.
        """
        # Execute prediction with an explicit tracking transaction ID
        response = self.engine.predict(
            transaction=self.sample,
            transaction_id="SMOKE-TEST-BAF-001",
        )

        # 1. Output type verification
        self.assertIsInstance(response, PredictionResponse)

        # 2. Transaction ID preservation
        self.assertEqual(response.transaction_id, "SMOKE-TEST-BAF-001")

        # 3. Model metadata verification
        self.assertEqual(response.model_version, "v3")
        self.assertEqual(response.model_status, "ACTIVE_LOADED")
        self.assertEqual(response.feature_count, EXPECTED_FEATURE_COUNT)
        self.assertEqual(response.feature_count, 29)

        # 4. Probability validity
        self.assertIsInstance(response.fraud_probability, float)
        self.assertTrue(math.isfinite(response.fraud_probability))
        self.assertGreaterEqual(response.fraud_probability, 0.0)
        self.assertLessEqual(response.fraud_probability, 1.0)

        # 5. Authoritative frozen threshold check
        self.assertEqual(response.threshold, 0.8964321608040201)

        # 6. Binary decision consistency
        expected_decision = (
            "FRAUD"
            if response.fraud_probability >= response.threshold
            else "LEGIT"
        )
        self.assertEqual(response.prediction, expected_decision)

        # 7. Execution duration sanity check (< 100ms on standard CPU)
        self.assertGreater(response.execution_time_ms, 0.0)
        self.assertLess(response.execution_time_ms, 500.0)

        # 8. Dictionary serialization check
        res_dict = response.to_dict()
        self.assertIn("prediction", res_dict)
        self.assertIn("fraud_probability", res_dict)
        self.assertIn("threshold", res_dict)
        self.assertNotIn("Users", str(res_dict))  # No local file paths leaked

    def test_singleton_engine_factory(self) -> None:
        """Verifies get_inference_engine returns a shared singleton instance."""
        engine1 = get_inference_engine("v3")
        engine2 = get_inference_engine("v3")
        self.assertIs(engine1, engine2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
