"""
phase4_api/tests/test_risk_classification.py
============================================
Comprehensive Unit Test Suite for Phase 6: Prediction Result & Risk Classification.

Covers all 33 required test scenarios:
  Result Schema:
    1. Valid result creation.
    2. Required field validation.
    3. Probability range validation.
    4. Threshold validation.
    5. Model version validation.
    6. Missing optional fields handled properly.
    7. JSON serialization round-trip.
    8. Invalid data rejection.
  Risk Classification:
    9. LOW classification.
    10. MEDIUM classification.
    11. HIGH classification.
    12. CRITICAL classification.
    13. Exact boundary behavior.
    14. Invalid probability rejection (NaN, Inf, <0, >1, non-float).
    15. Invalid policy configuration rejection.
    16. Risk classification does not alter FRAUD/LEGIT decision.
    17. Risk category and fraud decision remain separate.
  Reason Codes:
    18. Correct fraud decision reason code.
    19. Correct risk-level reason code.
    20. Stable reason-code structure (code, description, category).
    21. No fabricated feature attribution.
    22. No raw transaction data in reason codes.
  Single Inference Integration:
    23. Standardized single result populated with Phase 6 fields.
    24. Existing Phase 4 behavior preserved.
    25. Frozen threshold unchanged.
    26. Existing error handling preserved.
  Batch Inference Integration:
    27. Standardized successful batch results.
    28. Partial-success behavior preserved.
    29. Failed transactions remain errors (NEVER classified as LOW risk).
    30. Input order preserved in batch.
    31. Single-versus-batch consistency (identical prob, decision, risk level, reason codes).
    32. Same risk policy used in both modes.
  Smoke & Regression:
    33. Real model smoke test with LightGBM v3 artifact.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
from typing import Any, Dict, List
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ensure workspace root is in sys.path
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase4_api.src.batch_inference_engine import (
    BatchRecordResult,
    BatchTransactionInferenceEngine,
    get_batch_inference_engine,
)
from phase4_api.src.inference_engine import (
    PredictionResponse,
    SingleTransactionInferenceEngine,
    get_inference_engine,
)
from phase4_api.src.model_loader import (
    APPROVED_DEFAULT_MODEL_VERSION,
    EXPECTED_FEATURE_COUNT,
    LoadedModelBundle,
    ModelInfo,
    get_model_bundle,
)
from phase4_api.src.prediction_schema import (
    InvalidSchemaDataError,
    PredictionResultModel,
    PredictionSchemaError,
    ReasonCodeModel,
    StandardizedPredictionResult,
)
from phase4_api.src.risk_classifier import (
    DEFAULT_LOW_THRESHOLD,
    DEFAULT_MEDIUM_THRESHOLD,
    DEFAULT_RISK_POLICY_VERSION,
    InvalidProbabilityError,
    InvalidRiskPolicyError,
    ReasonCode,
    ReasonCodeGenerationError,
    RiskClassificationError,
    RiskClassifier,
    RiskEvaluationResult,
    RiskLevel,
    RiskPolicy,
    get_risk_classifier,
)
from phase4_api.src.transaction_preprocessor import (
    ExtraFeatureError,
    MissingFeatureError,
    TransactionPreprocessor,
)
from phase4_api.tests.test_transaction_preprocessor import get_sample_valid_transaction

FROZEN_THRESHOLD: float = 0.8964321608040201


class MockBooster:
    """Mock LightGBM Booster returning controlled predictions."""

    def __init__(self, return_val: Any = 0.5) -> None:
        self.return_val = return_val

    def predict(self, df: pd.DataFrame) -> Any:
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
        feature_names=["f" + str(i) for i in range(EXPECTED_FEATURE_COUNT)],
        num_trees=500,
        frozen_threshold=threshold,
        model_loading_status="ACTIVE_LOADED",
        training_months=[0, 1, 2, 3, 4, 5, 6],
        model_hash="mock_hash_12345",
    )
    return LoadedModelBundle(
        booster=booster,
        info=info,
        metadata={"model_version": version},
        frozen_threshold=threshold,
    )


class TestPhase6PredictionResultAndRiskClassification(unittest.TestCase):
    """Test suite covering Phase 6 specifications."""

    def setUp(self) -> None:
        self.default_policy = RiskPolicy()
        self.default_classifier = RiskClassifier(self.default_policy)
        self.sample_tx = get_sample_valid_transaction()

    # -----------------------------------------------------------------------
    # Result Schema Tests (1 - 8)
    # -----------------------------------------------------------------------
    def test_01_valid_result_creation(self) -> None:
        """1. Valid StandardizedPredictionResult creation."""
        res = StandardizedPredictionResult(
            status="SUCCESS",
            transaction_id="TX-1001",
            prediction="LEGIT",
            fraud_probability=0.15,
            threshold=FROZEN_THRESHOLD,
            risk_level="LOW",
            model_version="v3",
            model_status="ACTIVE_LOADED",
            feature_count=29,
            execution_time_ms=2.5,
        )
        self.assertEqual(res.status, "SUCCESS")
        self.assertEqual(res.transaction_id, "TX-1001")
        self.assertEqual(res.prediction, "LEGIT")
        self.assertEqual(res.fraud_probability, 0.15)
        self.assertEqual(res.risk_level, "LOW")

    def test_02_required_field_validation(self) -> None:
        """2. Required field validation: status is required."""
        with self.assertRaises(InvalidSchemaDataError):
            StandardizedPredictionResult(status="INVALID_STATUS")

    def test_03_probability_range_validation(self) -> None:
        """3. Probability range validation: must be within [0.0, 1.0]."""
        with self.assertRaises(InvalidSchemaDataError):
            StandardizedPredictionResult(
                status="SUCCESS",
                prediction="FRAUD",
                fraud_probability=1.5,  # Invalid: > 1.0
                threshold=FROZEN_THRESHOLD,
                risk_level="CRITICAL",
            )
        with self.assertRaises(InvalidSchemaDataError):
            StandardizedPredictionResult(
                status="SUCCESS",
                prediction="LEGIT",
                fraud_probability=-0.1,  # Invalid: < 0.0
                threshold=FROZEN_THRESHOLD,
                risk_level="LOW",
            )

    def test_04_threshold_validation(self) -> None:
        """4. Threshold validation: must be within [0.0, 1.0]."""
        with self.assertRaises(InvalidSchemaDataError):
            StandardizedPredictionResult(
                status="SUCCESS",
                prediction="FRAUD",
                fraud_probability=0.9,
                threshold=2.0,  # Invalid
                risk_level="CRITICAL",
            )

    def test_05_model_version_validation(self) -> None:
        """5. Model version validation in Pydantic schema."""
        model = PredictionResultModel(
            status="SUCCESS",
            prediction="FRAUD",
            fraud_probability=0.95,
            threshold=FROZEN_THRESHOLD,
            risk_level="CRITICAL",
            model_version="v3",
            feature_count=29,
        )
        self.assertEqual(model.model_version, "v3")

    def test_06_missing_optional_fields_handled(self) -> None:
        """6. Missing optional fields (e.g. transaction_id, execution_time_ms) allowed."""
        res = StandardizedPredictionResult(
            status="SUCCESS",
            prediction="LEGIT",
            fraud_probability=0.05,
            threshold=FROZEN_THRESHOLD,
            risk_level="LOW",
        )
        self.assertIsNone(res.transaction_id)
        self.assertIsNone(res.execution_time_ms)
        self.assertEqual(res.reason_codes, [])

    def test_07_json_serialization_round_trip(self) -> None:
        """7. JSON serialization round-trip."""
        rc = ReasonCode(code="MODEL_DECISION_LEGIT", description="Legit tx", category="MODEL_DECISION")
        res = StandardizedPredictionResult(
            status="SUCCESS",
            transaction_id="TX-55",
            prediction="LEGIT",
            fraud_probability=0.08,
            threshold=FROZEN_THRESHOLD,
            risk_level="LOW",
            model_version="v3",
            model_status="ACTIVE_LOADED",
            feature_count=29,
            reason_codes=[rc],
            explanation_metadata={"test": True},
            execution_time_ms=1.23,
        )
        json_str = res.to_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["transaction_id"], "TX-55")
        self.assertEqual(parsed["reason_codes"][0]["code"], "MODEL_DECISION_LEGIT")

        restored = StandardizedPredictionResult.from_dict(parsed)
        self.assertEqual(restored.transaction_id, "TX-55")
        self.assertEqual(restored.risk_level, "LOW")
        self.assertEqual(restored.reason_codes[0].code, "MODEL_DECISION_LEGIT")

    def test_08_invalid_data_rejection(self) -> None:
        """8. Invalid data rejection (e.g. invalid prediction string)."""
        with self.assertRaises(InvalidSchemaDataError):
            StandardizedPredictionResult(
                status="SUCCESS",
                prediction="MAYBE_FRAUD",  # Invalid
                fraud_probability=0.5,
                threshold=0.5,
                risk_level="MEDIUM",
            )
        with self.assertRaises(InvalidSchemaDataError):
            StandardizedPredictionResult(
                status="SUCCESS",
                prediction="FRAUD",
                fraud_probability=0.9,
                threshold=0.5,
                risk_level="EXTREME",  # Invalid risk tier
            )

    # -----------------------------------------------------------------------
    # Risk Classification Tests (9 - 17)
    # -----------------------------------------------------------------------
    def test_09_low_classification(self) -> None:
        """9. LOW classification: p in [0.0, 0.20)."""
        self.assertEqual(self.default_classifier.classify(0.00, FROZEN_THRESHOLD), RiskLevel.LOW)
        self.assertEqual(self.default_classifier.classify(0.10, FROZEN_THRESHOLD), RiskLevel.LOW)
        self.assertEqual(self.default_classifier.classify(0.199999, FROZEN_THRESHOLD), RiskLevel.LOW)

    def test_10_medium_classification(self) -> None:
        """10. MEDIUM classification: p in [0.20, 0.50)."""
        self.assertEqual(self.default_classifier.classify(0.20, FROZEN_THRESHOLD), RiskLevel.MEDIUM)
        self.assertEqual(self.default_classifier.classify(0.35, FROZEN_THRESHOLD), RiskLevel.MEDIUM)
        self.assertEqual(self.default_classifier.classify(0.499999, FROZEN_THRESHOLD), RiskLevel.MEDIUM)

    def test_11_high_classification(self) -> None:
        """11. HIGH classification: p in [0.50, frozen_threshold)."""
        self.assertEqual(self.default_classifier.classify(0.50, FROZEN_THRESHOLD), RiskLevel.HIGH)
        self.assertEqual(self.default_classifier.classify(0.70, FROZEN_THRESHOLD), RiskLevel.HIGH)
        self.assertEqual(
            self.default_classifier.classify(FROZEN_THRESHOLD - 1e-6, FROZEN_THRESHOLD),
            RiskLevel.HIGH,
        )

    def test_12_critical_classification(self) -> None:
        """12. CRITICAL classification: p in [frozen_threshold, 1.0]."""
        self.assertEqual(self.default_classifier.classify(FROZEN_THRESHOLD, FROZEN_THRESHOLD), RiskLevel.CRITICAL)
        self.assertEqual(self.default_classifier.classify(0.95, FROZEN_THRESHOLD), RiskLevel.CRITICAL)
        self.assertEqual(self.default_classifier.classify(1.0, FROZEN_THRESHOLD), RiskLevel.CRITICAL)

    def test_13_exact_boundary_behavior(self) -> None:
        """13. Exact boundary behavior: cutoffs belong to higher category."""
        # 0.20 -> MEDIUM
        self.assertEqual(self.default_classifier.classify(0.20, FROZEN_THRESHOLD), RiskLevel.MEDIUM)
        # 0.50 -> HIGH
        self.assertEqual(self.default_classifier.classify(0.50, FROZEN_THRESHOLD), RiskLevel.HIGH)
        # FROZEN_THRESHOLD -> CRITICAL
        self.assertEqual(self.default_classifier.classify(FROZEN_THRESHOLD, FROZEN_THRESHOLD), RiskLevel.CRITICAL)

    def test_14_invalid_probability_rejection(self) -> None:
        """14. Invalid probability rejection (NaN, Inf, <0, >1, non-numeric)."""
        with self.assertRaises(InvalidProbabilityError):
            self.default_classifier.classify(float("nan"), FROZEN_THRESHOLD)
        with self.assertRaises(InvalidProbabilityError):
            self.default_classifier.classify(float("inf"), FROZEN_THRESHOLD)
        with self.assertRaises(InvalidProbabilityError):
            self.default_classifier.classify(-0.01, FROZEN_THRESHOLD)
        with self.assertRaises(InvalidProbabilityError):
            self.default_classifier.classify(1.01, FROZEN_THRESHOLD)
        with self.assertRaises(InvalidProbabilityError):
            self.default_classifier.classify("high", FROZEN_THRESHOLD)  # type: ignore

    def test_15_invalid_policy_configuration(self) -> None:
        """15. Invalid policy configuration rejection."""
        # Non-monotonic
        with self.assertRaises(InvalidRiskPolicyError):
            RiskPolicy(low_threshold=0.6, medium_threshold=0.4)
        # Out of bounds
        with self.assertRaises(InvalidRiskPolicyError):
            RiskPolicy(low_threshold=-0.1)
        with self.assertRaises(InvalidRiskPolicyError):
            RiskPolicy(medium_threshold=1.5)
        # Medium exceeds high
        with self.assertRaises(InvalidRiskPolicyError):
            RiskPolicy(low_threshold=0.2, medium_threshold=0.8, high_threshold=0.7)

    def test_16_risk_classification_does_not_change_fraud_decision(self) -> None:
        """16. Risk classification must not change the frozen threshold FRAUD/LEGIT decision."""
        # Below threshold: LEGIT regardless of whether LOW, MEDIUM, or HIGH risk
        prob_high = 0.75  # HIGH risk, but below ~0.8964
        decision = "FRAUD" if prob_high >= FROZEN_THRESHOLD else "LEGIT"
        risk_lvl = self.default_classifier.classify(prob_high, FROZEN_THRESHOLD)
        self.assertEqual(decision, "LEGIT")
        self.assertEqual(risk_lvl, RiskLevel.HIGH)

        # Above threshold: FRAUD
        prob_crit = 0.92
        decision_crit = "FRAUD" if prob_crit >= FROZEN_THRESHOLD else "LEGIT"
        risk_crit = self.default_classifier.classify(prob_crit, FROZEN_THRESHOLD)
        self.assertEqual(decision_crit, "FRAUD")
        self.assertEqual(risk_crit, RiskLevel.CRITICAL)

    def test_17_risk_category_and_fraud_decision_remain_separate(self) -> None:
        """17. Risk category and fraud decision separation: High risk with LEGIT decision."""
        eval_result = self.default_classifier.evaluate(
            probability=0.75,
            frozen_threshold=FROZEN_THRESHOLD,
            decision="LEGIT",
            model_version="v3",
            model_status="ACTIVE_LOADED",
            feature_count=29,
        )
        self.assertEqual(eval_result.risk_level, RiskLevel.HIGH)
        # Decision reason must be LEGIT
        codes = [rc.code for rc in eval_result.reason_codes]
        self.assertIn("MODEL_DECISION_LEGIT", codes)
        self.assertIn("RISK_LEVEL_HIGH", codes)
        self.assertIn("THRESHOLD_NOT_MET", codes)

    # -----------------------------------------------------------------------
    # Reason Codes Tests (18 - 22)
    # -----------------------------------------------------------------------
    def test_18_correct_fraud_decision_reason_code(self) -> None:
        """18. Correct fraud decision reason codes emitted."""
        eval_fraud = self.default_classifier.evaluate(
            probability=0.92,
            frozen_threshold=FROZEN_THRESHOLD,
            decision="FRAUD",
            model_version="v3",
            model_status="ACTIVE_LOADED",
            feature_count=29,
        )
        codes = [rc.code for rc in eval_fraud.reason_codes]
        self.assertIn("MODEL_DECISION_FRAUD", codes)
        self.assertIn("THRESHOLD_EXCEEDED", codes)

    def test_19_correct_risk_level_reason_code(self) -> None:
        """19. Correct risk-level reason codes emitted."""
        eval_low = self.default_classifier.evaluate(
            probability=0.05,
            frozen_threshold=FROZEN_THRESHOLD,
            decision="LEGIT",
            model_version="v3",
            model_status="ACTIVE_LOADED",
            feature_count=29,
        )
        codes = [rc.code for rc in eval_low.reason_codes]
        self.assertIn("RISK_LEVEL_LOW", codes)

        eval_med = self.default_classifier.evaluate(
            probability=0.30,
            frozen_threshold=FROZEN_THRESHOLD,
            decision="LEGIT",
            model_version="v3",
            model_status="ACTIVE_LOADED",
            feature_count=29,
        )
        codes_med = [rc.code for rc in eval_med.reason_codes]
        self.assertIn("RISK_LEVEL_MEDIUM", codes_med)

    def test_20_stable_reason_code_structure(self) -> None:
        """20. Stable reason-code structure (code, description, category)."""
        eval_res = self.default_classifier.evaluate(
            probability=0.10,
            frozen_threshold=FROZEN_THRESHOLD,
            decision="LEGIT",
            model_version="v3",
            model_status="ACTIVE_LOADED",
            feature_count=29,
        )
        for rc in eval_res.reason_codes:
            self.assertIsInstance(rc.code, str)
            self.assertIsInstance(rc.description, str)
            self.assertIsInstance(rc.category, str)
            d = rc.to_dict()
            self.assertIn("code", d)
            self.assertIn("description", d)
            self.assertIn("category", d)

    def test_21_no_fabricated_feature_attribution(self) -> None:
        """21. No fabricated feature attributions; neutral disclaimer included."""
        eval_res = self.default_classifier.evaluate(
            probability=0.95,
            frozen_threshold=FROZEN_THRESHOLD,
            decision="FRAUD",
            model_version="v3",
            model_status="ACTIVE_LOADED",
            feature_count=29,
        )
        codes = [rc.code for rc in eval_res.reason_codes]
        self.assertIn("FEATURE_EXPLANATION_UNAVAILABLE", codes)
        self.assertFalse(eval_res.explanation_metadata["feature_explanations_available"])

    def test_22_no_raw_transaction_data_in_reason_codes(self) -> None:
        """22. No raw transaction data in reason codes or metadata."""
        eval_res = self.default_classifier.evaluate(
            probability=0.95,
            frozen_threshold=FROZEN_THRESHOLD,
            decision="FRAUD",
            model_version="v3",
            model_status="ACTIVE_LOADED",
            feature_count=29,
        )
        raw_repr = str(eval_res.to_dict())
        # Confirm no feature names or customer attributes exist
        self.assertNotIn("customer_age", raw_repr)
        self.assertNotIn("income", raw_repr)
        self.assertNotIn("bank_months_count", raw_repr)

    # -----------------------------------------------------------------------
    # Single Inference Integration Tests (23 - 26)
    # -----------------------------------------------------------------------
    def test_23_standardized_single_result(self) -> None:
        """23. SingleTransactionInferenceEngine produces standardized Phase 6 result."""
        bundle = create_mock_bundle(MockBooster(0.12))
        engine = SingleTransactionInferenceEngine(model_bundle=bundle)
        resp = engine.predict(self.sample_tx, transaction_id="TX-SINGLE-1")

        self.assertEqual(resp.transaction_id, "TX-SINGLE-1")
        self.assertEqual(resp.prediction, "LEGIT")
        self.assertEqual(resp.risk_level, "LOW")
        self.assertEqual(resp.status, "SUCCESS")
        self.assertTrue(len(resp.reason_codes) >= 3)
        self.assertIsNotNone(resp.explanation_metadata)

        d = resp.to_dict()
        self.assertEqual(d["risk_level"], "LOW")
        self.assertEqual(d["status"], "SUCCESS")
        self.assertEqual(d["prediction"], "LEGIT")

    def test_24_existing_phase4_behavior_preserved(self) -> None:
        """24. Existing Phase 4 fields preserved identically."""
        bundle = create_mock_bundle(MockBooster(0.92))
        engine = SingleTransactionInferenceEngine(model_bundle=bundle)
        resp = engine.predict(self.sample_tx, transaction_id="TX-P4")

        self.assertEqual(resp.prediction, "FRAUD")
        self.assertAlmostEqual(resp.fraud_probability, 0.92, places=5)
        self.assertEqual(resp.threshold, FROZEN_THRESHOLD)
        self.assertEqual(resp.model_version, "v3")
        self.assertEqual(resp.model_status, "ACTIVE_LOADED")
        self.assertEqual(resp.feature_count, 29)
        self.assertGreater(resp.execution_time_ms, 0.0)

    def test_25_frozen_threshold_unchanged(self) -> None:
        """25. Frozen threshold is never modified by risk evaluation."""
        bundle = create_mock_bundle(MockBooster(0.8964321608040201))
        engine = SingleTransactionInferenceEngine(model_bundle=bundle)
        resp = engine.predict(self.sample_tx)
        self.assertEqual(resp.threshold, FROZEN_THRESHOLD)
        self.assertEqual(engine.threshold, FROZEN_THRESHOLD)

    def test_26_existing_error_handling_preserved(self) -> None:
        """26. Existing Phase 4 error handling (e.g. missing feature) preserved."""
        bundle = create_mock_bundle(MockBooster(0.5))
        engine = SingleTransactionInferenceEngine(model_bundle=bundle)
        bad_tx = dict(self.sample_tx)
        del bad_tx["income"]
        with self.assertRaises(MissingFeatureError):
            engine.predict(bad_tx)

    # -----------------------------------------------------------------------
    # Batch Inference Integration Tests (27 - 32)
    # -----------------------------------------------------------------------
    def test_27_standardized_successful_batch_results(self) -> None:
        """27. Batch inference enriches successful records with Phase 6 fields."""
        bundle = create_mock_bundle(MockBooster(0.35))
        engine = BatchTransactionInferenceEngine(model_bundle=bundle)
        records = [dict(self.sample_tx, transaction_id=f"B-{i}") for i in range(3)]
        resp = engine.predict_batch(records)

        self.assertEqual(resp.batch_status, "all_success")
        for rec in resp.results:
            self.assertEqual(rec.status, "success")
            self.assertEqual(rec.prediction, "LEGIT")
            self.assertEqual(rec.risk_level, "MEDIUM")
            self.assertIsNotNone(rec.reason_codes)
            self.assertIsNotNone(rec.explanation_metadata)
            d = rec.to_dict()
            self.assertEqual(d["risk_level"], "MEDIUM")

    def test_28_partial_success_behavior_preserved(self) -> None:
        """28. Partial-success preserved: valid records classified, errors untouched."""
        bundle = create_mock_bundle(MockBooster(0.95))
        engine = BatchTransactionInferenceEngine(model_bundle=bundle)

        good_rec = dict(self.sample_tx, transaction_id="GOOD")
        bad_rec = {"transaction_id": "BAD"}  # Missing 29 features

        resp = engine.predict_batch([good_rec, bad_rec])
        self.assertEqual(resp.batch_status, "partial_success")
        self.assertEqual(resp.successful_transactions, 1)
        self.assertEqual(resp.failed_transactions, 1)

        # Index 0: success -> CRITICAL risk
        self.assertEqual(resp.results[0].status, "success")
        self.assertEqual(resp.results[0].risk_level, "CRITICAL")
        self.assertEqual(resp.results[0].prediction, "FRAUD")

        # Index 1: error -> no risk level assigned
        self.assertEqual(resp.results[1].status, "error")
        self.assertIsNone(resp.results[1].risk_level)
        self.assertIsNone(resp.results[1].prediction)

    def test_29_failed_transactions_remain_errors_not_low_risk(self) -> None:
        """29. Failed transactions must NEVER be classified as LOW risk."""
        bundle = create_mock_bundle(MockBooster(0.5))
        engine = BatchTransactionInferenceEngine(model_bundle=bundle)
        bad_records = [
            {"invalid": 123},
            dict(self.sample_tx, income="not-a-number"),
        ]
        resp = engine.predict_batch(bad_records)
        self.assertEqual(resp.batch_status, "all_failed")
        for rec in resp.results:
            self.assertEqual(rec.status, "error")
            self.assertIsNone(rec.risk_level)
            self.assertNotEqual(rec.risk_level, "LOW")

    def test_30_input_order_preserved(self) -> None:
        """30. Original input order and indices preserved."""
        bundle = create_mock_bundle(MockBooster(0.25))
        engine = BatchTransactionInferenceEngine(model_bundle=bundle)
        records = [
            dict(self.sample_tx, transaction_id="T0"),
            {"transaction_id": "T1"},  # Bad
            dict(self.sample_tx, transaction_id="T2"),
        ]
        resp = engine.predict_batch(records)
        self.assertEqual(resp.results[0].input_index, 0)
        self.assertEqual(resp.results[0].transaction_id, "T0")
        self.assertEqual(resp.results[0].status, "success")

        self.assertEqual(resp.results[1].input_index, 1)
        self.assertEqual(resp.results[1].transaction_id, "T1")
        self.assertEqual(resp.results[1].status, "error")

        self.assertEqual(resp.results[2].input_index, 2)
        self.assertEqual(resp.results[2].transaction_id, "T2")
        self.assertEqual(resp.results[2].status, "success")

    def test_31_single_vs_batch_consistency(self) -> None:
        """31. Single and batch inference produce mathematically identical outputs."""
        bundle = create_mock_bundle(MockBooster(0.65))
        single_engine = SingleTransactionInferenceEngine(model_bundle=bundle)
        batch_engine = BatchTransactionInferenceEngine(model_bundle=bundle)

        tx = dict(self.sample_tx, transaction_id="TX-CONSISTENT")
        single_res = single_engine.predict(tx)
        batch_res = batch_engine.predict_batch([tx])
        batch_item = batch_res.results[0]

        self.assertEqual(single_res.prediction, batch_item.prediction)
        self.assertAlmostEqual(single_res.fraud_probability, batch_item.fraud_probability, places=6)
        self.assertEqual(single_res.risk_level, batch_item.risk_level)
        self.assertEqual(single_res.threshold, batch_item.threshold)
        self.assertEqual(single_res.model_version, batch_item.model_version)

        single_codes = [rc.code for rc in single_res.reason_codes]
        batch_codes = [rc.code for rc in batch_item.reason_codes]
        self.assertEqual(single_codes, batch_codes)

    def test_32_same_risk_policy_used_in_both_modes(self) -> None:
        """32. Custom risk policy applies identically to single and batch engines."""
        custom_policy = RiskPolicy(low_threshold=0.30, medium_threshold=0.60, high_threshold=0.85, version="v-custom")
        custom_classifier = RiskClassifier(custom_policy)

        bundle = create_mock_bundle(MockBooster(0.25))
        single_engine = SingleTransactionInferenceEngine(model_bundle=bundle, risk_classifier=custom_classifier)
        batch_engine = BatchTransactionInferenceEngine(model_bundle=bundle, risk_classifier=custom_classifier)

        tx = dict(self.sample_tx, transaction_id="TX-CUSTOM")
        # 0.25 is < 0.30 -> LOW in custom policy (whereas in default policy it would be MEDIUM)
        s_res = single_engine.predict(tx)
        b_res = batch_engine.predict_batch([tx]).results[0]

        self.assertEqual(s_res.risk_level, "LOW")
        self.assertEqual(b_res.risk_level, "LOW")
        self.assertEqual(s_res.explanation_metadata["risk_policy_version"], "v-custom")
        self.assertEqual(b_res.explanation_metadata["risk_policy_version"], "v-custom")

    # -----------------------------------------------------------------------
    # Smoke Test with Real LightGBM v3 Model (33)
    # -----------------------------------------------------------------------
    def test_33_real_model_smoke_test(self) -> None:
        """33. End-to-end smoke test using the authoritative LightGBM v3 model."""
        real_bundle = get_model_bundle(model_version="v3")
        self.assertEqual(real_bundle.info.model_version, "v3")
        self.assertAlmostEqual(real_bundle.frozen_threshold, FROZEN_THRESHOLD, places=5)

        single_engine = SingleTransactionInferenceEngine(model_version="v3")
        batch_engine = BatchTransactionInferenceEngine(model_version="v3")

        tx = dict(self.sample_tx, transaction_id="REAL-SMOKE-TX")

        single_res = single_engine.predict(tx)
        self.assertIn(single_res.prediction, ("FRAUD", "LEGIT"))
        self.assertIn(single_res.risk_level, ("LOW", "MEDIUM", "HIGH", "CRITICAL"))
        self.assertAlmostEqual(single_res.threshold, FROZEN_THRESHOLD, places=5)
        self.assertEqual(single_res.model_version, "v3")
        self.assertEqual(single_res.status, "SUCCESS")
        self.assertTrue(len(single_res.reason_codes) >= 3)

        batch_res = batch_engine.predict_batch([tx])
        self.assertEqual(batch_res.batch_status, "all_success")
        rec = batch_res.results[0]
        self.assertEqual(rec.prediction, single_res.prediction)
        self.assertEqual(rec.risk_level, single_res.risk_level)
        self.assertAlmostEqual(rec.fraud_probability, single_res.fraud_probability, places=5)


if __name__ == "__main__":
    unittest.main()
