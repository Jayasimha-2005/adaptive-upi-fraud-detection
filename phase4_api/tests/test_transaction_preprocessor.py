"""
phase4_api/tests/test_transaction_preprocessor.py
=================================================
Unit test suite for BAF transaction preprocessor.

Validates:
  1. Valid transaction preprocessing.
  2. Exactly 29 output columns.
  3. Correct numerical feature handling.
  4. Correct categorical feature handling.
  5. Exact feature ordering.
  6. Missing required feature rejection.
  7. Unexpected feature rejection.
  8. Invalid numerical value rejection.
  9. Invalid categorical value handling (strict rejection & permissive NaN mode).
  10. Missing numerical values (None/NaN preserved as np.nan).
  11. Missing categorical values (None preserved as NaN category).
  12. Empty input rejection.
  13. Correct pandas DataFrame output.
  14. Compatibility with the loaded LightGBM v3 model (prediction probability).
  15. Original input object remains unchanged.
"""
from __future__ import annotations

import copy
import math
import pathlib
import sys
import unittest

import numpy as np
import pandas as pd

# Setup path
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase4_api.src.model_loader import (
    CANONICAL_CATEGORICAL_COLS,
    CANONICAL_FEATURE_COLS,
    CANONICAL_NUMERICAL_COLS,
    EXPECTED_FEATURE_COUNT,
    get_model_bundle,
)
from phase4_api.src.transaction_preprocessor import (
    CANONICAL_CATEGORICAL_LEVELS,
    EmptyTransactionError,
    ExtraFeatureError,
    InvalidCategoricalValueError,
    InvalidNumericalValueError,
    InvalidTransactionFormatError,
    MissingFeatureError,
    PreprocessedOutputValidationError,
    TransactionPreprocessor,
    preprocess_transaction,
)


def get_sample_valid_transaction() -> dict:
    """Returns a valid 29-feature transaction dictionary based on BAF schema."""
    return {
        # 24 Numerical features
        "income": 0.5,
        "name_email_similarity": 0.825,
        "prev_address_months_count": -1,
        "current_address_months_count": 24,
        "customer_age": 35,
        "days_since_request": 0.015,
        "intended_balcon_amount": -1.0,
        "zip_count_4w": 1250,
        "velocity_6h": 2500.0,
        "velocity_24h": 4500.0,
        "velocity_4w": 5200.0,
        "bank_branch_count_8w": 2,
        "date_of_birth_distinct_emails_4w": 1,
        "credit_risk_score": 140,
        "email_is_free": 1,
        "phone_home_valid": 1,
        "phone_mobile_valid": 1,
        "bank_months_count": 15,
        "has_other_cards": 0,
        "proposed_credit_limit": 1000.0,
        "foreign_request": 0,
        "session_length_in_minutes": 7.5,
        "keep_alive_session": 1,
        "device_distinct_emails_8w": 1,
        # 5 Categorical features
        "payment_type": "AA",
        "employment_status": "CA",
        "housing_status": "BC",
        "source": "INTERNET",
        "device_os": "windows",
    }


class TestTransactionPreprocessor(unittest.TestCase):
    """Test suite for TransactionPreprocessor and convenience helpers."""

    def setUp(self) -> None:
        self.preprocessor = TransactionPreprocessor(
            allow_unseen_categories=False,
            allow_missing_numerical=True,
        )
        self.sample = get_sample_valid_transaction()

    # -------------------------------------------------------------------------
    # Test 1 & 13: Valid transaction produces correct pandas DataFrame
    # -------------------------------------------------------------------------
    def test_01_valid_transaction_preprocessing(self) -> None:
        df = self.preprocessor.transform(self.sample)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)

    # -------------------------------------------------------------------------
    # Test 2: Exactly 29 output columns
    # -------------------------------------------------------------------------
    def test_02_exactly_29_output_columns(self) -> None:
        df = self.preprocessor.transform(self.sample)
        self.assertEqual(len(df.columns), EXPECTED_FEATURE_COUNT)
        self.assertEqual(len(df.columns), 29)

    # -------------------------------------------------------------------------
    # Test 3: Correct numerical feature handling
    # -------------------------------------------------------------------------
    def test_03_correct_numerical_feature_handling(self) -> None:
        df = self.preprocessor.transform(self.sample)
        for num_col in CANONICAL_NUMERICAL_COLS:
            val = df[num_col].iloc[0]
            self.assertTrue(
                np.issubdtype(df[num_col].dtype, np.number),
                f"Column {num_col} dtype is not numeric: {df[num_col].dtype}",
            )
            self.assertEqual(val, float(self.sample[num_col]))

    # -------------------------------------------------------------------------
    # Test 4: Correct categorical feature handling
    # -------------------------------------------------------------------------
    def test_04_correct_categorical_feature_handling(self) -> None:
        df = self.preprocessor.transform(self.sample)
        for cat_col in CANONICAL_CATEGORICAL_COLS:
            series = df[cat_col]
            self.assertIsInstance(
                series.dtype,
                pd.CategoricalDtype,
                f"Column {cat_col} must have CategoricalDtype",
            )
            self.assertEqual(series.iloc[0], self.sample[cat_col])
            # Check training level domain
            expected_levels = list(CANONICAL_CATEGORICAL_LEVELS[cat_col])
            self.assertEqual(list(series.cat.categories), expected_levels)

    # -------------------------------------------------------------------------
    # Test 5: Exact feature ordering
    # -------------------------------------------------------------------------
    def test_05_exact_feature_ordering(self) -> None:
        df = self.preprocessor.transform(self.sample)
        actual_cols = list(df.columns)
        self.assertEqual(actual_cols, list(CANONICAL_FEATURE_COLS))

        # First 24 must be numerical
        self.assertEqual(actual_cols[:24], list(CANONICAL_NUMERICAL_COLS))
        # Last 5 must be categorical
        self.assertEqual(actual_cols[24:], list(CANONICAL_CATEGORICAL_COLS))

    # -------------------------------------------------------------------------
    # Test 6: Missing required feature rejection
    # -------------------------------------------------------------------------
    def test_06_missing_required_feature_rejection(self) -> None:
        bad_sample = dict(self.sample)
        del bad_sample["income"]
        del bad_sample["payment_type"]

        with self.assertRaises(MissingFeatureError) as ctx:
            self.preprocessor.transform(bad_sample)
        err = str(ctx.exception)
        self.assertIn("missing 2 required features", err.lower())
        self.assertIn("income", err)
        self.assertIn("payment_type", err)

    # -------------------------------------------------------------------------
    # Test 7: Unexpected feature rejection (including leakage prevention)
    # -------------------------------------------------------------------------
    def test_07_unexpected_feature_rejection(self) -> None:
        # A: Random extra field
        extra_sample = dict(self.sample)
        extra_sample["extra_metadata"] = "unexpected_value"
        with self.assertRaises(ExtraFeatureError) as ctx:
            self.preprocessor.transform(extra_sample)
        self.assertIn("unexpected extra features", str(ctx.exception).lower())
        self.assertIn("extra_metadata", str(ctx.exception))

        # B: Target label leakage check
        leak_target = dict(self.sample)
        leak_target["fraud_bool"] = 0
        with self.assertRaises(ExtraFeatureError) as ctx_target:
            self.preprocessor.transform(leak_target)
        self.assertIn("fraud_bool", str(ctx_target.exception))

        # C: Temporal column leakage check
        leak_month = dict(self.sample)
        leak_month["month"] = 5
        with self.assertRaises(ExtraFeatureError) as ctx_month:
            self.preprocessor.transform(leak_month)
        self.assertIn("month", str(ctx_month.exception))

        # D: Excluded constant column check
        leak_device = dict(self.sample)
        leak_device["device_fraud_count"] = 0
        with self.assertRaises(ExtraFeatureError) as ctx_dev:
            self.preprocessor.transform(leak_device)
        self.assertIn("device_fraud_count", str(ctx_dev.exception))

    # -------------------------------------------------------------------------
    # Test 8: Invalid numerical value rejection
    # -------------------------------------------------------------------------
    def test_08_invalid_numerical_value_rejection(self) -> None:
        bad_numeric = dict(self.sample)
        bad_numeric["customer_age"] = "not_a_valid_age"
        with self.assertRaises(InvalidNumericalValueError) as ctx:
            self.preprocessor.transform(bad_numeric)
        self.assertIn("cannot be converted to float", str(ctx.exception))
        self.assertIn("customer_age", str(ctx.exception))

        # Infinite value rejection
        bad_inf = dict(self.sample)
        bad_inf["velocity_6h"] = float("inf")
        with self.assertRaises(InvalidNumericalValueError) as ctx_inf:
            self.preprocessor.transform(bad_inf)
        self.assertIn("non-finite infinite value", str(ctx_inf.exception))

    # -------------------------------------------------------------------------
    # Test 9: Invalid categorical value handling
    # -------------------------------------------------------------------------
    def test_09_invalid_categorical_value_handling(self) -> None:
        bad_cat = dict(self.sample)
        bad_cat["payment_type"] = "INVALID_PAYMENT_XYZ"

        # Strict mode (default): raises InvalidCategoricalValueError
        with self.assertRaises(InvalidCategoricalValueError) as ctx:
            self.preprocessor.transform(bad_cat)
        self.assertIn("Invalid categorical value 'INVALID_PAYMENT_XYZ'", str(ctx.exception))
        self.assertIn("payment_type", str(ctx.exception))

        # Permissive mode: converts to NaN category
        permissive = TransactionPreprocessor(allow_unseen_categories=True)
        df = permissive.transform(bad_cat)
        self.assertTrue(pd.isna(df["payment_type"].iloc[0]))
        self.assertEqual(len(df.columns), 29)

    # -------------------------------------------------------------------------
    # Test 10: Missing numerical values (preserved as NaN)
    # -------------------------------------------------------------------------
    def test_10_missing_numerical_values_handling(self) -> None:
        sample_missing = dict(self.sample)
        sample_missing["intended_balcon_amount"] = None
        sample_missing["prev_address_months_count"] = float("nan")

        df = self.preprocessor.transform(sample_missing)
        self.assertTrue(np.isnan(df["intended_balcon_amount"].iloc[0]))
        self.assertTrue(np.isnan(df["prev_address_months_count"].iloc[0]))

        # When missing numericals are disallowed:
        strict_num = TransactionPreprocessor(allow_missing_numerical=False)
        with self.assertRaises(InvalidNumericalValueError) as ctx:
            strict_num.transform(sample_missing)
        self.assertIn("is missing (None/NaN)", str(ctx.exception))

    # -------------------------------------------------------------------------
    # Test 11: Missing categorical values (None preserved as NaN category)
    # -------------------------------------------------------------------------
    def test_11_missing_categorical_values_handling(self) -> None:
        sample_missing_cat = dict(self.sample)
        sample_missing_cat["employment_status"] = None

        df = self.preprocessor.transform(sample_missing_cat)
        self.assertTrue(pd.isna(df["employment_status"].iloc[0]))
        self.assertIsInstance(df["employment_status"].dtype, pd.CategoricalDtype)

    # -------------------------------------------------------------------------
    # Test 12: Empty input rejection
    # -------------------------------------------------------------------------
    def test_12_empty_input_rejection(self) -> None:
        with self.assertRaises(EmptyTransactionError):
            self.preprocessor.transform({})

        with self.assertRaises(EmptyTransactionError):
            self.preprocessor.transform(None)  # type: ignore[arg-type]

        with self.assertRaises(InvalidTransactionFormatError):
            self.preprocessor.transform(["not", "a", "dict"])  # type: ignore[arg-type]

    # -------------------------------------------------------------------------
    # Test 14: Model compatibility test with loaded LightGBM v3 model
    # -------------------------------------------------------------------------
    def test_14_model_compatibility_with_lightgbm_v3(self) -> None:
        # Load approved v3 model
        bundle = get_model_bundle("v3")
        booster = bundle.booster

        # Preprocess valid sample
        df = self.preprocessor.transform(self.sample)

        # Execute prediction
        pred_proba = booster.predict(df)
        self.assertEqual(len(pred_proba), 1)

        prob = float(pred_proba[0])
        self.assertTrue(math.isfinite(prob))
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

        # Test threshold decisioning
        threshold = bundle.frozen_threshold
        is_fraud = bool(prob >= threshold)
        self.assertIn(is_fraud, (True, False))

    # -------------------------------------------------------------------------
    # Test 15: Original input object remains unchanged (non-mutation guarantee)
    # -------------------------------------------------------------------------
    def test_15_original_input_object_unchanged(self) -> None:
        original = get_sample_valid_transaction()
        snapshot = copy.deepcopy(original)

        df = self.preprocessor.transform(original)

        self.assertEqual(original, snapshot)
        self.assertEqual(len(original), 29)


if __name__ == "__main__":
    unittest.main(verbosity=2)
