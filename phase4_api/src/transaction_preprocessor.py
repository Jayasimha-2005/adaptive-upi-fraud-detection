"""
phase4_api/src/transaction_preprocessor.py
==========================================
Inference preprocessing engine for BAF transaction records.

Transforms raw transaction payloads into the exact single-row pandas DataFrame
format expected by the Phase 4 LightGBM v3 model:
1. Validates presence of all 29 canonical features.
2. Rejects unexpected extra features (including excluded cols fraud_bool, month, device_fraud_count).
3. Validates numerical types, casting strings to floats, preserving NaN for missing.
4. Validates and encodes categorical features using native pandas Categorical dtypes
   matching training-time categories from booster metadata.
5. Produces an invariant 1x29 pandas DataFrame with strict column ordering.
6. Guarantees non-mutation of input records.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from phase4_api.src.model_loader import (
    CANONICAL_CATEGORICAL_COLS,
    CANONICAL_FEATURE_COLS,
    CANONICAL_NUMERICAL_COLS,
    EXPECTED_FEATURE_COUNT,
)

# ---------------------------------------------------------------------------
# Training-Time Categorical Domain Levels
# Extracted directly from LightGBM booster dump ('pandas_categorical')
# and verified against psi_reference_months0_3.json
# ---------------------------------------------------------------------------
CANONICAL_CATEGORICAL_LEVELS: Dict[str, Tuple[str, ...]] = {
    "payment_type": ("AA", "AB", "AC", "AD", "AE"),
    "employment_status": ("CA", "CB", "CC", "CD", "CE", "CF", "CG"),
    "housing_status": ("BA", "BB", "BC", "BD", "BE", "BF", "BG"),
    "source": ("INTERNET", "TELEAPP"),
    "device_os": ("linux", "macintosh", "other", "windows", "x11"),
}

CANONICAL_CATEGORICAL_LEVELS_SET: Dict[str, set[str]] = {
    k: set(v) for k, v in CANONICAL_CATEGORICAL_LEVELS.items()
}

CANONICAL_FEATURE_SET: set[str] = set(CANONICAL_FEATURE_COLS)
CANONICAL_NUMERICAL_SET: set[str] = set(CANONICAL_NUMERICAL_COLS)
CANONICAL_CATEGORICAL_SET: set[str] = set(CANONICAL_CATEGORICAL_COLS)


# ---------------------------------------------------------------------------
# Preprocessor Exception Hierarchy
# ---------------------------------------------------------------------------
class PreprocessorError(Exception):
    """Base exception for all transaction preprocessing errors."""


class EmptyTransactionError(PreprocessorError):
    """Raised when the input transaction record is empty or None."""


class InvalidTransactionFormatError(PreprocessorError):
    """Raised when the input transaction record is not a dictionary/mapping."""


class MissingFeatureError(PreprocessorError):
    """Raised when one or more required canonical features are absent."""


class ExtraFeatureError(PreprocessorError):
    """Raised when unexpected features are present in the transaction record."""


class InvalidNumericalValueError(PreprocessorError):
    """Raised when a numerical feature cannot be converted to a valid float/number."""


class InvalidCategoricalValueError(PreprocessorError):
    """Raised when a categorical feature contains an unrecognized or invalid value."""


class PreprocessedOutputValidationError(PreprocessorError):
    """Raised when the output DataFrame fails final shape or schema validation."""


# ---------------------------------------------------------------------------
# Transaction Preprocessor Class
# ---------------------------------------------------------------------------
class TransactionPreprocessor:
    """
    Validates and transforms a single BAF transaction dictionary into
    a 1-row pandas DataFrame ready for LightGBM Booster prediction.
    """

    def __init__(
        self,
        allow_unseen_categories: bool = False,
        allow_missing_numerical: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        allow_unseen_categories : bool, default=False
            If False (strict, default), unseen categorical values raise
            InvalidCategoricalValueError. If True, converts unseen categories to NaN.
        allow_missing_numerical : bool, default=True
            If True, None/NaN in numerical columns is preserved as np.nan
            (which LightGBM natively routes to default split branches).
            If False, raises InvalidNumericalValueError.
        """
        self.allow_unseen_categories = allow_unseen_categories
        self.allow_missing_numerical = allow_missing_numerical

    def validate_keys(self, record: Mapping[str, Any]) -> None:
        """
        Validates that record contains exactly the 29 canonical features.
        """
        record_keys = set(record.keys())

        # Check for missing features
        missing = CANONICAL_FEATURE_SET - record_keys
        if missing:
            # Sort for deterministic error message
            sorted_missing = sorted(missing)
            raise MissingFeatureError(
                f"Transaction missing {len(missing)} required features: {sorted_missing}"
            )

        # Check for extra features (including target/temporal/constant excluded features)
        extra = record_keys - CANONICAL_FEATURE_SET
        if extra:
            sorted_extra = sorted(extra)
            raise ExtraFeatureError(
                f"Transaction contains {len(extra)} unexpected extra features: {sorted_extra}. "
                "Inputs must contain only the 29 canonical features."
            )

    def process_numerical(self, col: str, val: Any) -> float:
        """
        Validates and converts a numerical feature value to float.
        """
        # Handle None or NaN
        if val is None or (isinstance(val, float) and math.isnan(val)):
            if not self.allow_missing_numerical:
                raise InvalidNumericalValueError(
                    f"Numerical feature '{col}' is missing (None/NaN), but missing numericals are disallowed."
                )
            return float("nan")

        # Handle numeric types or castable strings
        try:
            num = float(val)
        except (ValueError, TypeError) as e:
            raise InvalidNumericalValueError(
                f"Numerical feature '{col}' cannot be converted to float: received {val!r} (type: {type(val).__name__})"
            ) from e

        # Check for infinite values
        if math.isinf(num):
            raise InvalidNumericalValueError(
                f"Numerical feature '{col}' has non-finite infinite value: {val!r}"
            )

        return num

    def process_categorical(self, col: str, val: Any) -> Optional[str]:
        """
        Validates and normalizes a categorical feature value.
        """
        allowed = CANONICAL_CATEGORICAL_LEVELS_SET[col]

        # Handle missing categorical values
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return None

        # Convert to string representation
        str_val = str(val).strip()

        if str_val not in allowed:
            if not self.allow_unseen_categories:
                raise InvalidCategoricalValueError(
                    f"Invalid categorical value '{val}' for feature '{col}'. "
                    f"Allowed categories: {list(CANONICAL_CATEGORICAL_LEVELS[col])}"
                )
            return None

        return str_val

    def transform(self, transaction: Mapping[str, Any]) -> pd.DataFrame:
        """
        Validates and transforms a single transaction record into a 1-row DataFrame.
        Guarantees non-mutation of the input dictionary.

        Parameters
        ----------
        transaction : Mapping[str, Any]
            Raw transaction record mapping feature names to values.

        Returns
        -------
        pd.DataFrame
            1-row DataFrame with exactly 29 columns matching CANONICAL_FEATURE_COLS.
        """
        if transaction is None:
            raise EmptyTransactionError("Transaction record cannot be None.")

        if not isinstance(transaction, Mapping):
            raise InvalidTransactionFormatError(
                f"Transaction record must be a dict/mapping, received {type(transaction).__name__}."
            )

        if len(transaction) == 0:
            raise EmptyTransactionError("Transaction record is empty (0 features).")

        # Validate schema keys
        self.validate_keys(transaction)

        # Build column data dictionary
        row_dict: Dict[str, Any] = {}

        # 1. Process Numerical Columns (0..23)
        for col in CANONICAL_NUMERICAL_COLS:
            val = transaction[col]
            row_dict[col] = [self.process_numerical(col, val)]

        # 2. Process Categorical Columns (24..28)
        for col in CANONICAL_CATEGORICAL_COLS:
            val = transaction[col]
            processed_cat = self.process_categorical(col, val)
            levels = list(CANONICAL_CATEGORICAL_LEVELS[col])
            # Construct pandas Categorical with exact training levels
            row_dict[col] = pd.Categorical([processed_cat], categories=levels)

        # Construct DataFrame with explicit column order
        df = pd.DataFrame(row_dict)[list(CANONICAL_FEATURE_COLS)]

        # Output shape and schema validation
        self.validate_output_dataframe(df)

        return df

    @staticmethod
    def validate_output_dataframe(df: pd.DataFrame) -> None:
        """
        Ensures the generated DataFrame strictly adheres to the model input contract.
        """
        if len(df) != 1:
            raise PreprocessedOutputValidationError(
                f"Output DataFrame must contain exactly 1 row, got {len(df)}."
            )

        if len(df.columns) != EXPECTED_FEATURE_COUNT:
            raise PreprocessedOutputValidationError(
                f"Output DataFrame must contain exactly {EXPECTED_FEATURE_COUNT} columns, got {len(df.columns)}."
            )

        actual_cols = list(df.columns)
        if actual_cols != list(CANONICAL_FEATURE_COLS):
            raise PreprocessedOutputValidationError(
                f"Output DataFrame column sequence does not match canonical schema: {actual_cols} != {list(CANONICAL_FEATURE_COLS)}"
            )

        # Verify categorical dtypes and category levels
        for col in CANONICAL_CATEGORICAL_COLS:
            series = df[col]
            if not isinstance(series.dtype, pd.CategoricalDtype):
                raise PreprocessedOutputValidationError(
                    f"Column '{col}' must have CategoricalDtype, got {series.dtype}."
                )
            expected_levels = list(CANONICAL_CATEGORICAL_LEVELS[col])
            actual_levels = list(series.cat.categories)
            if actual_levels != expected_levels:
                raise PreprocessedOutputValidationError(
                    f"Column '{col}' category levels mismatch: {actual_levels} != {expected_levels}."
                )


# ---------------------------------------------------------------------------
# Functional Helper API
# ---------------------------------------------------------------------------
_DEFAULT_PREPROCESSOR = TransactionPreprocessor(
    allow_unseen_categories=False,
    allow_missing_numerical=True,
)


def preprocess_transaction(
    transaction: Mapping[str, Any],
    allow_unseen_categories: bool = False,
    allow_missing_numerical: bool = True,
) -> pd.DataFrame:
    """
    Convenience function to preprocess a single transaction record.
    """
    if not allow_unseen_categories and allow_missing_numerical:
        return _DEFAULT_PREPROCESSOR.transform(transaction)
    
    preprocessor = TransactionPreprocessor(
        allow_unseen_categories=allow_unseen_categories,
        allow_missing_numerical=allow_missing_numerical,
    )
    return preprocessor.transform(transaction)
