"""
phase4_api/benchmarks/dataset_loader.py
=======================================
Read-only Dataset Loader & Validator for BAF Transactions.

Part of Member 3 — API Serving Layer for Adaptive UPI / BAF Fraud Detection.
Phase 8: Performance Testing, Benchmarking & Parity Validation.

Strictly preserves:
- Read-only file access (no modifications or moves)
- Transaction sequence and ordering
- Separation of transaction IDs, ground truth labels (fraud_bool), and model features
- Accurate reporting of valid vs invalid transactions
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from phase4_api.src.model_loader import (
    CANONICAL_CATEGORICAL_COLS,
    CANONICAL_FEATURE_COLS,
    CANONICAL_NUMERICAL_COLS,
    EXPECTED_FEATURE_COUNT,
)
from phase4_api.src.transaction_preprocessor import (
    CANONICAL_CATEGORICAL_LEVELS,
)

logger = logging.getLogger("phase4_api.benchmarks.dataset_loader")


@dataclass(frozen=True)
class ValidatedTransactionRecord:
    """
    Individual validated BAF transaction record.
    Transaction ID and ground truth label are stored separately from model features.
    """
    transaction_id: str
    features: Dict[str, Any]
    ground_truth_label: Optional[int] = None
    is_synthetic: bool = False
    row_index: int = 0


@dataclass
class DatasetValidationResult:
    """Detailed summary of the dataset loading and validation process."""
    dataset_path: str
    total_rows_read: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    records: List[ValidatedTransactionRecord] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    missing_columns: List[str] = field(default_factory=list)
    is_synthetic: bool = False


class BAFDatasetLoader:
    """
    Safe, read-only loader for Bank Account Fraud (BAF) CSV datasets.
    """

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed

    def load_dataset(
        self,
        dataset_path: Optional[str],
        limit: int = 100,
        allow_synthetic_fallback: bool = False,
    ) -> DatasetValidationResult:
        """
        Loads and validates up to `limit` transactions from the given BAF CSV path.
        If path is None or empty and allow_synthetic_fallback is True, generates synthetic records
        explicitly marked as synthetic.
        """
        if not dataset_path:
            if allow_synthetic_fallback:
                logger.info("No dataset path provided; generating %d SYNTHETIC benchmark records.", limit)
                return self.generate_synthetic_dataset(limit=limit)
            raise ValueError(
                "Dataset path was not provided. Please provide a valid BAF CSV path via --dataset "
                "or enable --synthetic for mock testing."
            )

        resolved_path = Path(dataset_path).resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"BAF dataset file not found at: {resolved_path}")
        if not resolved_path.is_file():
            raise ValueError(f"Specified dataset path is not a file: {resolved_path}")

        logger.info("Opening BAF dataset in read-only mode: %s (limit=%d)", resolved_path, limit)

        # Enforce read-only inspection
        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                header_line = f.readline().strip()
        except Exception as e:
            raise IOError(f"Failed to read dataset header in read-only mode: {e}") from e

        csv_columns = [col.strip() for col in header_line.split(",") if col.strip()]

        # Validate that all 29 canonical features are present
        missing_features = [col for col in CANONICAL_FEATURE_COLS if col not in csv_columns]
        if missing_features:
            return DatasetValidationResult(
                dataset_path=str(resolved_path),
                total_rows_read=0,
                valid_count=0,
                invalid_count=0,
                missing_columns=missing_features,
                validation_errors=[f"Dataset is missing required canonical features: {missing_features}"],
            )

        # Read up to limit rows using pandas with read-only guarantee
        try:
            df = pd.read_csv(resolved_path, nrows=limit)
        except Exception as e:
            raise RuntimeError(f"Error reading CSV rows from {resolved_path}: {e}") from e

        result = DatasetValidationResult(
            dataset_path=str(resolved_path),
            total_rows_read=len(df),
            is_synthetic=False,
        )

        for idx, row in df.iterrows():
            tx_id = f"baf_tx_{idx:06d}"
            label: Optional[int] = None
            if "fraud_bool" in row and pd.notna(row["fraud_bool"]):
                try:
                    label = int(row["fraud_bool"])
                except Exception:
                    label = None

            # Validate row features
            features: Dict[str, Any] = {}
            row_errors: List[str] = []

            # 24 Numerical features
            for num_col in CANONICAL_NUMERICAL_COLS:
                val = row[num_col]
                if pd.isna(val) or val is None:
                    row_errors.append(f"Row {idx}: missing numerical value for '{num_col}'")
                    continue
                try:
                    float_val = float(val)
                    if math.isnan(float_val) or math.isinf(float_val):
                        row_errors.append(f"Row {idx}: numerical '{num_col}' is NaN/Inf")
                    else:
                        features[num_col] = float_val
                except (ValueError, TypeError):
                    row_errors.append(f"Row {idx}: could not cast '{num_col}' to float: {val}")

            # 5 Categorical features
            for cat_col in CANONICAL_CATEGORICAL_COLS:
                val = row[cat_col]
                if pd.isna(val) or val is None:
                    row_errors.append(f"Row {idx}: missing categorical value for '{cat_col}'")
                    continue
                cat_str = str(val).strip()
                allowed_levels = CANONICAL_CATEGORICAL_LEVELS.get(cat_col, set())
                if cat_str not in allowed_levels:
                    row_errors.append(
                        f"Row {idx}: invalid category '{cat_str}' for '{cat_col}' (allowed: {allowed_levels})"
                    )
                else:
                    features[cat_col] = cat_str

            if row_errors:
                result.invalid_count += 1
                result.validation_errors.extend(row_errors)
            else:
                result.valid_count += 1
                result.records.append(
                    ValidatedTransactionRecord(
                        transaction_id=tx_id,
                        features=features,
                        ground_truth_label=label,
                        is_synthetic=False,
                        row_index=int(idx),
                    )
                )

        logger.info(
            "Dataset loading complete: %d read, %d valid, %d invalid",
            result.total_rows_read,
            result.valid_count,
            result.invalid_count,
        )
        return result

    def generate_synthetic_dataset(self, limit: int = 10) -> DatasetValidationResult:
        """
        Generates synthetic records explicitly flagged as synthetic for offline testing.
        Never to be presented as real dataset performance.
        """
        result = DatasetValidationResult(
            dataset_path="SYNTHETIC_GENERATED",
            total_rows_read=limit,
            valid_count=limit,
            invalid_count=0,
            is_synthetic=True,
        )

        base_numerical = {
            "income": 0.5,
            "name_email_similarity": 0.85,
            "prev_address_months_count": 12.0,
            "current_address_months_count": 24.0,
            "customer_age": 35.0,
            "days_since_request": 0.05,
            "intended_balcon_amount": 50.0,
            "zip_count_4w": 500.0,
            "velocity_6h": 1200.0,
            "velocity_24h": 3400.0,
            "velocity_4w": 4500.0,
            "bank_branch_count_8w": 2.0,
            "date_of_birth_distinct_emails_4w": 1.0,
            "credit_risk_score": 140.0,
            "email_is_free": 1.0,
            "phone_home_valid": 1.0,
            "phone_mobile_valid": 1.0,
            "bank_months_count": 10.0,
            "has_other_cards": 0.0,
            "proposed_credit_limit": 1000.0,
            "foreign_request": 0.0,
            "session_length_in_minutes": 5.5,
            "keep_alive_session": 1.0,
            "device_distinct_emails_8w": 1.0,
        }
        base_categorical = {
            "payment_type": "AA",
            "employment_status": "CA",
            "housing_status": "BA",
            "source": "INTERNET",
            "device_os": "windows",
        }

        for i in range(limit):
            features = dict(base_numerical)
            features.update(base_categorical)
            # slight perturbation for uniqueness
            features["income"] = round(0.1 + (i % 9) * 0.1, 2)
            features["customer_age"] = float(20 + (i % 50))
            features["session_length_in_minutes"] = round(1.0 + (i * 0.5), 2)
            
            rec = ValidatedTransactionRecord(
                transaction_id=f"synth_tx_{i:06d}",
                features=features,
                ground_truth_label=0 if i % 10 != 0 else 1,
                is_synthetic=True,
                row_index=i,
            )
            result.records.append(rec)

        return result
