"""
preprocessing/serving_wrapper.py
Serving wrapper around the approved E1 LightGBM preprocessor artifact.

Handles the inference edge case where online/serving raw transactions do not
have an 'isFraud' target column, while ensuring zero target leakage and strict
validation of feature count (406), feature names, and feature ordering.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

# Add project root and serving directory to sys.path
SERVING_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SERVING_DIR.parent
for p in [str(REPO_ROOT), str(SERVING_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.features.ieee_cis_features import IEEECISPreprocessor

logger = logging.getLogger(__name__)

CANONICAL_E1_DIR = REPO_ROOT / "experiments" / "E1_lightgbm"


def resolve_e1_path(path: str | Path | None, filename: str) -> Path:
    """
    Resolve E1 artifact path with preference for canonical experiments/E1_lightgbm/.
    Falls back gracefully if explicit path or local serving path is provided.
    """
    if path:
        p = Path(path)
        if p.exists():
            return p
    canonical = CANONICAL_E1_DIR / filename
    if canonical.exists():
        return canonical
    local = SERVING_DIR / "models" / "E1" / filename
    if local.exists():
        return local
    return Path(path) if path else canonical


class ServingPreprocessor:
    """
    Serving wrapper for E1 preprocessor artifact.
    """

    EXPECTED_FEATURE_COUNT = 406

    def __init__(
        self,
        preprocessor_path: str | Path = "models/E1/preprocessing.joblib",
        feature_names_path: str | Path = "models/E1/feature_names.json",
    ):
        self.preprocessor_path = resolve_e1_path(preprocessor_path, "preprocessing.joblib")
        self.feature_names_path = resolve_e1_path(feature_names_path, "feature_names.json")

        # 1. Load feature names
        if not self.feature_names_path.exists():
            raise FileNotFoundError(f"Feature names file not found: {self.feature_names_path}")
        with open(self.feature_names_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.expected_feature_names: list[str] = data["feature_names"]

        if len(self.expected_feature_names) != self.EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"Expected {self.EXPECTED_FEATURE_COUNT} features in {self.feature_names_path}, "
                f"got {len(self.expected_feature_names)}"
            )

        # 2. Load preprocessor
        if not self.preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found: {self.preprocessor_path}")
        self.preprocessor: IEEECISPreprocessor = joblib.load(self.preprocessor_path)

        # 3. Validate preprocessor feature compatibility
        pp_features = self.preprocessor.feature_names
        if len(pp_features) != self.EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"Preprocessor artifact has {len(pp_features)} features, "
                f"expected {self.EXPECTED_FEATURE_COUNT}"
            )

        if pp_features != self.expected_feature_names:
            raise ValueError(
                "Preprocessor feature names or order do not match feature_names.json!"
            )

        logger.info(
            "ServingPreprocessor initialized successfully. Loaded 406 features matching schema."
        )

    def transform(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, dict[str, Any]]:
        """
        Safely preprocess raw transactions for inference.

        Parameters
        ----------
        df_raw : pd.DataFrame
            Raw transaction dataframe (with or without 'isFraud' / 'TransactionID').

        Returns
        -------
        X : pd.DataFrame
            Processed feature matrix of shape (N, 406), ordered strictly per feature_names.json.
        meta : dict
            Metadata dictionary containing:
            - 'transaction_ids': list of TransactionIDs (or RangeIndex if missing)
            - 'actual_labels': list of actual isFraud labels (or None if missing)
            - 'row_count': N
            - 'feature_count': 406
        """
        df = df_raw.copy()
        n_rows = len(df)

        # Extract TransactionID for metadata tracking if present
        if "TransactionID" in df.columns:
            tx_ids = df["TransactionID"].tolist()
        else:
            tx_ids = df.index.tolist()

        # Extract actual label for offline validation metadata tracking if present
        actual_labels = None
        if "isFraud" in df.columns:
            actual_labels = df["isFraud"].astype(int).tolist()
        else:
            # Handle serving edge case: inject dummy column for preprocessor.transform()
            df["isFraud"] = 0

        # Ensure all expected raw columns exist in df (fill missing raw fields with np.nan)
        # This guarantees full consistency when online serving JSON payloads omit missing fields.
        all_expected_raw = set(self.preprocessor._num_medians.keys()) | set(self.preprocessor._cat_encoders.keys()) | {"TransactionDT"}
        missing_raw = [c for c in all_expected_raw if c not in df.columns]
        if missing_raw:
            df = df.reindex(columns=list(df.columns) + missing_raw, fill_value=np.nan)

        # Transform raw features using approved preprocessor
        X_raw, _ = self.preprocessor.transform(df)

        # Ensure all expected feature columns exist in X_raw (fill missing with NaN)
        missing_feats = [c for c in self.expected_feature_names if c not in X_raw.columns]
        if missing_feats:
            X_raw = X_raw.reindex(columns=list(X_raw.columns) + missing_feats, fill_value=np.nan)

        # Re-order and align columns strictly with approved feature_names.json
        X = X_raw[self.expected_feature_names].copy()

        # Ensure all columns in X have numeric dtypes for LightGBM inference
        for col in self.expected_feature_names:
            if X[col].dtype == object:
                X[col] = pd.to_numeric(X[col], errors="coerce")

        # ── Feature Validation Checks ──────────────────────────────────────────
        # Check 1: Feature count
        if X.shape[1] != self.EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"Feature validation failed: Processed shape is {X.shape}, expected {self.EXPECTED_FEATURE_COUNT} columns"
            )

        # Check 2: Feature names & ordering match
        if list(X.columns) != self.expected_feature_names:
            raise ValueError("Feature validation failed: Feature column order mismatch")

        # Check 3: Zero target leakage
        if "isFraud" in X.columns:
            raise ValueError("Leakage detection: 'isFraud' target column detected in model feature matrix!")

        # Check 4: Zero identifier leakage
        if "TransactionID" in X.columns:
            raise ValueError("Leakage detection: 'TransactionID' detected in model feature matrix!")

        meta = {
            "transaction_ids": tx_ids,
            "actual_labels": actual_labels,
            "row_count": n_rows,
            "feature_count": self.EXPECTED_FEATURE_COUNT,
            "feature_names_match": True,
        }

        return X, meta
