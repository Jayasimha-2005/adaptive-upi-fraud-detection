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
import math
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
            self.expected_feature_names: List[str] = list(data["feature_names"])

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

        # 4. Pre-compile fast feature schema for sub-millisecond serving
        self._compiled_schema: List[Tuple[str, str, Optional[Dict[str, float]], float]] = []
        cat_maps: Dict[str, Dict[str, float]] = {}
        unseen_codes: Dict[str, float] = {}

        cat_encoders = getattr(self.preprocessor, "_cat_encoders", {})
        for c, le in cat_encoders.items():
            mapping = {str(cls): float(idx) for idx, cls in enumerate(le.classes_)}
            cat_maps[c] = mapping
            unseen_codes[c] = float(mapping.get("UNSEEN", mapping.get("MISSING", 0.0)))

        num_medians: Dict[str, float] = dict(getattr(self.preprocessor, "_num_medians", {}))

        for f_name in self.expected_feature_names:
            if f_name in cat_maps:
                self._compiled_schema.append(("cat", f_name, cat_maps[f_name], unseen_codes[f_name]))
            elif f_name == "hour_sin":
                self._compiled_schema.append(("hour_sin", f_name, None, 0.0))
            elif f_name == "hour_cos":
                self._compiled_schema.append(("hour_cos", f_name, None, 0.0))
            elif f_name == "day_index":
                self._compiled_schema.append(("day_index", f_name, None, 0.0))
            elif f_name.endswith("_missing"):
                base_col = f_name[:-8]
                self._compiled_schema.append(("missing", base_col, None, 1.0))
            else:
                self._compiled_schema.append(("num", f_name, None, float(num_medians.get(f_name, 0.0))))

        logger.info(
            "ServingPreprocessor initialized successfully. Loaded 406 features matching schema."
        )

    def transform_dict(self, raw_dict: Dict[str, Any]) -> np.ndarray:
        """
        Fast-path: directly convert a single transaction dictionary into a canonical
        (1, 406) float64 NumPy feature vector without DataFrame allocation overhead.
        """
        dt_val = raw_dict.get("TransactionDT")
        if dt_val is not None:
            try:
                dt_f = float(dt_val)
                hour = (dt_f % 86400) / 3600.0
                h_sin = math.sin(2 * math.pi * hour / 24)
                h_cos = math.cos(2 * math.pi * hour / 24)
                d_idx = (dt_f - 86400.0) / 86400.0
            except Exception:
                h_sin = h_cos = d_idx = 0.0
        else:
            h_sin = h_cos = d_idx = 0.0

        arr = np.empty((1, self.EXPECTED_FEATURE_COUNT), dtype=np.float64)
        for idx, (f_type, col_name, cat_map, default_val) in enumerate(self._compiled_schema):
            if f_type == "num":
                v = raw_dict.get(col_name)
                if v is None or v == "" or (isinstance(v, float) and np.isnan(v)):
                    arr[0, idx] = default_val
                else:
                    try:
                        arr[0, idx] = float(v)
                    except Exception:
                        arr[0, idx] = default_val
            elif f_type == "cat":
                v = raw_dict.get(col_name)
                if v is None or v == "" or (isinstance(v, float) and np.isnan(v)):
                    s = "MISSING"
                else:
                    s = str(v)
                arr[0, idx] = cat_map.get(s, default_val) if cat_map is not None else default_val
            elif f_type == "missing":
                v = raw_dict.get(col_name)
                if v is None or v == "" or (isinstance(v, float) and np.isnan(v)):
                    arr[0, idx] = 1.0
                else:
                    arr[0, idx] = 0.0
            elif f_type == "hour_sin":
                arr[0, idx] = h_sin
            elif f_type == "hour_cos":
                arr[0, idx] = h_cos
            elif f_type == "day_index":
                arr[0, idx] = d_idx

        return arr

    def transform_dicts(self, raw_dicts: List[Dict[str, Any]]) -> np.ndarray:
        """
        Fast-path: directly convert a batch of N transaction dictionaries into a
        (N, 406) float64 NumPy feature matrix in single contiguous memory.
        """
        n_rows = len(raw_dicts)
        arr = np.empty((n_rows, self.EXPECTED_FEATURE_COUNT), dtype=np.float64)

        for row_i, raw_dict in enumerate(raw_dicts):
            dt_val = raw_dict.get("TransactionDT")
            if dt_val is not None:
                try:
                    dt_f = float(dt_val)
                    hour = (dt_f % 86400) / 3600.0
                    h_sin = math.sin(2 * math.pi * hour / 24)
                    h_cos = math.cos(2 * math.pi * hour / 24)
                    d_idx = (dt_f - 86400.0) / 86400.0
                except Exception:
                    h_sin = h_cos = d_idx = 0.0
            else:
                h_sin = h_cos = d_idx = 0.0

            for col_i, (f_type, col_name, cat_map, default_val) in enumerate(self._compiled_schema):
                if f_type == "num":
                    v = raw_dict.get(col_name)
                    if v is None or v == "" or (isinstance(v, float) and np.isnan(v)):
                        arr[row_i, col_i] = default_val
                    else:
                        try:
                            arr[row_i, col_i] = float(v)
                        except Exception:
                            arr[row_i, col_i] = default_val
                elif f_type == "cat":
                    v = raw_dict.get(col_name)
                    if v is None or v == "" or (isinstance(v, float) and np.isnan(v)):
                        s = "MISSING"
                    else:
                        s = str(v)
                    arr[row_i, col_i] = cat_map.get(s, default_val) if cat_map is not None else default_val
                elif f_type == "missing":
                    v = raw_dict.get(col_name)
                    if v is None or v == "" or (isinstance(v, float) and np.isnan(v)):
                        arr[row_i, col_i] = 1.0
                    else:
                        arr[row_i, col_i] = 0.0
                elif f_type == "hour_sin":
                    arr[row_i, col_i] = h_sin
                elif f_type == "hour_cos":
                    arr[row_i, col_i] = h_cos
                elif f_type == "day_index":
                    arr[row_i, col_i] = d_idx

        return arr

    def transform(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Safely preprocess raw transactions for inference.
        Preserves backward compatibility for full DataFrame inputs.
        """
        df = df_raw.copy()
        n_rows = len(df)

        if "TransactionID" in df.columns:
            tx_ids = df["TransactionID"].tolist()
        else:
            tx_ids = df.index.tolist()

        actual_labels = None
        if "isFraud" in df.columns:
            actual_labels = df["isFraud"].astype(int).tolist()
        else:
            df["isFraud"] = 0

        # Fast vectorization across dataframe records
        dicts = df.to_dict(orient="records")
        X_mat = self.transform_dicts(dicts)
        X = pd.DataFrame(X_mat, columns=self.expected_feature_names, index=df.index)

        meta = {
            "transaction_ids": tx_ids,
            "actual_labels": actual_labels,
            "row_count": n_rows,
            "feature_count": self.EXPECTED_FEATURE_COUNT,
            "feature_names_match": True,
        }

        return X, meta
