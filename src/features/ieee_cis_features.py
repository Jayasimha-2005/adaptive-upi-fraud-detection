"""
src/features/ieee_cis_features.py
Feature engineering and leakage-safe preprocessing for IEEE-CIS Phase 1.

DESIGN PRINCIPLES
-----------------
- All preprocessing objects (imputers, encoders, scalers) are fit ONLY on training data.
- Validation and test data are only transformed (never fit).
- Every dropped or engineered column is documented.
- No future information is used.
- TransactionID and isFraud are never included in X.

FEATURE GROUPS (per forensic analysis)
---------------------------------------
A. Transaction:     TransactionAmt, ProductCD
B. Card:            card1–card6
C. Address:         addr1, addr2
D. Email:           P_emaildomain, R_emaildomain
E. Distance:        dist1 (dist2 dropped: 93.6% missing)
F. C-columns:       C1–C14 (counting behavioural aggregates, 0% missing)
G. D-columns:       D1, D2, D3, D4, D5, D10, D11, D15 (others dropped >80% missing)
H. M-columns:       M1–M9 (match flags)
I. V-columns:       V1–V95 low-missing band; high-missing bands dropped
J. Identity/device: id_01–id_38, DeviceType, DeviceInfo (from identity join)
K. Temporal:        hour_sin, hour_cos, day_index (from TransactionDT)
L. Missingness:     {col}_missing indicators for key columns
"""
from __future__ import annotations

import gc
import logging
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ── Column definitions (based on forensic analysis) ───────────────────────────

# D-columns safe to use (<= 80% missing)
SAFE_D_COLS = ["D1", "D2", "D3", "D4", "D5", "D10", "D11", "D15"]

# D-columns to drop (>80% missing: D6,D7,D8,D9,D12,D13,D14)
DROP_D_COLS = ["D6", "D7", "D8", "D9", "D12", "D13", "D14"]

# dist2 drops separately (93.6% missing)
DROP_OTHER = ["dist2"]

# Identifiers and target — never in X
ALWAYS_DROP = ["isFraud", "TransactionID", "TransactionDT"]

# Categorical columns for label encoding
CATEGORICAL_COLS = [
    "ProductCD",
    "card4", "card6",
    "P_emaildomain", "R_emaildomain",
    "id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29",
    "id_30", "id_31", "id_33", "id_34", "id_35", "id_36", "id_37", "id_38",
    "DeviceType", "DeviceInfo",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
]

# V-column missingness threshold for dropping (set in config, default 0.80)
V_COL_MISS_THRESHOLD = 0.80

# Columns that warrant missingness indicators (>5% missing — from forensic analysis)
MISSINGNESS_INDICATOR_COLS = [
    "D2", "D3", "D4", "D5", "D10", "D11", "D15",
    "addr1", "addr2", "dist1",
    "P_emaildomain", "R_emaildomain",
    "card2", "card3", "card5",
    "DeviceInfo", "DeviceType",
    "id_03", "id_04", "id_05", "id_06", "id_09", "id_10",
    "id_14", "id_18", "id_30", "id_33",
]


# ── Public API ────────────────────────────────────────────────────────────────

class IEEECISPreprocessor:
    """
    Leakage-safe preprocessing pipeline for IEEE-CIS.

    Usage
    -----
    pp = IEEECISPreprocessor(cfg)
    pp.fit(df_train)
    X_train, y_train = pp.transform(df_train)
    X_val,   y_val   = pp.transform(df_val)
    X_test,  y_test  = pp.transform(df_test)
    pp.save("experiments/E1_lightgbm/preprocessing.joblib")
    """

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self._is_fitted = False

        # State learned from training data only
        self._drop_cols: list[str] = []
        self._v_drop_cols: list[str] = []
        self._num_medians: dict[str, float] = {}
        self._cat_encoders: dict[str, LabelEncoder] = {}
        self._feature_cols: list[str] = []
        self._leakage_log: list[dict] = []

    # ── Fit (training data only) ──────────────────────────────────────────────

    def fit(self, df_train: pd.DataFrame) -> "IEEECISPreprocessor":
        """
        Learn all preprocessing parameters from training data.
        Must be called before transform().
        """
        logger.info("Fitting preprocessor on training data (%d rows) ...", len(df_train))
        n = len(df_train)

        # 1. Identify V-columns to drop (>80% missing IN TRAINING)
        v_threshold = self.cfg.get("features", {}).get("v_column_drop_missing_threshold", 0.80)
        all_v = [c for c in df_train.columns if c.startswith("V")]
        self._v_drop_cols = [
            c for c in all_v
            if df_train[c].isnull().mean() > v_threshold
        ]
        logger.info("  V-cols to drop (>%.0f%% missing in train): %d", v_threshold * 100, len(self._v_drop_cols))
        for c in self._v_drop_cols:
            self._leakage_log.append({
                "column": c, "action": "DROP",
                "reason": f">={v_threshold*100:.0f}% missing in training data",
            })

        # 2. Compute numerical medians (fit on TRAIN only)
        logger.info("  Computing numerical medians ...")
        drop_set = (
            set(ALWAYS_DROP) | set(DROP_D_COLS) | set(DROP_OTHER) |
            set(self._v_drop_cols) | set(CATEGORICAL_COLS)
        )
        num_cols = [
            c for c in df_train.columns
            if c not in drop_set and df_train[c].dtype in (np.float64, np.float32, np.int64, np.int32, np.int16)
        ]
        for c in num_cols:
            median_val = df_train[c].median()
            self._num_medians[c] = float(median_val) if not math.isnan(median_val) else 0.0

        # 3. Fit categorical encoders (LabelEncoder on TRAIN only)
        logger.info("  Fitting label encoders for %d categorical columns ...", len(CATEGORICAL_COLS))
        for c in CATEGORICAL_COLS:
            if c not in df_train.columns:
                continue
            le = LabelEncoder()
            # Fill missing with "MISSING" before encoding
            series = df_train[c].fillna("MISSING").astype(str)
            le.fit(series)
            # Add "MISSING" and "UNSEEN" to classes so transform doesn't fail
            extra = []
            if "MISSING" not in le.classes_:
                extra.append("MISSING")
            if "UNSEEN" not in le.classes_:
                extra.append("UNSEEN")
            if extra:
                le.classes_ = np.concatenate([le.classes_, extra])
            self._cat_encoders[c] = le

        # 4. Build final feature list (done after we know what we drop)
        self._feature_cols = self._build_feature_list(df_train)
        logger.info("  Final feature count: %d", len(self._feature_cols))

        self._is_fitted = True
        return self

    # ── Transform ─────────────────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Transform a dataframe (train, val, or test) using fitted parameters.

        Returns
        -------
        X : pd.DataFrame  (shape: n_rows × len(feature_cols))
        y : pd.Series     (isFraud)
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor is not fitted. Call fit() first.")

        df = df.copy()
        y = df["isFraud"].astype(int)

        # ── Temporal features from TransactionDT ──────────────────────────────
        if self.cfg.get("features", {}).get("create_dt_features", True):
            df = _create_temporal_features(df)

        # ── Missingness indicators ─────────────────────────────────────────────
        if self.cfg.get("features", {}).get("create_missingness_indicators", True):
            threshold = self.cfg.get("features", {}).get("missingness_indicator_threshold", 0.05)
            df = _create_missingness_indicators(df, MISSINGNESS_INDICATOR_COLS)

        # ── Numerical imputation (training medians) ────────────────────────────
        for c, med in self._num_medians.items():
            if c in df.columns:
                df[c] = df[c].fillna(med)

        # ── Categorical encoding ───────────────────────────────────────────────
        for c, le in self._cat_encoders.items():
            if c not in df.columns:
                continue
            series = df[c].fillna("MISSING").astype(str)
            # Map unseen categories to "UNSEEN"
            known = set(le.classes_)
            series = series.apply(lambda x: x if x in known else "UNSEEN")
            df[c] = le.transform(series)

        # ── Select and order feature columns ──────────────────────────────────
        # Only keep columns that were present during fit and are in self._feature_cols
        avail = [c for c in self._feature_cols if c in df.columns]
        X = df[avail].copy()

        return X, y

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save the fitted preprocessor to disk using joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Preprocessor saved: %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "IEEECISPreprocessor":
        """Load a previously saved preprocessor."""
        return joblib.load(Path(path))

    @property
    def feature_names(self) -> list[str]:
        return self._feature_cols

    @property
    def leakage_log(self) -> list[dict]:
        return self._leakage_log

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_feature_list(self, df: pd.DataFrame) -> list[str]:
        """
        Return the ordered list of feature columns for X.
        Called once during fit().
        """
        exclude = (
            set(ALWAYS_DROP) |
            set(DROP_D_COLS) |
            set(DROP_OTHER) |
            set(self._v_drop_cols)
        )

        # Temporal feature names (will be created by transform)
        temporal = ["hour_sin", "hour_cos", "day_index"]

        # Missingness indicator names
        miss_names = [f"{c}_missing" for c in MISSINGNESS_INDICATOR_COLS]

        # All original columns (excluding drops)
        orig = [c for c in df.columns if c not in exclude]

        # Combine and deduplicate while preserving order
        all_cols = orig + temporal + miss_names
        seen = set()
        result = []
        for c in all_cols:
            if c not in seen:
                seen.add(c)
                result.append(c)

        # Log all dropped columns to leakage_log
        for c in DROP_D_COLS:
            self._leakage_log.append({
                "column": c, "action": "DROP",
                "reason": ">80% missing in IEEE-CIS (measured in forensic analysis)",
            })
        self._leakage_log.append({
            "column": "dist2", "action": "DROP",
            "reason": "93.6% missing (measured in forensic analysis)",
        })
        self._leakage_log.append({
            "column": "TransactionID", "action": "DROP",
            "reason": "Identifier — not a predictive feature",
        })
        self._leakage_log.append({
            "column": "isFraud", "action": "DROP",
            "reason": "Target — must not appear in X",
        })
        self._leakage_log.append({
            "column": "TransactionDT", "action": "TRANSFORM",
            "reason": "Converted to hour_sin/cos + day_index; raw DT excluded from X to prevent temporal positional memorisation",
        })

        return result


# ── Helper functions ──────────────────────────────────────────────────────────

def _create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive temporal features from TransactionDT (seconds since reference epoch).

    Features created:
    - hour_sin, hour_cos : cyclic encoding of hour-of-day (0–23)
    - day_index          : integer day number (0-based from start of dataset)

    WHY cyclic encoding?
      Hour 23 and hour 0 are adjacent in reality, but numerically distant.
      sin/cos encoding preserves this neighbourhood.

    WHY NOT use raw TransactionDT as a feature?
      Using the raw monotone timestamp as a feature would let the model learn
      "time → fraud" directly instead of learning fraud patterns. In deployment
      the absolute timestamp has no meaning — only temporal patterns matter.
    """
    dt = df["TransactionDT"]
    dt_min = 86400.0   # confirmed start from forensic analysis (day 1)

    # Hour of day (0–23)
    hour = ((dt % 86400) / 3600).astype(float)
    df["hour_sin"] = np.sin(2 * math.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * math.pi * hour / 24)

    # Day index (0-based from dataset start)
    df["day_index"] = ((dt - dt_min) / 86400).astype(float)

    return df


def _create_missingness_indicators(
    df: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame:
    """
    Create binary 0/1 indicator columns for specified missing-value columns.
    Column {col}_missing = 1 if the original value was NaN, else 0.

    WHY missingness indicators?
      Forensic analysis showed several columns have differential fraud rates
      depending on whether the value is present (e.g. D7 missing → fraud 12.9%
      vs D7 present → fraud 2.3%).
    """
    for c in cols:
        if c in df.columns:
            df[f"{c}_missing"] = df[c].isnull().astype(np.int8)
    return df


def get_feature_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a feature inventory classifying every column in the joined dataframe.
    For documentation and leakage audit purposes.
    """
    rows = []
    all_cols = list(df.columns)
    v_cols = [c for c in all_cols if c.startswith("V")]
    c_cols = [c for c in all_cols if c.startswith("C") and len(c) <= 3]
    d_cols = [c for c in all_cols if c.startswith("D") and len(c) <= 3]
    m_cols = [c for c in all_cols if c.startswith("M") and len(c) <= 3]

    def classify(col: str) -> str:
        if col == "isFraud":            return "M_TARGET"
        if col == "TransactionID":      return "L_IDENTIFIER"
        if col == "TransactionDT":      return "K_TEMPORAL"
        if col == "TransactionAmt":     return "A_TRANSACTION"
        if col == "ProductCD":          return "A_TRANSACTION"
        if col.startswith("card"):      return "B_CARD"
        if col.startswith("addr"):      return "C_ADDRESS"
        if "email" in col.lower():      return "D_EMAIL"
        if col.startswith("dist"):      return "E_DISTANCE"
        if col in c_cols:               return "F_COUNTING"
        if col in d_cols:               return "G_TIMEDELTA"
        if col in m_cols:               return "H_MATCH_FLAG"
        if col in v_cols:               return "I_ANONYMIZED_V"
        if col.startswith("id_"):       return "J_IDENTITY"
        if col in ("DeviceType", "DeviceInfo"): return "J_IDENTITY"
        return "Z_OTHER"

    for col in all_cols:
        miss_pct = round(df[col].isnull().mean() * 100, 2) if col in df.columns else 0.0
        rows.append({
            "column":        col,
            "group":         classify(col),
            "dtype":         str(df[col].dtype),
            "missing_pct":   miss_pct,
            "drop_reason":   (
                "High missingness (>80%)" if col in DROP_D_COLS or col in DROP_OTHER or col in ["D6","D7","D8","D9","D12","D13","D14","dist2"]
                else "Target" if col == "isFraud"
                else "Identifier" if col == "TransactionID"
                else "Converted to cyclic features" if col == "TransactionDT"
                else ""
            ),
        })
    return pd.DataFrame(rows)
