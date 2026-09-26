"""
data_loader.py — Phase 4 Data Loader
=====================================
Single source of truth for loading BAF Base.csv and splitting it into
the locked temporal periods defined in Protocol v1.1.

PROTOCOL CONSTRAINTS:
- NO random splits of any kind
- Temporal ordering is mandatory
- Month 4 is returned separately (threshold validation ONLY)
- Month 7 is returned separately (protected final evaluation)
- All splits are integer-indexed on the 'month' column only
"""
from __future__ import annotations
import pathlib
import pandas as pd

# ---------------------------------------------------------------------------
# Fixed feature schema — from Protocol v1.1 / PHASE4_PROTOCOL_MANIFEST.json
# ---------------------------------------------------------------------------
EXCLUDED_COLS = {"fraud_bool", "month", "device_fraud_count"}

CATEGORICAL_COLS = [
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
]

NUMERICAL_COLS = [
    "income",
    "name_email_similarity",
    "prev_address_months_count",
    "current_address_months_count",
    "customer_age",
    "days_since_request",
    "intended_balcon_amount",
    "zip_count_4w",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w",
    "credit_risk_score",
    "email_is_free",
    "phone_home_valid",
    "phone_mobile_valid",
    "bank_months_count",
    "has_other_cards",
    "proposed_credit_limit",
    "foreign_request",
    "session_length_in_minutes",
    "keep_alive_session",
    "device_distinct_emails_8w",
]

FEATURE_COLS = NUMERICAL_COLS + CATEGORICAL_COLS  # 24 + 5 = 29
TARGET_COL   = "fraud_bool"
TIME_COL     = "month"

# ---------------------------------------------------------------------------
# Expected integrity values (integrity checks against protocol)
# ---------------------------------------------------------------------------
EXPECTED_COUNTS = {
    0: (132440, 1500),
    1: (127620, 1198),
    2: (136979, 1198),
    3: (150936, 1392),
    4: (127691, 1452),
    5: (119323, 1411),
    6: (108168, 1450),
    7: (96843,  1428),
}
EXPECTED_TRAINING_ROWS  = 547975
EXPECTED_TRAINING_FRAUD = 5288


def load_baf(path: str | pathlib.Path, verify: bool = True) -> pd.DataFrame:
    """
    Load BAF Base.csv with categorical dtype conversion.
    Optionally verifies row/column counts against protocol expectations.
    """
    df = pd.read_csv(path, low_memory=False)

    # Convert categorical columns to pandas Categorical
    for col in CATEGORICAL_COLS:
        df[col] = pd.Categorical(df[col])

    if verify:
        assert len(df) == 1_000_000,       f"Row count mismatch: {len(df)}"
        assert len(df.columns) == 32,      f"Column count mismatch: {len(df.columns)}"
        assert TARGET_COL in df.columns,   "Missing fraud_bool"
        assert TIME_COL in df.columns,     "Missing month"
        assert df[TIME_COL].nunique() == 8, "Expected 8 temporal periods"
        assert set(df[TIME_COL].unique()) == set(range(8)), "Periods must be 0..7"
        for col in FEATURE_COLS:
            assert col in df.columns, f"Missing feature: {col}"

    return df


def get_temporal_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Apply the locked temporal split from Protocol v1.1.

    Returns dict with keys:
        'train'       — Months 0-3 (initial training)
        'val'         — Month 4   (threshold selection ONLY)
        'monitor1'    — Month 5   (drift window 1)
        'eval1'       — Month 6   (evaluation 1 + drift window 2)
        'eval2'       — Month 7   (protected final evaluation)

    NO random split is ever applied.
    """
    return {
        "train":    df[df[TIME_COL].isin([0, 1, 2, 3])].reset_index(drop=True),
        "val":      df[df[TIME_COL] == 4].reset_index(drop=True),
        "monitor1": df[df[TIME_COL] == 5].reset_index(drop=True),
        "eval1":    df[df[TIME_COL] == 6].reset_index(drop=True),
        "eval2":    df[df[TIME_COL] == 7].reset_index(drop=True),
    }


def split_Xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and label vector y. Never includes target/month/device_fraud_count."""
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    # Integrity check: no leakage
    assert TARGET_COL not in X.columns
    assert TIME_COL not in X.columns
    assert "device_fraud_count" not in X.columns
    return X, y


def verify_period_counts(splits: dict[str, pd.DataFrame]) -> None:
    """Assert row and fraud counts match protocol expectations."""
    period_map = {
        "train":    [0, 1, 2, 3],
        "val":      [4],
        "monitor1": [5],
        "eval1":    [6],
        "eval2":    [7],
    }
    for key, months in period_map.items():
        split = splits[key]
        for m in months:
            exp_rows, exp_fraud = EXPECTED_COUNTS[m]
            actual_rows  = len(split[split[TIME_COL] == m])
            actual_fraud = int(split[split[TIME_COL] == m][TARGET_COL].sum())
            assert actual_rows == exp_rows, (
                f"Month {m} rows: expected {exp_rows}, got {actual_rows}"
            )
            assert actual_fraud == exp_fraud, (
                f"Month {m} fraud: expected {exp_fraud}, got {actual_fraud}"
            )

    # Training aggregate
    train = splits["train"]
    assert len(train) == EXPECTED_TRAINING_ROWS, (
        f"Training rows: expected {EXPECTED_TRAINING_ROWS}, got {len(train)}"
    )
    assert int(train[TARGET_COL].sum()) == EXPECTED_TRAINING_FRAUD, (
        f"Training fraud: expected {EXPECTED_TRAINING_FRAUD}, got {int(train[TARGET_COL].sum())}"
    )
