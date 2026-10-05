"""
serving/hydration/provenance.py
Authoritative 5-tier Feature Provenance Classification for IEEE-CIS / E1 Model.

Every feature evaluated by the E1 LightGBM baseline (406 final tabular features)
is explicitly mapped to its operational source of truth.
"""
from __future__ import annotations

from typing import Dict, List, Set

# ── Provenance Tier Definitions ───────────────────────────────────────────────
TIER_DIRECT_EVENT = "EVENT_DIRECT"             # Directly in the raw streaming Kafka event
TIER_DERIVED_EVENT = "EVENT_DERIVED"           # Deterministically computable from raw event fields
TIER_ENTITY_PROFILE = "ENTITY_PROFILE"         # Semi-static cardholder/account attributes from store
TIER_HISTORICAL_AGG = "HISTORICAL_AGGREGATION" # Point-in-time rolling state (C-counts, D-deltas, Flink windows)
TIER_UNRECONSTRUCTABLE = "UNRECONSTRUCTABLE"   # Cannot be safely reconstructed online without leakage

# ── Direct Event Fields ───────────────────────────────────────────────────────
EVENT_DIRECT_COLS: Set[str] = {
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
}

# ── Event-Derived Features ────────────────────────────────────────────────────
EVENT_DERIVED_COLS: Set[str] = {
    "hour_sin",
    "hour_cos",
    "day_index",
}

# ── Entity & Device Profile Features (Cardholder Store) ───────────────────────
ENTITY_PROFILE_COLS: Set[str] = {
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "dist1",
    "P_emaildomain", "R_emaildomain",
    "DeviceType", "DeviceInfo",
} | {f"id_{i:02d}" for i in range(1, 39)}

# ── Historical Behavioral & Delta Features ────────────────────────────────────
HISTORICAL_AGG_COLS: Set[str] = (
    {f"C{i}" for i in range(1, 15)} |
    {f"D{i}" for i in range(1, 16)} |
    {f"M{i}" for i in range(1, 10)} |
    {f"V{i}" for i in range(1, 340)}
)

# ── Missingness Indicators ────────────────────────────────────────────────────
# Imputed/created by IEEECISPreprocessor
MISSINGNESS_INDICATORS: Set[str] = {
    "dist2_missing", "D5_missing", "D6_missing", "D7_missing",
    "D8_missing", "D9_missing", "D12_missing", "D13_missing", "D14_missing",
    "id_01_missing", "id_02_missing", "id_03_missing", "id_04_missing",
    "id_05_missing", "id_06_missing", "id_07_missing", "id_08_missing",
    "id_09_missing", "id_10_missing", "id_11_missing", "id_14_missing",
    "id_18_missing", "id_21_missing", "id_22_missing", "id_23_missing",
    "id_24_missing", "id_25_missing", "id_26_missing", "id_30_missing",
    "id_31_missing", "id_32_missing", "id_33_missing", "id_34_missing",
    "DeviceInfo_missing",
}


def classify_feature(col_name: str) -> str:
    """
    Classify a feature name into its authoritative provenance tier.
    """
    if col_name in EVENT_DIRECT_COLS:
        return TIER_DIRECT_EVENT
    if col_name in EVENT_DERIVED_COLS or col_name in MISSINGNESS_INDICATORS:
        return TIER_DERIVED_EVENT
    if col_name in ENTITY_PROFILE_COLS:
        return TIER_ENTITY_PROFILE
    if col_name in HISTORICAL_AGG_COLS:
        return TIER_HISTORICAL_AGG
    if col_name in {"isFraud", "is_fraud", "fraud_bool"}:
        return TIER_UNRECONSTRUCTABLE
    return TIER_UNRECONSTRUCTABLE


def get_provenance_manifest() -> Dict[str, List[str]]:
    """
    Return complete dictionary of all feature names mapped by provenance tier.
    """
    return {
        TIER_DIRECT_EVENT: sorted(list(EVENT_DIRECT_COLS)),
        TIER_DERIVED_EVENT: sorted(list(EVENT_DERIVED_COLS | MISSINGNESS_INDICATORS)),
        TIER_ENTITY_PROFILE: sorted(list(ENTITY_PROFILE_COLS)),
        TIER_HISTORICAL_AGG: sorted(list(HISTORICAL_AGG_COLS)),
        TIER_UNRECONSTRUCTABLE: ["isFraud"],
    }
