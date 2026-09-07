"""
src/data/split.py
Chronological train/validation/test splitting for IEEE-CIS.

DESIGN PRINCIPLES
-----------------
- Splits are ALWAYS chronological (no random shuffling).
- No transaction from a later time period can appear in an earlier split.
- Split boundaries are defined in config (TransactionDT seconds).
- Returns metadata documenting every split's row counts, fraud rates, and DT ranges.

WHY CHRONOLOGICAL SPLIT?
  Future transaction behaviour must not leak into training.
  A model trained on random rows would see transactions from all time periods,
  including future months. This would inflate all metrics and produce a model
  that cannot generalise under temporal drift.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def chronological_split(
    df: pd.DataFrame,
    train_dt_max: float,
    val_dt_max: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Split ``df`` into train / validation / test using TransactionDT thresholds.

    Parameters
    ----------
    df : pd.DataFrame
        Full joined dataframe containing TransactionDT and isFraud.
    train_dt_max : float
        TransactionDT upper bound (inclusive) for training rows.
    val_dt_max : float
        TransactionDT upper bound (inclusive) for validation rows.
        All rows with TransactionDT > val_dt_max go to test.

    Returns
    -------
    df_train, df_val, df_test : pd.DataFrame
    meta : dict
        Split statistics.

    Raises
    ------
    ValueError
        If temporal ordering is violated (any train DT > val DT, etc.).
    """
    logger.info("Creating chronological split ...")
    logger.info(
        "  Boundaries: train_dt_max=%.0f (day %.1f) | val_dt_max=%.0f (day %.1f)",
        train_dt_max, train_dt_max / 86400,
        val_dt_max,   val_dt_max   / 86400,
    )

    dt = df["TransactionDT"]
    train_mask = dt <= train_dt_max
    val_mask   = (dt > train_dt_max) & (dt <= val_dt_max)
    test_mask  = dt > val_dt_max

    df_train = df[train_mask].copy()
    df_val   = df[val_mask].copy()
    df_test  = df[test_mask].copy()

    # ── Temporal integrity assertions ─────────────────────────────────────────
    assert df_train["TransactionDT"].max() <= train_dt_max, "Train DT exceeds boundary"
    assert df_val["TransactionDT"].min()   >  train_dt_max, "Val DT overlaps train"
    assert df_val["TransactionDT"].max()   <= val_dt_max,   "Val DT exceeds boundary"
    assert df_test["TransactionDT"].min()  >  val_dt_max,   "Test DT overlaps val"

    # ── Overlap check on TransactionIDs ──────────────────────────────────────
    train_ids = set(df_train["TransactionID"])
    val_ids   = set(df_val["TransactionID"])
    test_ids  = set(df_test["TransactionID"])
    tv_overlap = len(train_ids & val_ids)
    vt_overlap = len(val_ids   & test_ids)
    tt_overlap = len(train_ids & test_ids)
    if tv_overlap or vt_overlap or tt_overlap:
        raise ValueError(
            f"TransactionID overlap detected: "
            f"train∩val={tv_overlap}, val∩test={vt_overlap}, train∩test={tt_overlap}"
        )

    def _split_stats(d: pd.DataFrame, name: str) -> dict:
        fraud = int(d["isFraud"].sum())
        total = len(d)
        return {
            "name":       name,
            "rows":       total,
            "fraud":      fraud,
            "legit":      total - fraud,
            "fraud_pct":  round(fraud / total * 100, 4) if total else 0.0,
            "dt_min":     float(d["TransactionDT"].min()) if total else 0.0,
            "dt_max":     float(d["TransactionDT"].max()) if total else 0.0,
            "day_min":    round(float(d["TransactionDT"].min()) / 86400, 1) if total else 0.0,
            "day_max":    round(float(d["TransactionDT"].max()) / 86400, 1) if total else 0.0,
        }

    meta = {
        "train":      _split_stats(df_train, "train"),
        "validation": _split_stats(df_val,   "validation"),
        "test":       _split_stats(df_test,  "test"),
        "train_dt_max_sec": train_dt_max,
        "val_dt_max_sec":   val_dt_max,
        "id_overlap_train_val":   tv_overlap,
        "id_overlap_val_test":    vt_overlap,
        "id_overlap_train_test":  tt_overlap,
    }

    for s, d in [("TRAIN", df_train), ("VAL", df_val), ("TEST", df_test)]:
        st = meta[{"TRAIN": "train", "VAL": "validation", "TEST": "test"}[s]]
        logger.info(
            "  %s: %d rows | fraud=%d (%.2f%%) | days %.1f→%.1f",
            s, st["rows"], st["fraud"], st["fraud_pct"],
            st["day_min"], st["day_max"],
        )

    return df_train, df_val, df_test, meta
