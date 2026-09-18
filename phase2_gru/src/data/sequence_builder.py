"""
Phase 2 — E2 GRU Sequence Builder
====================================
Builds (X, y, metadata) sequences from IEEE-CIS processed parquets.

Locked design (sequence_specification.md v1.1):
  - Entity key    : card1 (grouping only, NOT in GRU input)
  - History       : 4 transactions
  - Target        : 5th transaction
  - E2 features   : 405 retained E1 features + 1 causal gap = 406 total
  - Gap feature   : time_since_previous_transaction (T1=0.0, T2=DT2-DT1, ...)
  - Target gap    : NOT included (T5-T4 is forbidden)
  - Split assign  : based on TARGET transaction's split
  - Causal history: prior-split transactions allowed for context

ISOLATION: Reads only from datasets/processed/ieee_cis/ (read-only).
           All output goes to phase2_gru/ only.
           No existing files modified.
"""

from __future__ import annotations

import json
import logging
import pathlib
import math
from typing import Optional

import numpy as np
import pandas as pd

# ── Paths (relative to project root) ─────────────────────────────────────────
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
_PARQUET_DIR  = _PROJECT_ROOT / "datasets" / "processed" / "ieee_cis"
_FEAT_FILE    = _PROJECT_ROOT / "experiments" / "E1_lightgbm" / "feature_names.json"
_MANIFEST     = _PROJECT_ROOT / "phase2_gru" / "reports" / "E2_feature_manifest.json"

# ── Split boundaries (locked from split_meta.json) ───────────────────────────
TRAIN_DT_MAX = 10_972_800
VAL_DT_MAX   = 13_392_000

# ── Sequence parameters (locked) ─────────────────────────────────────────────
HISTORY_LEN    = 4   # L-1 history steps
WINDOW_LEN     = 5   # L = 5 (history + target)
MIN_CARD_TXN   = 5   # minimum card transactions to form any window

ENTITY_KEY     = "card1"
GAP_FEATURE    = "time_since_previous_transaction"

log = logging.getLogger(__name__)


# ── Feature manifest ──────────────────────────────────────────────────────────

def load_e2_features() -> list[str]:
    """
    Load the locked E2 feature list from E2_feature_manifest.json.
    Returns a list of 406 feature names.
    Raises AssertionError if any invariant is violated.
    """
    manifest = json.loads(_MANIFEST.read_text())
    features = manifest["e2_final_features"]

    # Hard assertions — fail loudly on any violation
    assert len(features) == 406,          f"E2 feature count must be 406, got {len(features)}"
    assert "card1"         not in features, "card1 must NOT be in E2 features"
    assert "isFraud"       not in features, "isFraud must NOT be in E2 features"
    assert "TransactionID" not in features, "TransactionID must NOT be in E2 features"
    assert "TransactionDT" not in features, "TransactionDT must NOT be in E2 features"
    assert GAP_FEATURE     in features,     f"'{GAP_FEATURE}' must be in E2 features"

    return features


E2_FEATURES: list[str] = load_e2_features()


# ── Split assignment helper ───────────────────────────────────────────────────

def assign_split(dt: float) -> str:
    """Assign a split label based on TransactionDT."""
    if dt <= TRAIN_DT_MAX:
        return "train"
    elif dt <= VAL_DT_MAX:
        return "validation"
    else:
        return "test"


# ── Data loader ───────────────────────────────────────────────────────────────

def load_all_splits(
    parquet_dir: Optional[pathlib.Path] = None,
) -> pd.DataFrame:
    """
    Load all three processed parquets and concatenate into a single
    chronologically-sorted DataFrame. Adds a 'split' column.

    Returns columns: TransactionID, TransactionDT, card1, isFraud,
                     [405 E1 features minus gap], split
    """
    pdir = parquet_dir or _PARQUET_DIR
    frames = []
    for name in ["train", "validation", "test"]:
        path = pdir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Parquet not found: {path}")
        df = pd.read_parquet(path)
        df["split"] = name
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)

    # Verify no boundary violations
    train_mask = full["split"] == "train"
    val_mask   = full["split"] == "validation"
    test_mask  = full["split"] == "test"

    assert (full.loc[train_mask, "TransactionDT"] <= TRAIN_DT_MAX).all(), \
        "BOUNDARY VIOLATION: train rows exceed TRAIN_DT_MAX"
    assert (full.loc[val_mask, "TransactionDT"] > TRAIN_DT_MAX).all(), \
        "BOUNDARY VIOLATION: val rows not above TRAIN_DT_MAX"
    assert (full.loc[val_mask, "TransactionDT"] <= VAL_DT_MAX).all(), \
        "BOUNDARY VIOLATION: val rows exceed VAL_DT_MAX"
    assert (full.loc[test_mask, "TransactionDT"] > VAL_DT_MAX).all(), \
        "BOUNDARY VIOLATION: test rows not above VAL_DT_MAX"

    log.info(
        "Loaded %d total transactions (train=%d, val=%d, test=%d)",
        len(full),
        train_mask.sum(), val_mask.sum(), test_mask.sum()
    )
    return full


# ── Gap feature computation ───────────────────────────────────────────────────

def compute_gaps(dt_array: np.ndarray) -> np.ndarray:
    """
    Compute causal time gaps for a sorted sequence of TransactionDTs.

    For positions [T1, T2, T3, T4]:
        T1 → 0.0  (no previous within window)
        T2 → DT(T2) - DT(T1)
        T3 → DT(T3) - DT(T2)
        T4 → DT(T4) - DT(T3)

    The TARGET position's gap (T5-T4) is NOT computed here.

    Args:
        dt_array: sorted TransactionDT values for HISTORY positions only
                  (length = HISTORY_LEN = 4)
    Returns:
        gaps: float32 array of shape (HISTORY_LEN,)
    """
    assert len(dt_array) == HISTORY_LEN, \
        f"compute_gaps expects exactly {HISTORY_LEN} timestamps, got {len(dt_array)}"

    gaps = np.zeros(HISTORY_LEN, dtype=np.float32)
    for i in range(1, HISTORY_LEN):
        g = float(dt_array[i]) - float(dt_array[i - 1])
        assert g >= 0, f"Negative gap at position {i}: {g} — timestamps not sorted?"
        # Apply log1p for scale normalisation (documented in spec)
        gaps[i] = math.log1p(g)

    # T1 gap = log1p(0) = 0.0 explicitly
    gaps[0] = 0.0
    return gaps


# ── Core sequence generation ──────────────────────────────────────────────────

def _build_entity_windows(
    entity_df: pd.DataFrame,
    e1_feature_cols: list[str],
) -> list[dict]:
    """
    Build all sliding windows for a single card1 entity.

    entity_df must be sorted by TransactionDT ascending.
    Returns a list of window dicts with keys:
        X        : np.ndarray [HISTORY_LEN, 406]
        y        : int (0 or 1)
        card1    : entity id (metadata only)
        target_tid: TransactionID of T5
        target_dt : TransactionDT of T5
        target_split: 'train'/'validation'/'test'
    """
    n = len(entity_df)
    if n < WINDOW_LEN:
        return []

    # Sort guard — must be strictly ascending by DT (ties broken by TransactionID,
    # but within-window we enforce strictly increasing DTs; windows with equal DTs
    # are EXCLUDED from the primary E2 dataset — see below).
    dts = entity_df["TransactionDT"].values
    assert np.all(np.diff(dts) >= 0), \
        f"Entity not sorted by TransactionDT: {dts}"

    windows = []
    n_excluded_dup_dt = 0
    rows = entity_df.reset_index(drop=True)

    for start in range(n - WINDOW_LEN + 1):
        hist_idx   = list(range(start, start + HISTORY_LEN))
        target_idx = start + HISTORY_LEN

        hist_rows   = rows.iloc[hist_idx]
        target_row  = rows.iloc[target_idx]

        # ── Extract DTs for ordering checks
        hist_dts  = hist_rows["TransactionDT"].values
        target_dt = float(target_row["TransactionDT"])

        # ── OPTION A: Strict temporal ordering required
        # Every DT in [T1, T2, T3, T4, T5] must be STRICTLY increasing.
        # Rationale: same-second transactions (59/12,421 card entities, 146 pairs)
        # cannot be unambiguously ordered in real time. TransactionID provides
        # a deterministic tiebreaker for candidate enumeration ONLY — it does NOT
        # establish true temporal precedence. Any window with equal DTs is
        # EXCLUDED from the primary E2 dataset and counted as n_excluded_dup_dt.
        all_window_dts = np.append(hist_dts, target_dt)
        if not np.all(np.diff(all_window_dts) > 0):
            n_excluded_dup_dt += 1
            continue  # skip this window — non-strict ordering

        # Strict assertion (now guaranteed by the filter above)
        assert np.all(hist_dts < target_dt), \
            f"LEAKAGE: history DT >= target DT after strict filter. " \
            f"hist_dts={hist_dts}, target_dt={target_dt}"


        # ── Target info
        target_split  = assign_split(target_dt)
        target_label  = int(target_row["isFraud"])
        target_tid    = int(target_row["TransactionID"])
        entity_id     = target_row[ENTITY_KEY]

        # ── Build feature matrix for history [HISTORY_LEN, 405]
        hist_features = hist_rows[e1_feature_cols].values.astype(np.float32)
        # STOP if isFraud somehow crept in
        assert hist_features.shape == (HISTORY_LEN, len(e1_feature_cols)), \
            f"Unexpected hist shape: {hist_features.shape}"

        # ── Compute causal gap feature [HISTORY_LEN]
        gaps = compute_gaps(hist_dts)  # shape (HISTORY_LEN,)

        # ── Concatenate: [HISTORY_LEN, 405] + [HISTORY_LEN, 1] = [HISTORY_LEN, 406]
        gap_col = gaps.reshape(-1, 1)
        X = np.concatenate([hist_features, gap_col], axis=1)
        assert X.shape == (HISTORY_LEN, 406), \
            f"Final X shape mismatch: {X.shape}, expected ({HISTORY_LEN}, 406)"

        windows.append({
            "X":            X,
            "y":            target_label,
            "card1":        entity_id,
            "target_tid":   target_tid,
            "target_dt":    target_dt,
            "target_split": target_split,
            "n_excluded_dup_dt": 0,  # per-window field (aggregated at entity level)
        })

    # Attach exclusion count to first window for aggregation (if any windows exist)
    if windows:
        windows[0]["n_excluded_dup_dt"] = n_excluded_dup_dt
    elif n_excluded_dup_dt > 0:
        # All windows excluded — return sentinel
        windows = [{"_all_excluded": True, "n_excluded_dup_dt": n_excluded_dup_dt}]
    return windows


# ── Public API ────────────────────────────────────────────────────────────────

def build_sequences(
    df: pd.DataFrame,
    smoke_test_n: Optional[int] = None,
    verbose: bool = True,
) -> dict[str, dict]:
    """
    Build all sliding-window sequences from the full concatenated DataFrame.

    Args:
        df            : full dataset (all splits concatenated), sorted by card1 + DT
        smoke_test_n  : if set, use only this many unique card1 entities
        verbose       : log progress

    Returns dict with keys 'train', 'validation', 'test', each containing:
        X          : np.ndarray [N_windows, HISTORY_LEN, 406]
        y          : np.ndarray [N_windows]
        card1      : list of entity ids (metadata)
        target_tid : list of TransactionIDs (metadata)
        target_dt  : list of TransactionDTs (metadata)
    """
    # E1 features without card1 and without gap (gap is added per-window)
    e1_feature_cols = [f for f in E2_FEATURES if f != GAP_FEATURE]
    assert len(e1_feature_cols) == 405, \
        f"Expected 405 E1 feature cols (excluding gap), got {len(e1_feature_cols)}"

    # Verify none of the forbidden columns are in e1_feature_cols
    for forbidden in ["isFraud", "TransactionID", "TransactionDT", "card1"]:
        assert forbidden not in e1_feature_cols, \
            f"FORBIDDEN column '{forbidden}' found in e1_feature_cols"

    # Sort entire dataset by card1 then TransactionDT.
    # Tiebreaker: TransactionID (stable, monotonically increasing in IEEE-CIS).
    # Rationale: 59/12,421 card entities have duplicate TransactionDT values
    # (same-second transactions). TransactionID provides a deterministic but
    # arbitrary ordering for ties — documented in sequence_generation_report.md.
    df_sorted = df.sort_values([ENTITY_KEY, "TransactionDT", "TransactionID"]).reset_index(drop=True)

    # Optionally limit to smoke_test_n entities
    all_entities = df_sorted[ENTITY_KEY].unique()
    if smoke_test_n is not None:
        all_entities = all_entities[:smoke_test_n]
        df_sorted = df_sorted[df_sorted[ENTITY_KEY].isin(all_entities)]
        log.info("SMOKE TEST: using %d entities", len(all_entities))

    # Group by card1 and build windows
    result = {s: {"X": [], "y": [], "card1": [], "target_tid": [], "target_dt": []}
              for s in ["train", "validation", "test"]}

    n_entities_with_windows = 0
    n_excluded_dup_dt_total = 0
    all_entities_list = list(df_sorted.groupby(ENTITY_KEY, sort=False))
    n_total_entities = len(all_entities_list)

    for entity_num, (entity_id, entity_df) in enumerate(all_entities_list):
        if entity_num % 1000 == 0:
            log.info("Progress: %d / %d entities processed (%.1f%%)",
                     entity_num, n_total_entities, 100*entity_num/n_total_entities)

        if len(entity_df) < WINDOW_LEN:
            continue
        windows = _build_entity_windows(entity_df, e1_feature_cols)
        if not windows:
            continue

        # Collect exclusion count from sentinel or first window
        if windows[0].get("_all_excluded"):
            n_excluded_dup_dt_total += windows[0]["n_excluded_dup_dt"]
            continue

        n_entities_with_windows += 1
        for i, w in enumerate(windows):
            if i == 0:
                n_excluded_dup_dt_total += w.get("n_excluded_dup_dt", 0)
            if w.get("_all_excluded"):
                continue
            s = w["target_split"]
            result[s]["X"].append(w["X"])
            result[s]["y"].append(w["y"])
            result[s]["card1"].append(w["card1"])
            result[s]["target_tid"].append(w["target_tid"])
            result[s]["target_dt"].append(w["target_dt"])

    if verbose:
        log.info("Entities with >=1 retained window: %d", n_entities_with_windows)
        log.info("Windows excluded (duplicate DT): %d", n_excluded_dup_dt_total)

    # Store exclusion count in result for reporting
    result["_meta"] = {"n_excluded_dup_dt": n_excluded_dup_dt_total}

    # Convert to numpy arrays
    for split_name, data in result.items():
        if split_name == "_meta":
            continue
        if data["X"]:
            data["X"] = np.stack(data["X"], axis=0)  # [N, 4, 406]
            data["y"] = np.array(data["y"], dtype=np.float32)
            if verbose:
                n = len(data["y"])
                n_fraud = int(data["y"].sum())
                log.info(
                    "[%s] windows=%d  fraud=%d (%.2f%%)  legit=%d",
                    split_name, n, n_fraud, 100*n_fraud/n, n-n_fraud
                )
        else:
            data["X"] = np.empty((0, HISTORY_LEN, 406), dtype=np.float32)
            data["y"] = np.empty(0, dtype=np.float32)
            log.warning("[%s] No windows generated", split_name)

    return result


# ── Smoke test ────────────────────────────────────────────────────────────────

def run_smoke_test(n_entities: int = 50) -> dict:
    """Quick smoke test on n_entities entities."""
    log.info("=== SMOKE TEST: %d entities ===", n_entities)
    df = load_all_splits()
    seqs = build_sequences(df, smoke_test_n=n_entities, verbose=True)

    for split_name, data in seqs.items():
        X = data["X"]
        y = data["y"]
        if len(y) > 0:
            assert X.shape[1] == HISTORY_LEN, f"X.shape[1] must be {HISTORY_LEN}"
            assert X.shape[2] == 406,         "X.shape[2] must be 406"
            assert not np.any(np.isnan(X)),   "NaN in X"
            assert not np.any(np.isinf(X)),   "Inf in X"
            log.info("  [%s] X=%s  y_fraud_pct=%.2f%%",
                     split_name, X.shape, 100*y.mean())
    log.info("=== SMOKE TEST PASSED ===")
    return seqs


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    seqs = run_smoke_test(n_entities=100)
    for s, d in seqs.items():
        print(f"  {s}: {d['X'].shape if len(d['y'])>0 else 'empty'}")
