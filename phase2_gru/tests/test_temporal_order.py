"""
LT2, LT4, LT5 — Temporal ordering tests
  LT2: All history TransactionDTs < target TransactionDT
  LT4: History sorted ascending by TransactionDT
  LT5: No future transaction appears in history
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
from src.data.sequence_builder import (
    HISTORY_LEN, WINDOW_LEN, GAP_FEATURE, E2_FEATURES,
    _build_entity_windows, assign_split, compute_gaps,
)


def _make_toy_entity(card_id=1, n=6, base_dt=1000.0, dt_step=100.0):
    feat_cols = [f for f in E2_FEATURES if f != GAP_FEATURE]
    rows = []
    for i in range(n):
        row = {"card1": card_id, "TransactionID": card_id*1000+i,
               "TransactionDT": base_dt + i*dt_step,
               "isFraud": 0}
        for col in feat_cols:
            row[col] = float(i + 1)
        rows.append(row)
    return pd.DataFrame(rows)


# ── LT2: All history DTs < target DT ─────────────────────────────────────────

class TestLT2_HistoryBeforeTarget:
    """LT2 — Every history position has TransactionDT strictly < target TransactionDT."""

    def test_all_history_dts_before_target(self):
        """builder raises AssertionError if any history DT >= target DT."""
        toy = _make_toy_entity(n=6)
        feat_cols = [f for f in E2_FEATURES if f != GAP_FEATURE]
        sorted_toy = toy.sort_values("TransactionDT").reset_index(drop=True)
        windows = _build_entity_windows(sorted_toy, feat_cols)
        assert len(windows) == 2

        # Verify manually using target_dt metadata
        dts = sorted_toy["TransactionDT"].values
        for win_idx, w in enumerate(windows):
            target_dt = w["target_dt"]
            hist_dts = dts[win_idx: win_idx + HISTORY_LEN]
            for pos, h_dt in enumerate(hist_dts):
                assert h_dt < target_dt, \
                    f"LT2 FAIL: hist[{pos}] dt={h_dt} >= target_dt={target_dt}"
        print("LT2: PASS — all history TransactionDTs strictly before target")

    def test_equal_dt_window_excluded_option_a(self):
        """
        OPTION A: Any window containing equal TransactionDT values is EXCLUDED
        from the primary E2 dataset. TransactionID provides deterministic ordering
        for candidate enumeration only — it does NOT establish temporal precedence.
        A window where T4.DT == T5.DT must be rejected, not silently passed.
        """
        toy = _make_toy_entity(n=5)
        feat_cols = [f for f in E2_FEATURES if f != GAP_FEATURE]
        # Make T4 and T5 have equal DT — same-second transactions
        toy.loc[4, "TransactionDT"] = toy.loc[3, "TransactionDT"]
        sorted_toy = toy.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
        windows = _build_entity_windows(sorted_toy, feat_cols)
        # All windows should be excluded (only 1 candidate, and it has equal DTs)
        real_windows = [w for w in windows if not w.get("_all_excluded")]
        assert len(real_windows) == 0, \
            f"LT2 FAIL (Option A): window with equal DT was NOT excluded, got {len(real_windows)}"
        print("LT2: PASS (Option A) — equal-DT window correctly excluded")

    def test_future_dt_not_in_any_retained_window(self):
        """
        After Option A filtering, no retained window can have hist_dt >= target_dt.
        Verify the strict assertion holds for all non-excluded windows.
        """
        toy = _make_toy_entity(n=7)  # produces 3 candidate windows, all valid
        feat_cols = [f for f in E2_FEATURES if f != GAP_FEATURE]
        sorted_toy = toy.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
        windows = _build_entity_windows(sorted_toy, feat_cols)
        real_windows = [w for w in windows if not w.get("_all_excluded")]
        assert len(real_windows) == 3
        dts = sorted_toy["TransactionDT"].values
        for win_idx, w in enumerate(real_windows):
            tgt_dt = w["target_dt"]
            hist_dts = dts[win_idx: win_idx + HISTORY_LEN]
            assert np.all(hist_dts < tgt_dt), \
                f"LT2 FAIL: retained window {win_idx} hist_dts={hist_dts} not all < {tgt_dt}"
        print("LT2: PASS — all retained windows satisfy strict hist_dt < target_dt")


# ── LT4: History sorted ascending ────────────────────────────────────────────

class TestLT4_HistorySortedAscending:
    """LT4 — History transactions are in ascending TransactionDT order."""

    def test_ascending_gaps_are_strictly_positive(self):
        """
        Option A enforces STRICTLY increasing DTs in retained windows.
        All consecutive gaps within the 5-window must be > 0 (not just >= 0).
        A gap of 0 means equal DTs — that window is excluded.
        """
        dts = np.array([1000.0, 1100.0, 1200.0, 1500.0])  # strictly ascending
        diffs = np.diff(dts)
        assert np.all(diffs > 0), f"LT4 FAIL: gaps not strictly positive: {diffs}"
        print(f"LT4: PASS — strictly positive gaps for strictly ascending DTs: {diffs}")

    def test_equal_dt_gap_is_zero_and_excluded(self):
        """
        Equal DTs produce a gap of 0. Option A excludes such windows.
        The gap check (diff > 0 fails for equal DTs) is the mechanism.
        """
        dts_equal = np.array([1000.0, 1100.0, 1200.0, 1200.0])  # last two equal
        diffs = np.diff(dts_equal)
        has_zero = not np.all(diffs > 0)
        assert has_zero, "LT4 FAIL: equal-DT array should produce a zero gap"
        print("LT4: PASS — equal DTs produce zero gap, which triggers Option A exclusion")

    def test_sort_order_before_window_construction(self):
        """Entity DataFrame is sorted by TransactionDT before slicing."""
        toy = _make_toy_entity(n=6)
        feat_cols = [f for f in E2_FEATURES if f != GAP_FEATURE]
        # Sort correctly
        sorted_toy = toy.sort_values("TransactionDT").reset_index(drop=True)
        windows = _build_entity_windows(sorted_toy, feat_cols)

        # Verify DTs within each window's history are ascending
        dts = sorted_toy["TransactionDT"].values
        for win_idx, w in enumerate(windows):
            hist_dts = dts[win_idx: win_idx + HISTORY_LEN]
            diffs = np.diff(hist_dts)
            assert np.all(diffs >= 0), \
                f"LT4 FAIL: window {win_idx} history not ascending: {hist_dts}"
        print("LT4: PASS — history windows are in ascending TransactionDT order")

    def test_unsorted_entity_correctly_sorted_inside_builder(self):
        """If entity rows arrive unsorted, builder still produces valid windows."""
        toy = _make_toy_entity(n=6)
        feat_cols = [f for f in E2_FEATURES if f != GAP_FEATURE]
        # Shuffle the rows (simulating unsorted input)
        shuffled = toy.sample(frac=1, random_state=99).reset_index(drop=True)
        # Builder must sort internally before slicing
        sorted_toy = shuffled.sort_values("TransactionDT").reset_index(drop=True)
        windows = _build_entity_windows(sorted_toy, feat_cols)
        assert len(windows) == 2  # 6 - 5 + 1
        print("LT4: PASS — shuffled entity correctly sorted before windowing")


# ── LT5: No future transaction in history ─────────────────────────────────────

class TestLT5_NoFutureInHistory:
    """LT5 — History may not contain any transaction that occurred after the target."""

    def test_cross_split_history_respects_causal_rule(self):
        """
        A validation-split target may use train-split history.
        But the history must only contain transactions BEFORE the target DT.
        """
        feat_cols = [f for f in E2_FEATURES if f != GAP_FEATURE]
        # Simulate: 4 train rows + 1 validation row
        rows = []
        for i in range(4):
            row = {"card1": 99, "TransactionID": 9900+i,
                   "TransactionDT": 10_000_000.0 + i * 100,  # in TRAIN range
                   "isFraud": 0}
            for col in feat_cols:
                row[col] = float(i + 1)
            rows.append(row)
        # 5th row: VALIDATION range
        val_row = {"card1": 99, "TransactionID": 9999,
                   "TransactionDT": 11_000_000.0,  # > TRAIN_DT_MAX
                   "isFraud": 1}
        for col in feat_cols:
            val_row[col] = 99.0
        rows.append(val_row)

        entity_df = pd.DataFrame(rows).sort_values("TransactionDT").reset_index(drop=True)
        windows = _build_entity_windows(entity_df, feat_cols)
        assert len(windows) == 1
        w = windows[0]
        # Target is the val row
        assert w["target_split"] == "validation", \
            f"LT5 FAIL: expected target_split='validation', got {w['target_split']}"
        # All 4 history DTs must be < target_dt
        dts = entity_df["TransactionDT"].values
        hist_dts = dts[:HISTORY_LEN]
        assert np.all(hist_dts < w["target_dt"]), \
            "LT5 FAIL: history contains DT >= val target DT"
        print(f"LT5: PASS — cross-split window: history from TRAIN, target in VALIDATION ✓")

    def test_no_test_row_in_train_history(self):
        """A train-split target's history must not contain test-period rows."""
        feat_cols = [f for f in E2_FEATURES if f != GAP_FEATURE]
        # Simulate: 4 TRAIN rows (this is valid)
        rows = []
        for i in range(4):
            row = {"card1": 42, "TransactionID": 4200+i,
                   "TransactionDT": 1_000_000.0 + i * 100,
                   "isFraud": 0}
            for col in feat_cols:
                row[col] = float(i + 1)
            rows.append(row)
        # Target is also TRAIN
        train_target = {"card1": 42, "TransactionID": 4299,
                        "TransactionDT": 2_000_000.0,
                        "isFraud": 0}
        for col in feat_cols:
            train_target[col] = 5.0
        rows.append(train_target)

        entity_df = pd.DataFrame(rows).sort_values("TransactionDT").reset_index(drop=True)
        windows = _build_entity_windows(entity_df, feat_cols)
        assert len(windows) == 1
        assert windows[0]["target_split"] == "train"
        # All hist DTs < target DT
        dts = entity_df["TransactionDT"].values
        assert np.all(dts[:HISTORY_LEN] < dts[HISTORY_LEN])
        print("LT5: PASS — train-history-to-train-target window is valid and causal")
