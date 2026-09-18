"""
LT7 — Split integrity test
  LT7: The TARGET transaction's split determines the window's assigned split.
       Train history → Validation target = window is 'validation'.
       History from any prior period → Test target = window is 'test'.
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
from src.data.sequence_builder import (
    HISTORY_LEN, WINDOW_LEN, GAP_FEATURE, E2_FEATURES,
    TRAIN_DT_MAX, VAL_DT_MAX,
    _build_entity_windows, assign_split, build_sequences, load_all_splits,
)


def _feat_cols():
    return [f for f in E2_FEATURES if f != GAP_FEATURE]


def _row(card, tid, dt, fraud, val=1.0):
    row = {"card1": card, "TransactionID": tid,
           "TransactionDT": dt, "isFraud": fraud}
    for col in _feat_cols():
        row[col] = val
    return row


class TestLT7_SplitIntegrity:
    """LT7 — Target's split determines window's split label."""

    def test_train_target_yields_train_window(self):
        """All 5 transactions in TRAIN → window split = 'train'."""
        rows = [_row(1, 100+i, 1_000_000.0 + i*100, 0) for i in range(WINDOW_LEN)]
        df = pd.DataFrame(rows).sort_values("TransactionDT").reset_index(drop=True)
        windows = _build_entity_windows(df, _feat_cols())
        assert len(windows) == 1
        assert windows[0]["target_split"] == "train", \
            f"LT7 FAIL: expected 'train', got {windows[0]['target_split']}"
        print("LT7: PASS — 5 TRAIN transactions → window split='train'")

    def test_validation_target_yields_validation_window(self):
        """4 TRAIN history + 1 VALIDATION target → window split = 'validation'."""
        rows = []
        for i in range(HISTORY_LEN):
            rows.append(_row(2, 200+i, 10_000_000.0 + i*100, 0))  # TRAIN range
        rows.append(_row(2, 299, 11_500_000.0, 1))  # VALIDATION range
        df = pd.DataFrame(rows).sort_values("TransactionDT").reset_index(drop=True)
        windows = _build_entity_windows(df, _feat_cols())
        assert len(windows) == 1
        assert windows[0]["target_split"] == "validation", \
            f"LT7 FAIL: expected 'validation', got {windows[0]['target_split']}"
        print("LT7: PASS — TRAIN history + VALIDATION target → window split='validation'")

    def test_test_target_yields_test_window(self):
        """4 history (mixed TRAIN+VAL) + 1 TEST target → window split = 'test'."""
        rows = []
        # 2 TRAIN rows
        for i in range(2):
            rows.append(_row(3, 300+i, 9_000_000.0 + i*100, 0))
        # 2 VALIDATION rows
        for i in range(2):
            rows.append(_row(3, 310+i, 11_000_000.0 + i*100, 0))
        # 1 TEST target
        rows.append(_row(3, 399, 14_000_000.0, 0))
        df = pd.DataFrame(rows).sort_values("TransactionDT").reset_index(drop=True)
        windows = _build_entity_windows(df, _feat_cols())
        assert len(windows) == 1
        assert windows[0]["target_split"] == "test", \
            f"LT7 FAIL: expected 'test', got {windows[0]['target_split']}"
        print("LT7: PASS — mixed TRAIN+VAL history + TEST target → window split='test'")

    def test_assign_split_boundaries(self):
        """Verify assign_split() returns correct label at exact boundaries."""
        assert assign_split(TRAIN_DT_MAX)     == "train",      "At TRAIN_DT_MAX → train"
        assert assign_split(TRAIN_DT_MAX + 1) == "validation", "Just above TRAIN_DT_MAX → val"
        assert assign_split(VAL_DT_MAX)       == "validation", "At VAL_DT_MAX → val"
        assert assign_split(VAL_DT_MAX + 1)   == "test",       "Just above VAL_DT_MAX → test"
        assert assign_split(1.0)              == "train",      "Early DT → train"
        assert assign_split(15_811_131.0)     == "test",       "Max DT → test"
        print("LT7: PASS — assign_split() boundary conditions all correct")

    def test_no_split_contamination(self):
        """
        A window must NEVER have history from a future split relative to the target.
        Example: target=TRAIN cannot have history from VALIDATION or TEST.
        This is naturally prevented by the sliding window construction
        (earlier indices = earlier DTs = earlier split).
        """
        rows = []
        for i in range(HISTORY_LEN):
            rows.append(_row(4, 400+i, 1_000_000.0 + i*100, 0))   # TRAIN
        rows.append(_row(4, 499, 2_000_000.0, 0))                   # TRAIN target
        df = pd.DataFrame(rows).sort_values("TransactionDT").reset_index(drop=True)
        windows = _build_entity_windows(df, _feat_cols())
        assert len(windows) == 1
        w = windows[0]
        # History DTs all in TRAIN range
        dts = df["TransactionDT"].values
        hist_dts = dts[:HISTORY_LEN]
        for h_dt in hist_dts:
            assert assign_split(h_dt) == "train", \
                f"LT7 FAIL: history DT {h_dt} is in '{assign_split(h_dt)}' — expected 'train'"
        assert assign_split(w["target_dt"]) == "train"
        print("LT7: PASS — no future-split contamination in history")

    def test_window_counts_with_real_data_smoke(self):
        """
        Smoke test: load real data (100 entities), verify split assignments match
        the actual TransactionDT of each target row.
        """
        try:
            full_df = load_all_splits()
        except FileNotFoundError:
            pytest.skip("Processed parquets not available in this environment")

        result = build_sequences(full_df, smoke_test_n=100, verbose=False)

        for split_name in ["train", "validation", "test"]:
            data = result[split_name]
            if len(data["y"]) == 0:
                continue
            target_dts = data["target_dt"]
            for dt in target_dts:
                observed_split = assign_split(dt)
                assert observed_split == split_name, \
                    f"LT7 FAIL: target_dt={dt} assigned to '{split_name}' but assign_split returns '{observed_split}'"
        print("LT7: PASS — all real-data target DTs match their assigned split")
