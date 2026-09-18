"""
LT3 — Label isolation test
  LT3: isFraud is absent from the GRU input feature matrix X
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd
import pytest
from src.data.sequence_builder import (
    HISTORY_LEN, GAP_FEATURE, E2_FEATURES,
    _build_entity_windows, load_e2_features,
)


def _make_toy_entity(n=6):
    feat_cols = [f for f in E2_FEATURES if f != GAP_FEATURE]
    rows = []
    for i in range(n):
        row = {"card1": 1, "TransactionID": 1000+i,
               "TransactionDT": 1000.0 + i*100, "isFraud": i % 2}  # alternating label
        for col in feat_cols:
            row[col] = float(i + 1)
        rows.append(row)
    return pd.DataFrame(rows)


class TestLT3_LabelIsolation:
    """LT3 — isFraud is completely absent from X. Only used as y."""

    def test_isfraud_not_in_e2_feature_list(self):
        """Feature manifest must not contain isFraud."""
        assert "isFraud" not in E2_FEATURES, \
            "LT3 FAIL: 'isFraud' found in E2_FEATURES"
        print("LT3: PASS — 'isFraud' not in E2_FEATURES")

    def test_transactionid_not_in_e2_features(self):
        assert "TransactionID" not in E2_FEATURES
        print("LT3: PASS — 'TransactionID' not in E2_FEATURES")

    def test_transactiondt_not_in_e2_features(self):
        assert "TransactionDT" not in E2_FEATURES
        print("LT3: PASS — 'TransactionDT' not in E2_FEATURES")

    def test_card1_not_in_e2_features(self):
        assert "card1" not in E2_FEATURES
        print("LT3: PASS — 'card1' not in E2_FEATURES (entity key only)")

    def test_isfraud_not_passed_as_feature_to_builder(self):
        """
        The e1_feature_cols passed to _build_entity_windows must not contain isFraud.
        Builder uses only E2_FEATURES - GAP_FEATURE for column selection.
        """
        e1_cols = [f for f in E2_FEATURES if f != GAP_FEATURE]
        assert "isFraud" not in e1_cols, \
            "LT3 FAIL: 'isFraud' present in e1_feature_cols"
        print(f"LT3: PASS — e1_feature_cols has {len(e1_cols)} features, no isFraud")

    def test_x_values_not_equal_to_isfraud_column(self):
        """
        In toy data, features are set to float(i+1) and isFraud alternates 0/1.
        The last column (gap) and first 405 columns should never equal the raw
        isFraud column values in a detectable way that indicates leakage.
        More precisely: X[*, :405] uses selected feature columns, not isFraud.
        """
        toy = _make_toy_entity(n=6)
        feat_cols = [f for f in E2_FEATURES if f != GAP_FEATURE]
        windows = _build_entity_windows(toy, feat_cols)

        for w in windows:
            X = w["X"]   # [4, 406]
            y = w["y"]   # scalar
            # Verify y is 0 or 1
            assert y in (0, 1), f"LT3 FAIL: y={y} is not 0 or 1"
            # The label is NOT in X — X[*,0] is TransactionAmt, not isFraud
            # In our toy: TransactionAmt = float(i+1) ∈ {1.0, 2.0, ...} not ∈ {0,1}
            assert not np.all(X[:, 0] <= 1.0), \
                "LT3 WARNING: first feature column looks like binary — verify not isFraud"
        print("LT3: PASS — X values consistent with transaction features, not labels")

    def test_feature_manifest_json_confirms_exclusion(self):
        """Read E2_feature_manifest.json and verify isFraud excluded."""
        manifest_path = pathlib.Path(__file__).resolve().parents[1] / \
                        "reports" / "E2_feature_manifest.json"
        assert manifest_path.exists(), "E2_feature_manifest.json not found"
        manifest = json.loads(manifest_path.read_text())

        excluded = manifest.get("excluded_from_gru_input", {})
        assert "isFraud" in excluded, \
            "LT3 FAIL: 'isFraud' not listed in excluded_from_gru_input in manifest"
        assert manifest["e2_final_feature_count"] == 406
        print(f"LT3: PASS — manifest confirms isFraud excluded, final count={manifest['e2_final_feature_count']}")

    def test_e2_feature_count_is_406(self):
        """Final E2 feature count must be exactly 406."""
        features = load_e2_features()
        assert len(features) == 406, f"LT3 FAIL: E2 features = {len(features)}, expected 406"
        print(f"LT3: PASS — E2 feature count = {len(features)}")
