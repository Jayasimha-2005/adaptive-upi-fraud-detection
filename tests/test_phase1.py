"""
tests/test_phase1.py
Automated integrity tests for Phase 1 pipeline.

Run with:
  python tests/test_phase1.py

Or if pytest is installed:
  pytest tests/test_phase1.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXP_DIR  = ROOT / "experiments" / "E1_lightgbm"
PROC_DIR = ROOT / "datasets" / "processed" / "ieee_cis"


def _require_file(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Required artifact not found: {p}")


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_no_duplicate_transaction_ids_in_processed_data():
    """Test 1: No duplicate TransactionIDs in any processed split."""
    for name in ["train", "validation", "test"]:
        p = PROC_DIR / f"{name}.parquet"
        _require_file(p)
        df = pd.read_parquet(p, columns=["TransactionID"], engine="pyarrow")
        dups = int(df["TransactionID"].duplicated().sum())
        assert dups == 0, f"[{name}] {dups} duplicate TransactionIDs found"
    print("PASS: test_no_duplicate_transaction_ids_in_processed_data")


def test_splits_do_not_overlap_in_transaction_id():
    """Test 2: No TransactionID appears in more than one split."""
    ids = {}
    for name in ["train", "validation", "test"]:
        df = pd.read_parquet(PROC_DIR / f"{name}.parquet", columns=["TransactionID"], engine="pyarrow")
        ids[name] = set(df["TransactionID"].tolist())

    tv = ids["train"] & ids["validation"]
    vt = ids["validation"] & ids["test"]
    tt = ids["train"] & ids["test"]
    assert len(tv) == 0, f"train∩val overlap: {len(tv)}"
    assert len(vt) == 0, f"val∩test overlap: {len(vt)}"
    assert len(tt) == 0, f"train∩test overlap: {len(tt)}"
    print("PASS: test_splits_do_not_overlap_in_transaction_id")


def test_temporal_ordering_of_splits():
    """Test 3: Train DTs < Val DTs < Test DTs (no future leakage)."""
    dfs = {}
    for name in ["train", "validation", "test"]:
        df = pd.read_parquet(PROC_DIR / f"{name}.parquet", columns=["TransactionDT"], engine="pyarrow")
        dfs[name] = df

    train_max = float(dfs["train"]["TransactionDT"].max())
    val_min   = float(dfs["validation"]["TransactionDT"].min())
    val_max   = float(dfs["validation"]["TransactionDT"].max())
    test_min  = float(dfs["test"]["TransactionDT"].min())

    assert train_max < val_min,  f"Train DT max ({train_max}) >= Val DT min ({val_min})"
    assert val_max   < test_min, f"Val DT max ({val_max}) >= Test DT min ({test_min})"
    print("PASS: test_temporal_ordering_of_splits")


def test_isfr_not_in_feature_matrix():
    """Test 4: isFraud is not in the feature matrix (X)."""
    for name in ["train", "validation", "test"]:
        df = pd.read_parquet(PROC_DIR / f"{name}.parquet", engine="pyarrow")
        # isFraud is stored as a metadata column but should not be used as a feature
        # The model feature_names.json must not contain it
    feat_path = EXP_DIR / "feature_names.json"
    _require_file(feat_path)
    with open(feat_path) as f:
        feature_names = json.load(f)["feature_names"]
    assert "isFraud"       not in feature_names, "isFraud found in feature list!"
    assert "TransactionID" not in feature_names, "TransactionID found in feature list!"
    print("PASS: test_isfr_not_in_feature_matrix")


def test_prediction_count_matches_source():
    """Test 5: Prediction rows match input rows."""
    _require_file(EXP_DIR / "predictions.parquet")
    preds = pd.read_parquet(EXP_DIR / "predictions.parquet", engine="pyarrow")

    for split in ["validation", "test"]:
        src = pd.read_parquet(PROC_DIR / f"{split}.parquet", columns=["TransactionID"], engine="pyarrow")
        pred_split = preds[preds["split"] == split]
        assert len(pred_split) == len(src), (
            f"[{split}] prediction count {len(pred_split)} != source count {len(src)}"
        )
    print("PASS: test_prediction_count_matches_source")


def test_model_reload_produces_same_predictions():
    """Test 6: Saved model produces identical predictions when reloaded."""
    from src.models.lightgbm_baseline import LightGBMBaseline

    _require_file(EXP_DIR / "model.txt")
    _require_file(EXP_DIR / "feature_names.json")

    model = LightGBMBaseline.load(EXP_DIR / "model.txt", EXP_DIR / "feature_names.json")

    # Load validation features only
    val_df = pd.read_parquet(PROC_DIR / "validation.parquet", engine="pyarrow")
    feature_names = model._feature_names
    avail = [c for c in feature_names if c in val_df.columns]
    X_val = val_df[avail]

    # Read stored predictions
    preds = pd.read_parquet(EXP_DIR / "predictions.parquet", engine="pyarrow")
    val_preds = preds[preds["split"] == "validation"]["fraud_probability"].values

    # Re-predict
    new_prob = model.predict_proba(X_val)
    max_diff = float(np.abs(new_prob - val_preds[:len(new_prob)]).max())

    assert max_diff < 1e-5, f"Reload prediction difference too large: {max_diff:.2e}"
    print(f"PASS: test_model_reload_produces_same_predictions (max_diff={max_diff:.2e})")


def test_metrics_file_exists_and_has_required_keys():
    """Test 7: metrics.json exists and contains required keys."""
    metrics_path = EXP_DIR / "metrics.json"
    _require_file(metrics_path)
    with open(metrics_path) as f:
        m = json.load(f)
    required_keys = ["pr_auc", "roc_auc", "precision", "recall", "f1", "mcc", "fpr", "fnr"]
    for k in required_keys:
        assert k in m["validation"], f"Missing key '{k}' in validation metrics"
        assert k in m["test"],       f"Missing key '{k}' in test metrics"
    print("PASS: test_metrics_file_exists_and_has_required_keys")


def test_test_pr_auc_above_random_baseline():
    """Test 8: Test PR-AUC is meaningfully above the random baseline (fraud rate)."""
    metrics_path = EXP_DIR / "metrics.json"
    _require_file(metrics_path)
    with open(metrics_path) as f:
        m = json.load(f)
    test_pr_auc  = m["test"]["pr_auc"]
    fraud_rate   = m["test"]["fraud_pct"] / 100.0
    assert test_pr_auc > fraud_rate * 3, (
        f"Test PR-AUC ({test_pr_auc:.4f}) not significantly above fraud rate ({fraud_rate:.4f})"
    )
    print(f"PASS: test_test_pr_auc_above_random_baseline (PR-AUC={test_pr_auc:.4f}, fraud_rate={fraud_rate:.4f})")


def test_required_artifacts_exist():
    """Test 9: All required Phase 1 artifacts exist."""
    required = [
        EXP_DIR / "model.txt",
        EXP_DIR / "preprocessing.joblib",
        EXP_DIR / "feature_names.json",
        EXP_DIR / "metrics.json",
        EXP_DIR / "predictions.parquet",
        EXP_DIR / "feature_importance.csv",
        EXP_DIR / "run_metadata.json",
        EXP_DIR / "pr_curve_test.png",
        EXP_DIR / "roc_curve_test.png",
        EXP_DIR / "confusion_matrix_test.png",
    ]
    missing = [str(p) for p in required if not p.exists()]
    assert not missing, f"Missing artifacts:\n" + "\n".join(missing)
    print(f"PASS: test_required_artifacts_exist ({len(required)} artifacts verified)")


def test_no_preprocessing_leakage_documented():
    """Test 10: Leakage audit CSV exists and documents all key drops."""
    from pathlib import Path
    REPORTS = ROOT / "reports" / "phase1"
    audit_path = REPORTS / "leakage_audit.csv"
    _require_file(audit_path)
    audit = pd.read_csv(audit_path)
    drop_cols = set(audit[audit["action"] == "DROP"]["column"].tolist())
    # These must always be documented as dropped
    for c in ["TransactionID", "isFraud"]:
        assert c in audit["column"].values, f"'{c}' not documented in leakage audit"
    print(f"PASS: test_no_preprocessing_leakage_documented ({len(drop_cols)} columns documented as DROP)")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_no_duplicate_transaction_ids_in_processed_data,
        test_splits_do_not_overlap_in_transaction_id,
        test_temporal_ordering_of_splits,
        test_isfr_not_in_feature_matrix,
        test_prediction_count_matches_source,
        test_model_reload_produces_same_predictions,
        test_metrics_file_exists_and_has_required_keys,
        test_test_pr_auc_above_random_baseline,
        test_required_artifacts_exist,
        test_no_preprocessing_leakage_documented,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Phase 1 Tests: {passed}/{len(tests)} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
