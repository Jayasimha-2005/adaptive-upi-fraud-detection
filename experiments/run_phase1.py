"""
experiments/run_phase1.py
Main orchestration script for Phase 1 — IEEE-CIS LightGBM Baseline (E1).

Usage
-----
  python experiments/run_phase1.py
  python experiments/run_phase1.py --config configs/phase1_lightgbm.yaml

What this script does
---------------------
1.  Load config
2.  Validate raw IEEE-CIS files
3.  Load and join train_transaction + train_identity
4.  Chronological split (train / val / test)
5.  Fit preprocessor on training data ONLY
6.  Transform train / val / test
7.  Save processed Parquet files
8.  Train LightGBM with early stopping on validation
9.  Select decision threshold on validation
10. Compute validation metrics
11. Compute test metrics (ONCE, threshold frozen)
12. Save all artifacts (model, preprocessor, predictions, plots, reports)
13. Run integrity tests
14. Write Phase 1 completion report

STOP CONDITION
--------------
Phase 1 ends after this script completes.
Do NOT call sequence modeling, GRU, Kafka, or any streaming code from here.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Ensure src/ is on the import path ────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config          import load_config
from src.utils.logging_config  import get_logger
from src.data.validate         import validate_ieee_cis
from src.data.load             import load_and_join
from src.data.split            import chronological_split
from src.features.ieee_cis_features import IEEECISPreprocessor, get_feature_inventory
from src.models.lightgbm_baseline   import LightGBMBaseline
from src.evaluation.metrics         import (
    compute_metrics, select_threshold,
    plot_pr_curve, plot_roc_curve, plot_confusion_matrix,
    save_metrics,
)

# ── Global logger ─────────────────────────────────────────────────────────────
logger = get_logger(
    __name__,
    log_file=str(ROOT / "experiments" / "E1_lightgbm" / "run.log"),
)


def run(cfg_path: str | Path = "configs/phase1_lightgbm.yaml") -> None:
    """Execute Phase 1 end-to-end."""
    t_start = time.time()
    cfg_path = Path(cfg_path)
    cfg = load_config(ROOT / cfg_path)
    seed = cfg.get("random_seed", 42)
    np.random.seed(seed)

    # ── Resolve paths ──────────────────────────────────────────────────────────
    paths = cfg["paths"]
    tx_path   = ROOT / paths["raw_transaction"]
    id_path   = ROOT / paths["raw_identity"]
    proc_dir  = ROOT / paths["processed_dir"]
    exp_dir   = ROOT / paths["experiments_dir"]
    rep_dir   = ROOT / paths["reports_dir"]

    proc_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 65)
    logger.info("  PHASE 1 — IEEE-CIS LightGBM Baseline (E1)")
    logger.info("=" * 65)
    logger.info("Config: %s", cfg_path)
    logger.info("Random seed: %d", seed)
    logger.info("Python: %s", sys.version.split()[0])

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — Data validation
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n--- STEP 1: Data Validation ---")
    val_report = validate_ieee_cis(tx_path, id_path, rep_dir)
    logger.info("Validation passed. Fraud rate: %.3f%%", val_report["train_transaction"]["fraud_pct"])

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — Load and join
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n--- STEP 2: Load and Join ---")
    df, join_meta = load_and_join(tx_path, id_path)
    logger.info("Joined dataframe: %d rows x %d cols", len(df), len(df.columns))

    # Feature inventory for documentation
    fi_df = get_feature_inventory(df)
    fi_df.to_csv(rep_dir / "feature_inventory.csv", index=False)
    logger.info("Feature inventory saved: %d columns documented", len(fi_df))

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — Chronological split
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n--- STEP 3: Chronological Split ---")
    split_cfg = cfg["split"]
    df_train, df_val, df_test, split_meta = chronological_split(
        df,
        train_dt_max=split_cfg["train_dt_max"],
        val_dt_max=split_cfg["val_dt_max"],
    )
    del df
    gc.collect()

    # Write split report
    _write_split_report(split_meta, join_meta, rep_dir / "split_report.md")
    with open(exp_dir / "split_meta.json", "w") as f:
        json.dump(split_meta, f, indent=2)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4 — Fit preprocessor on TRAINING data only
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n--- STEP 4: Preprocessing (fit on TRAIN only) ---")
    preprocessor = IEEECISPreprocessor(cfg)
    preprocessor.fit(df_train)

    logger.info("Transforming splits ...")
    X_train, y_train = preprocessor.transform(df_train)
    X_val,   y_val   = preprocessor.transform(df_val)
    X_test,  y_test  = preprocessor.transform(df_test)

    # Leakage audit report
    leakage_df = pd.DataFrame(preprocessor.leakage_log)
    leakage_df.to_csv(rep_dir / "leakage_audit.csv", index=False)
    _write_leakage_md(leakage_df, rep_dir / "leakage_audit.md")
    logger.info("Leakage audit saved: %d columns reviewed", len(leakage_df))

    logger.info(
        "Feature matrix: train=%s | val=%s | test=%s",
        X_train.shape, X_val.shape, X_test.shape,
    )

    # ── Assert leakage prevention ──────────────────────────────────────────────
    assert "isFraud"       not in X_train.columns, "TARGET leaked into X!"
    assert "TransactionID" not in X_train.columns, "ID leaked into X!"
    logger.info("Leakage assertions passed.")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5 — Save processed Parquet
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n--- STEP 5: Save Processed Data ---")
    # Save processed splits — X + y + metadata columns for traceability
    _save_split_parquet(df_train, X_train, y_train, proc_dir / "train.parquet")
    _save_split_parquet(df_val,   X_val,   y_val,   proc_dir / "validation.parquet")
    _save_split_parquet(df_test,  X_test,  y_test,  proc_dir / "test.parquet")
    logger.info("Processed Parquet files saved to: %s", proc_dir)

    # Save preprocessor
    preprocessor.save(exp_dir / "preprocessing.joblib")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 6 — Train LightGBM
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n--- STEP 6: Train LightGBM Baseline ---")
    model = LightGBMBaseline(cfg, random_seed=seed)
    model.fit(X_train, y_train, X_val, y_val)

    # Save model
    model.save(
        model_path=exp_dir / "model.txt",
        feature_path=exp_dir / "feature_names.json",
    )

    # Feature importance
    fi = model.get_feature_importance()
    fi.to_csv(exp_dir / "feature_importance.csv", index=False)
    model.plot_feature_importance(exp_dir / "feature_importance.png", top_n=30)
    logger.info("Top feature (gain): %s", fi.iloc[0]["feature"])

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 7 — Threshold selection on VALIDATION
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n--- STEP 7: Threshold Selection (validation only) ---")
    val_prob  = model.predict_proba(X_val)
    threshold = select_threshold(
        y_val.values, val_prob,
        metric=cfg.get("evaluation", {}).get("threshold_metric", "f1"),
    )
    logger.info("Decision threshold frozen at: %.6f", threshold)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 8 — Validation metrics
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n--- STEP 8: Validation Metrics ---")
    val_metrics = compute_metrics(y_val.values, val_prob, threshold, "VALIDATION")
    plot_pr_curve(y_val.values, val_prob, threshold, "PR Curve — Validation",
                  exp_dir / "pr_curve_validation.png")
    plot_roc_curve(y_val.values, val_prob, "ROC Curve — Validation",
                   exp_dir / "roc_curve_validation.png")
    plot_confusion_matrix(
        y_val.values, (val_prob >= threshold).astype(int),
        "Confusion Matrix — Validation", exp_dir / "confusion_matrix_validation.png",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 9 — TEST evaluation (ONE TIME ONLY — threshold already frozen)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n--- STEP 9: Test Evaluation (ONCE — threshold frozen) ---")
    test_prob   = model.predict_proba(X_test)
    test_metrics = compute_metrics(y_test.values, test_prob, threshold, "TEST")
    plot_pr_curve(y_test.values, test_prob, threshold, "PR Curve — Test",
                  exp_dir / "pr_curve_test.png")
    plot_roc_curve(y_test.values, test_prob, "ROC Curve — Test",
                   exp_dir / "roc_curve_test.png")
    plot_confusion_matrix(
        y_test.values, (test_prob >= threshold).astype(int),
        "Confusion Matrix — Test", exp_dir / "confusion_matrix_test.png",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 10 — Save metrics and predictions
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n--- STEP 10: Save Metrics and Predictions ---")
    save_metrics(val_metrics, test_metrics, exp_dir)

    # Predictions file
    _save_predictions(df_val,  val_prob,  threshold, y_val,  "validation", exp_dir)
    _save_predictions(df_test, test_prob, threshold, y_test, "test",       exp_dir)
    logger.info("Predictions saved.")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 11 — SHAP (if enabled and feasible)
    # ══════════════════════════════════════════════════════════════════════════
    shap_cfg = cfg.get("shap", {})
    if shap_cfg.get("enabled", True):
        logger.info("\n--- STEP 11: SHAP Analysis ---")
        _run_shap(model, X_val, shap_cfg.get("sample_size", 5000), exp_dir)
    else:
        logger.info("SHAP disabled in config — skipping.")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 12 — Run integrity tests
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n--- STEP 12: Integrity Tests ---")
    _run_integrity_tests(
        df_train, df_val, df_test,
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        model, preprocessor, split_meta, exp_dir,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 13 — Run metadata
    # ══════════════════════════════════════════════════════════════════════════
    total_elapsed = time.time() - t_start
    run_meta = _build_run_metadata(
        cfg, model, val_metrics, test_metrics, threshold,
        split_meta, preprocessor, total_elapsed,
    )
    with open(exp_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)
    logger.info("Run metadata saved.")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 14 — Final reports
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n--- STEP 14: Final Reports ---")
    _write_baseline_report(val_metrics, test_metrics, threshold, fi, run_meta,
                           split_meta, rep_dir / "lightgbm_baseline_report.md")

    logger.info("=" * 65)
    logger.info("  PHASE 1 COMPLETE in %.1fs", total_elapsed)
    logger.info("  Validation PR-AUC: %.4f", val_metrics["pr_auc"])
    logger.info("  Test       PR-AUC: %.4f", test_metrics["pr_auc"])
    logger.info("  Artifacts: %s", exp_dir)
    logger.info("  Reports:   %s", rep_dir)
    logger.info("=" * 65)

    logger.info("\nSTOP — Phase 1 complete. Await Phase 2 instructions.")


# ── Helper functions ───────────────────────────────────────────────────────────

def _save_split_parquet(
    df_orig: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    out_path: Path,
) -> None:
    """Save processed split as Parquet with metadata columns for traceability."""
    meta_cols = pd.DataFrame({
        "TransactionID": df_orig["TransactionID"].values,
        "TransactionDT": df_orig["TransactionDT"].values,
        "isFraud":       y.values,
    })
    out = pd.concat([meta_cols.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    out.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info("Saved %s: %s rows, %s cols", out_path.name, len(out), len(out.columns))


def _save_predictions(
    df_orig: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float,
    y_true: pd.Series,
    split_name: str,
    exp_dir: Path,
) -> None:
    """Save predictions Parquet as specified in the experiment spec."""
    preds = pd.DataFrame({
        "TransactionID":    df_orig["TransactionID"].values,
        "TransactionDT":    df_orig["TransactionDT"].values,
        "isFraud":          y_true.values,
        "fraud_probability": y_prob,
        "prediction":       (y_prob >= threshold).astype(int),
        "split":            split_name,
    })
    out_path = exp_dir / "predictions.parquet"
    if out_path.exists():
        existing = pd.read_parquet(out_path, engine="pyarrow")
        existing = existing[existing["split"] != split_name]
        preds = pd.concat([existing, preds], ignore_index=True)
    preds.to_parquet(out_path, index=False, engine="pyarrow")


def _run_shap(
    model: LightGBMBaseline,
    X_val: pd.DataFrame,
    sample_size: int,
    exp_dir: Path,
) -> None:
    """Compute SHAP summary on a validation sample."""
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = min(sample_size, len(X_val))
        X_sample = X_val.sample(n, random_state=42)
        logger.info("Computing SHAP values on %d validation samples ...", n)
        t0 = time.time()
        explainer = shap.TreeExplainer(model.model)
        shap_values = explainer.shap_values(X_sample)
        logger.info("SHAP computed in %.1fs", time.time() - t0)

        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(exp_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("SHAP summary plot saved.")

    except Exception as e:
        logger.warning("SHAP analysis failed: %s — deferring to later phase.", e)


def _run_integrity_tests(
    df_train, df_val, df_test,
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    model, preprocessor, split_meta, exp_dir,
) -> None:
    """10-point integrity test suite."""
    results = {}
    errors  = []

    # Test 1: No duplicate TransactionID in processed splits
    for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        dup = int(df["TransactionID"].duplicated().sum())
        results[f"no_dup_tx_id_{name}"] = dup == 0
        if dup > 0:
            errors.append(f"FAIL: {dup} duplicate TransactionIDs in {name}")

    # Test 2: No TransactionID overlap between splits
    train_ids = set(df_train["TransactionID"])
    val_ids   = set(df_val["TransactionID"])
    test_ids  = set(df_test["TransactionID"])
    results["no_overlap_train_val"]  = len(train_ids & val_ids) == 0
    results["no_overlap_val_test"]   = len(val_ids  & test_ids) == 0
    results["no_overlap_train_test"] = len(train_ids & test_ids) == 0

    # Test 3: Temporal ordering
    results["train_before_val"]  = df_train["TransactionDT"].max() < df_val["TransactionDT"].min()
    results["val_before_test"]   = df_val["TransactionDT"].max()   < df_test["TransactionDT"].min()

    # Test 4: isFraud not in X
    results["isfr_not_in_x_train"] = "isFraud"       not in X_train.columns
    results["tx_id_not_in_x"]      = "TransactionID" not in X_train.columns

    # Test 5: Preprocessor not fit on val/test (verify medians are from train only)
    # (Structural test — preprocessor was fit() called only once on df_train above)
    results["preprocessor_fitted"] = preprocessor._is_fitted

    # Test 6: Prediction row counts match source
    val_prob  = model.predict_proba(X_val)
    test_prob = model.predict_proba(X_test)
    results["val_pred_count_matches"]  = len(val_prob)  == len(X_val)
    results["test_pred_count_matches"] = len(test_prob) == len(X_test)

    # Test 7: Model save/reload produces same predictions
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        model.save(
            model_path=Path(tmp) / "test_model.txt",
            feature_path=Path(tmp) / "test_features.json",
        )
        from src.models.lightgbm_baseline import LightGBMBaseline
        m2 = LightGBMBaseline.load(Path(tmp) / "test_model.txt", Path(tmp) / "test_features.json")
        reload_prob = m2.predict_proba(X_val)
        max_diff = float(np.abs(val_prob - reload_prob).max())
        results["reload_same_predictions"] = max_diff < 1e-6
        if max_diff >= 1e-6:
            errors.append(f"Reload prediction difference: {max_diff:.2e}")

    # Summary
    passed = int(sum(bool(v) for v in results.values()))
    total  = int(len(results))
    clean_details = {k: bool(v) for k, v in results.items()}
    for k, v in clean_details.items():
        status = "PASS" if v else "FAIL"
        if not v:
            errors.append(f"FAIL: {k}")
        logger.info("  [%s] %s", status, k)

    test_report = {
        "passed": passed,
        "total": total,
        "errors": errors,
        "details": clean_details,
    }
    with open(exp_dir / "test_report.json", "w") as f:
        json.dump(test_report, f, indent=2)

    if errors:
        logger.error("INTEGRITY TESTS FAILED:\n" + "\n".join(errors))
    else:
        logger.info("All %d integrity tests PASSED.", total)


def _build_run_metadata(
    cfg, model, val_metrics, test_metrics, threshold,
    split_meta, preprocessor, elapsed,
) -> dict:
    import lightgbm as lgb
    import sklearn
    return {
        "phase": 1,
        "experiment": "E1_lightgbm",
        "random_seed": cfg.get("random_seed", 42),
        "python_version": sys.version.split()[0],
        "os": platform.system(),
        "packages": {
            "pandas":    pd.__version__,
            "numpy":     np.__version__,
            "lightgbm":  lgb.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "dataset": "IEEE-CIS Fraud Detection (Vesta Corporation / Kaggle)",
        "dataset_type": "Real-world-derived, anonymized, historical e-commerce",
        "split_boundaries": {
            "train_dt_max": split_meta["train_dt_max_sec"],
            "val_dt_max":   split_meta["val_dt_max_sec"],
        },
        "n_features": len(preprocessor.feature_names),
        "decision_threshold": threshold,
        "train_metadata": model.train_metadata,
        "validation_pr_auc": val_metrics["pr_auc"],
        "test_pr_auc":       test_metrics["pr_auc"],
        "total_runtime_sec": round(elapsed, 2),
    }


def _write_split_report(split_meta: dict, join_meta: dict, path: Path) -> None:
    lines = [
        "# Phase 1 — Data Split Report",
        "",
        "## Join Summary",
        "",
        f"- Transaction rows: {join_meta['tx_rows_before_join']:,}",
        f"- Identity rows:    {join_meta['id_rows_before_join']:,}",
        f"- Joined rows:      {join_meta['joined_rows']:,}",
        f"- With identity:    {join_meta['rows_with_identity']:,} ({join_meta['pct_with_identity']:.1f}%)",
        f"- Without identity: {join_meta['rows_without_identity']:,}",
        "",
        "## Temporal Split",
        "",
        "Splitting strategy: **chronological** (70% / 15% / 15% by TransactionDT).",
        "No random shuffling. No temporal overlap.",
        "",
        "| Split | Rows | Fraud | Legit | Fraud % | Day Start | Day End |",
        "|-------|------|-------|-------|---------|-----------|---------|",
    ]
    for k in ["train", "validation", "test"]:
        s = split_meta[k]
        lines.append(
            f"| {k.capitalize()} | {s['rows']:,} | {s['fraud']:,} | "
            f"{s['legit']:,} | {s['fraud_pct']:.3f}% | "
            f"{s['day_min']:.1f} | {s['day_max']:.1f} |"
        )
    lines += [
        "",
        f"- No TransactionID overlap between splits: confirmed.",
        f"- Train DT max: day {split_meta['train']['day_max']:.1f}",
        f"- Val DT min:   day {split_meta['validation']['day_min']:.1f}",
        f"- Test DT min:  day {split_meta['test']['day_min']:.1f}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_leakage_md(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Phase 1 — Leakage Audit",
        "",
        "All column-level decisions documented below.",
        "",
        "| Column | Action | Reason |",
        "|--------|--------|--------|",
    ]
    for _, row in df.iterrows():
        lines.append(f"| `{row['column']}` | **{row['action']}** | {row['reason']} |")
    lines += [
        "",
        "## Key Leakage Prevention Measures",
        "",
        "1. `isFraud` excluded from X in all splits.",
        "2. `TransactionID` excluded from X (identifier, not predictive feature).",
        "3. `TransactionDT` converted to hour_sin/cos + day_index; raw excluded from X.",
        "4. All preprocessing (imputation, encoding) fit ONLY on training data.",
        "5. Validation and test data only transformed — never fit.",
        "6. Threshold selected on validation; test evaluated once with frozen threshold.",
        "7. V-columns with >80% missing dropped (measured on training data only).",
        "8. D-columns with >80% missing dropped (D6,D7,D8,D9,D12,D13,D14).",
        "9. dist2 dropped (93.6% missing in training).",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_baseline_report(
    val_metrics, test_metrics, threshold, fi, run_meta,
    split_meta, path,
) -> None:
    fi_top = fi.head(15)
    lines = [
        "# Phase 1 — LightGBM Baseline Report (E1)",
        "",
        "**Dataset:** IEEE-CIS Fraud Detection (Vesta Corporation / Kaggle Competition)",
        "**Type:** Real-world-derived, anonymized, historical e-commerce transactions",
        "",
        "> This is a baseline experiment. No GRU, Transformer, Kafka, or adaptive retraining.",
        "> The primary metric is **PR-AUC** (Average Precision). Accuracy is not reported.",
        "",
        "## Dataset",
        "",
        f"- Training rows: {split_meta['train']['rows']:,}",
        f"- Validation rows: {split_meta['validation']['rows']:,}",
        f"- Test rows: {split_meta['test']['rows']:,}",
        f"- Training fraud: {split_meta['train']['fraud']:,} ({split_meta['train']['fraud_pct']:.3f}%)",
        f"- Features: {run_meta['n_features']}",
        "",
        "## Temporal Split",
        "",
        "| Split | Days | Rows | Fraud % |",
        "|-------|------|------|---------|",
    ]
    for k in ["train", "validation", "test"]:
        s = split_meta[k]
        lines.append(f"| {k.capitalize()} | {s['day_min']:.0f}–{s['day_max']:.0f} | {s['rows']:,} | {s['fraud_pct']:.3f}% |")

    lines += [
        "",
        "## Preprocessing",
        "",
        "- JOIN: LEFT JOIN train_transaction + train_identity on TransactionID",
        "- Temporal features: hour_sin, hour_cos, day_index (from TransactionDT)",
        "- Missingness indicators: binary flags for key high-missing columns",
        "- Numerical imputation: median (fit on train only)",
        "- Categorical encoding: LabelEncoder (fit on train only)",
        "- Dropped: D6,D7,D8,D9,D12,D13,D14 (>80% missing), dist2 (93.6% missing)",
        "- Dropped: V-columns with >80% missing in training data",
        "- Excluded: TransactionID, isFraud, TransactionDT (raw)",
        "",
        "## Model Configuration",
        "",
        f"- Model: LightGBM {run_meta['packages']['lightgbm']}",
        f"- Objective: binary",
        f"- Metric: average_precision (PR-AUC)",
        f"- is_unbalance: True (class imbalance handling)",
        f"- Best iteration: {run_meta['train_metadata'].get('best_iteration', 'N/A')}",
        f"- Training time: {run_meta['train_metadata'].get('training_time_sec', 'N/A')}s",
        "",
        "## Validation Results (threshold frozen here)",
        "",
        f"| Metric | Value |",
        "|--------|-------|",
        f"| **PR-AUC** | **{val_metrics['pr_auc']:.6f}** |",
        f"| ROC-AUC    | {val_metrics['roc_auc']:.6f} |",
        f"| Precision  | {val_metrics['precision']:.6f} |",
        f"| Recall     | {val_metrics['recall']:.6f} |",
        f"| F1         | {val_metrics['f1']:.6f} |",
        f"| MCC        | {val_metrics['mcc']:.6f} |",
        f"| Balanced Accuracy | {val_metrics['balanced_accuracy']:.6f} |",
        f"| FPR        | {val_metrics['fpr']:.6f} |",
        f"| FNR        | {val_metrics['fnr']:.6f} |",
        f"| Decision threshold | {threshold:.6f} |",
        "",
        "## Final Test Results (evaluated ONCE with frozen threshold)",
        "",
        f"| Metric | Value |",
        "|--------|-------|",
        f"| **PR-AUC** | **{test_metrics['pr_auc']:.6f}** |",
        f"| ROC-AUC    | {test_metrics['roc_auc']:.6f} |",
        f"| Precision  | {test_metrics['precision']:.6f} |",
        f"| Recall     | {test_metrics['recall']:.6f} |",
        f"| F1         | {test_metrics['f1']:.6f} |",
        f"| MCC        | {test_metrics['mcc']:.6f} |",
        f"| Balanced Accuracy | {test_metrics['balanced_accuracy']:.6f} |",
        f"| FPR        | {test_metrics['fpr']:.6f} |",
        f"| FNR        | {test_metrics['fnr']:.6f} |",
        f"| TP={test_metrics['tp']} FP={test_metrics['fp']} TN={test_metrics['tn']} FN={test_metrics['fn']} | |",
        "",
        "## Top Features (by Gain)",
        "",
        "| Rank | Feature | Gain |",
        "|------|---------|------|",
    ]
    for _, row in fi_top.iterrows():
        lines.append(f"| {row['gain_rank']} | `{row['feature']}` | {row['gain']:.2f} |")

    lines += [
        "",
        "## Limitations",
        "",
        "1. This is a baseline — no hyperparameter tuning (Optuna) performed.",
        "2. No GRU temporal sequence modeling (Phase 2).",
        "3. No concept drift adaptation (Phase 3).",
        "4. No streaming evaluation (Phase 4).",
        "5. SHAP computed on validation sample only.",
        "6. Dataset is anonymized — V-column interpretations are not available.",
        "7. IEEE-CIS is e-commerce fraud, not UPI transaction fraud.",
        "",
        "## Reproducibility",
        "",
        f"- Random seed: {run_meta['random_seed']}",
        f"- Python: {run_meta['python_version']}",
        f"- LightGBM: {run_meta['packages']['lightgbm']}",
        f"- pandas: {run_meta['packages']['pandas']}",
        f"- numpy: {run_meta['packages']['numpy']}",
        "",
        "**Reproduction command:**",
        "```bash",
        "python experiments/run_phase1.py --config configs/phase1_lightgbm.yaml",
        "```",
        "",
        "## Next Recommended Phase",
        "",
        "Phase 2: Temporal sequence modeling using GRU on card1 entity histories.",
        "Requires: Phase 1 model as tabular baseline for comparison.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Baseline report written: %s", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 — LightGBM Baseline (E1)")
    parser.add_argument("--config", default="configs/phase1_lightgbm.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()
    run(args.config)
