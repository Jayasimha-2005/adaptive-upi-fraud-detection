"""
src/audit/encoding_comparison.py
Phase 1.5D — Categorical encoding strategy comparison.

PURPOSE
  Compare three encoding strategies for the 22 string columns in IEEE-CIS.
  The current E1 baseline uses LabelEncoder, which imposes artificial ordinal
  ordering. This experiment tests whether alternative strategies improve results.

VARIANTS
  V1 : LabelEncoder (current E1) — artificial ordinal order
  V2 : Native LightGBM categorical support (dtype=category, pass to LightGBM)
  V3 : Frequency encoding (replace category with its training-set frequency count)

E1B DECISION CRITERION
  Unlike the original plan (PR-AUC > 0.01), the decision to run E1b uses a
  multi-metric assessment:
    - PR-AUC
    - ROC-AUC
    - Recall @ 1% FPR
    - Precision @ Top-1000
    - Bootstrap CI overlap (if CIs are available)
    - Training cost and inference overhead

  A variant is recommended for E1b only if it shows a meaningful AND consistent
  improvement across multiple metrics. A single-metric gain of 0.01 in isolation
  is NOT sufficient justification.

LEAKAGE POLICY
  All encoders are fit ONLY on training data. No test or validation statistics
  are used for encoding. Frequency counts are computed on the training split.

OUTPUT
  reports/phase1/encoding_comparison.md — multi-metric comparison table
  experiments/E1_encoding_comparison/results.json — raw results
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from src.evaluation.metrics import (
    compute_metrics,
    precision_at_top_k,
    recall_at_fixed_fpr,
    select_threshold,
)
from src.features.ieee_cis_features import CATEGORICAL_COLS
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# ── Encoding strategies ───────────────────────────────────────────────────────

def apply_label_encoding(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """V1: LabelEncoder — fit on train only."""
    from sklearn.preprocessing import LabelEncoder

    train = train_df.copy()
    val   = val_df.copy()
    test  = test_df.copy()

    for col in cat_cols:
        if col not in train.columns:
            continue
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str).fillna("__MISSING__"))
        # Map unseen categories to a safe value (-1 or len(classes))
        known = {v: i for i, v in enumerate(le.classes_)}
        fallback = len(le.classes_)
        val[col]  = val[col].astype(str).fillna("__MISSING__").map(known).fillna(fallback).astype(int)
        test[col] = test[col].astype(str).fillna("__MISSING__").map(known).fillna(fallback).astype(int)

    return train, val, test


def apply_native_categorical(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    V2: Native LightGBM categorical support.
    Set dtype to pd.Categorical, using categories fit on training data only.
    LightGBM will handle the encoding internally.
    """
    train = train_df.copy()
    val   = val_df.copy()
    test  = test_df.copy()

    for col in cat_cols:
        if col not in train.columns:
            continue
        # Fill missing
        fill_val = "__MISSING__"
        train[col] = train[col].astype(str).fillna(fill_val)
        val[col]   = val[col].astype(str).fillna(fill_val)
        test[col]  = test[col].astype(str).fillna(fill_val)

        # Fit categories on training data only
        categories = list(train[col].astype("category").cat.categories)
        train[col] = pd.Categorical(train[col], categories=categories)
        val[col]   = pd.Categorical(val[col],   categories=categories)
        test[col]  = pd.Categorical(test[col],  categories=categories)

    return train, val, test


def apply_frequency_encoding(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    V3: Frequency encoding.
    Replace each category with its training-set occurrence count.
    Unseen categories get count = 0 (or 1 as smoothing).
    No target information used — zero leakage risk.
    """
    train = train_df.copy()
    val   = val_df.copy()
    test  = test_df.copy()

    for col in cat_cols:
        if col not in train.columns:
            continue
        freq_map = (
            train[col].astype(str).fillna("__MISSING__")
            .value_counts()
            .to_dict()
        )
        fill_value = 0  # unseen in training → count 0
        train[col] = train[col].astype(str).fillna("__MISSING__").map(freq_map).fillna(fill_value).astype(int)
        val[col]   = val[col].astype(str).fillna("__MISSING__").map(freq_map).fillna(fill_value).astype(int)
        test[col]  = test[col].astype(str).fillna("__MISSING__").map(freq_map).fillna(fill_value).astype(int)

    return train, val, test


# ── Model training ────────────────────────────────────────────────────────────

def _train_and_evaluate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str] | None,
    lgbm_params: dict,
    variant_name: str,
) -> dict:
    """Train LightGBM with given encoding and return full metric suite."""
    import lightgbm as lgb

    target = "isFraud"
    X_train = train_df[feature_cols]
    y_train = train_df[target].to_numpy(dtype=int)
    X_val   = val_df[feature_cols]
    y_val   = val_df[target].to_numpy(dtype=int)
    X_test  = test_df[feature_cols]
    y_test  = test_df[target].to_numpy(dtype=int)

    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols or "auto")
    dval   = lgb.Dataset(X_val,   label=y_val,   reference=dtrain,
                         categorical_feature=cat_cols or "auto")

    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False),
                 lgb.log_evaluation(period=100)]

    t0 = time.time()
    booster = lgb.train(
        params=lgbm_params,
        train_set=dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=callbacks,
    )
    train_time = time.time() - t0

    y_prob_val  = booster.predict(X_val)
    y_prob_test = booster.predict(X_test)

    threshold = select_threshold(y_val, y_prob_val, metric="f1")

    val_metrics  = compute_metrics(y_val,  y_prob_val,  threshold, split_name=f"{variant_name}_val")
    test_metrics = compute_metrics(y_test, y_prob_test, threshold, split_name=f"{variant_name}_test")

    # Operational metrics
    op_val  = {**recall_at_fixed_fpr(y_val,  y_prob_val,  [0.01]),
               **precision_at_top_k(y_val,  y_prob_val,  [1000])}
    op_test = {**recall_at_fixed_fpr(y_test, y_prob_test, [0.01]),
               **precision_at_top_k(y_test, y_prob_test, [1000])}

    result = {
        "variant":       variant_name,
        "threshold":     threshold,
        "train_time_s":  round(train_time, 2),
        "n_trees":       booster.num_trees(),
        "val":           val_metrics,
        "test":          test_metrics,
        "val_operational":  op_val,
        "test_operational": op_test,
    }

    logger.info(
        "[%s] PR-AUC val=%.4f test=%.4f | Recall@1%%FPR val=%.4f test=%.4f "
        "| P@Top-1K val=%.4f test=%.4f | Trees=%d | Time=%.1fs",
        variant_name,
        val_metrics["pr_auc"], test_metrics["pr_auc"],
        op_val.get("recall_at_fpr_0.010", 0), op_test.get("recall_at_fpr_0.010", 0),
        op_val.get("precision_at_top_1000", 0), op_test.get("precision_at_top_1000", 0),
        booster.num_trees(), train_time,
    )
    return result


# ── Main comparison runner ────────────────────────────────────────────────────

def run_encoding_comparison(
    processed_dir: Path,
    output_dir: Path,
    report_dir: Path,
    lgbm_params: dict | None = None,
) -> dict:
    """
    Execute V1/V2/V3 encoding comparison.

    Parameters
    ----------
    processed_dir : Path to processed parquet files (train/validation/test.parquet).
    output_dir    : Path for raw results JSON.
    report_dir    : Path for markdown report.

    Returns
    -------
    dict with full results for all three variants.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # ── Load processed parquets ───────────────────────────────────────────────
    logger.info("Loading processed parquet files from %s", processed_dir)
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df   = pd.read_parquet(processed_dir / "validation.parquet")
    test_df  = pd.read_parquet(processed_dir / "test.parquet")

    # Identify feature columns (exclude target, ID, raw DT)
    exclude = {"isFraud", "TransactionID", "TransactionDT"}
    all_cols = [c for c in train_df.columns if c not in exclude]

    # Determine which categorical columns are present
    cat_cols_present = [c for c in CATEGORICAL_COLS if c in all_cols]
    non_cat_cols = [c for c in all_cols if c not in cat_cols_present]

    logger.info(
        "Feature matrix: %d total | %d categorical | %d numeric/binary",
        len(all_cols), len(cat_cols_present), len(non_cat_cols),
    )

    # Default LightGBM params (mirrors E1 config)
    if lgbm_params is None:
        lgbm_params = {
            "objective": "binary",
            "metric": "average_precision",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": -1,
            "min_child_samples": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "is_unbalance": True,
            "seed": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

    all_results = {}

    # ── V1: LabelEncoder (current E1 baseline) ────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("V1: LabelEncoder (current E1 baseline)")
    logger.info("=" * 60)
    tr1, va1, te1 = apply_label_encoding(train_df.copy(), val_df.copy(), test_df.copy(), cat_cols_present)
    all_results["V1_label_encoder"] = _train_and_evaluate(
        tr1, va1, te1, feature_cols=all_cols, cat_cols=None,
        lgbm_params=lgbm_params, variant_name="V1_label_encoder",
    )

    # ── V2: Native LightGBM categorical ──────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("V2: Native LightGBM categorical")
    logger.info("=" * 60)
    tr2, va2, te2 = apply_native_categorical(train_df.copy(), val_df.copy(), test_df.copy(), cat_cols_present)
    all_results["V2_native_categorical"] = _train_and_evaluate(
        tr2, va2, te2, feature_cols=all_cols, cat_cols=cat_cols_present,
        lgbm_params=lgbm_params, variant_name="V2_native_categorical",
    )

    # ── V3: Frequency encoding ────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("V3: Frequency encoding")
    logger.info("=" * 60)
    tr3, va3, te3 = apply_frequency_encoding(train_df.copy(), val_df.copy(), test_df.copy(), cat_cols_present)
    all_results["V3_frequency_encoding"] = _train_and_evaluate(
        tr3, va3, te3, feature_cols=all_cols, cat_cols=None,
        lgbm_params=lgbm_params, variant_name="V3_frequency_encoding",
    )

    # ── Save raw results ──────────────────────────────────────────────────────
    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Raw results saved: %s", results_path)

    # ── Write markdown report ─────────────────────────────────────────────────
    _write_encoding_report(all_results, report_dir)

    return all_results


def _write_encoding_report(results: dict, report_dir: Path) -> None:
    """Write reports/phase1/encoding_comparison.md."""
    out_path = report_dir / "encoding_comparison.md"

    def get(variant, *keys, default="N/A"):
        d = results.get(variant, {})
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k, default)
            else:
                return default
        return d if d != {} else default

    variants = ["V1_label_encoder", "V2_native_categorical", "V3_frequency_encoding"]
    names = {"V1_label_encoder": "V1 LabelEncoder (E1 baseline)",
             "V2_native_categorical": "V2 Native Categorical",
             "V3_frequency_encoding": "V3 Frequency Encoding"}

    lines = [
        "# Phase 1.5D — Categorical Encoding Comparison",
        "",
        "**Experiment:** E1 vs. encoding variants | **Dataset:** IEEE-CIS (test set)",
        "**Status:** E1 (V1) is IMMUTABLE — comparison only, not overwrite",
        "",
        "## E1b Decision Criterion",
        "",
        "A variant is recommended for E1b only if it shows **consistent improvement across**:",
        "- PR-AUC (test)",
        "- Recall @ 1% FPR (test)",
        "- Precision @ Top-1,000 (test)",
        "- No significant increase in training cost",
        "",
        "A single-metric marginal gain does NOT justify E1b.",
        "",
        "---",
        "",
        "## Multi-Metric Comparison Table",
        "",
        "| Metric | V1 LabelEncoder (E1) | V2 Native Categorical | V3 Frequency Encoding |",
        "|--------|---------------------|----------------------|----------------------|",
    ]

    metrics = [
        ("PR-AUC (test)", lambda v: f"{get(v, 'test', 'pr_auc'):.4f}"),
        ("ROC-AUC (test)", lambda v: f"{get(v, 'test', 'roc_auc'):.4f}"),
        ("Recall (test, frozen threshold)", lambda v: f"{get(v, 'test', 'recall'):.4f}"),
        ("Precision (test)", lambda v: f"{get(v, 'test', 'precision'):.4f}"),
        ("F1 (test)", lambda v: f"{get(v, 'test', 'f1'):.4f}"),
        ("Recall @ 1% FPR (test)", lambda v: f"{get(v, 'test_operational', 'recall_at_fpr_0.010'):.4f}"),
        ("Precision @ Top-1,000 (test)", lambda v: f"{get(v, 'test_operational', 'precision_at_top_1000'):.4f}"),
        ("PR-AUC (validation)", lambda v: f"{get(v, 'val', 'pr_auc'):.4f}"),
        ("Training time (seconds)", lambda v: f"{get(v, 'train_time_s')}"),
        ("Number of trees", lambda v: str(get(v, 'n_trees'))),
        ("Frozen threshold", lambda v: f"{get(v, 'threshold'):.6f}"),
    ]

    for label, fn in metrics:
        row = f"| {label} | " + " | ".join(fn(v) for v in variants) + " |"
        lines.append(row)

    lines += [
        "",
        "---",
        "",
        "## E1b Recommendation",
        "",
        "*(Fill in after running this script and reviewing the table above.)*",
        "",
        "| Decision | Reasoning |",
        "|----------|-----------|",
        "| E1b justified? | [YES / NO — state metrics] |",
        "| Recommended encoding | [V1 / V2 / V3] |",
        "| Next step | [Keep E1 as final OR create E1b_lightgbm experiment] |",
        "",
        "---",
        "",
        "## Notes on Encoding Strategies",
        "",
        "**V1 — LabelEncoder:** Assigns integers 0, 1, 2, ... to categories in sorted order. "
        "Creates artificial ordinal relationships. However, LightGBM's tree-splitting "
        "can partially compensate by treating each split as binary.",
        "",
        "**V2 — Native Categorical:** LightGBM internally uses the Fisher optimal split "
        "algorithm for categorical features, which finds the optimal subset partition "
        "rather than imposing ordinal order. Generally preferred for high-cardinality categoricals.",
        "",
        "**V3 — Frequency Encoding:** Replaces each category with its training-set count. "
        "Encodes popularity/rarity information. No target information used (unlike target encoding). "
        "Works well when frequency is a meaningful signal (e.g., rare email domains = higher fraud risk).",
        "",
        "*Generated by: src/audit/encoding_comparison.py | Phase 1.5D | E1 IMMUTABLE*",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Encoding comparison report saved: %s", out_path)


if __name__ == "__main__":
    get_logger(__name__)
    processed_dir = _ROOT / "datasets" / "processed" / "ieee_cis"
    output_dir    = _ROOT / "experiments" / "E1_encoding_comparison"
    report_dir    = _ROOT / "reports" / "phase1"

    run_encoding_comparison(processed_dir, output_dir, report_dir)
