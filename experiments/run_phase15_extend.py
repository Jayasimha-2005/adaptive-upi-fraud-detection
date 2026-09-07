"""
experiments/run_phase15_extend.py
Phase 1.5B — Statistical evaluation extension.

WHAT THIS SCRIPT DOES
  Reads the EXISTING E1_lightgbm predictions.parquet and metrics.json.
  It does NOT reload the model, does NOT touch the training data, and
  does NOT retrain anything. E1 is IMMUTABLE.

  Computes and writes:
    1. Bootstrap 95% confidence intervals (targeting 2000 valid samples)
    2. Recall @ fixed FPR operating points (0.1%, 0.5%, 1.0%, 2.0%)
    3. Precision @ Top-K (K = 100, 500, 1000, 5000)
    4. Calibration analysis (Brier score, ECE, reliability diagram)

  All outputs are written to experiments/E1_lightgbm/ alongside
  existing Phase 1 artifacts. New keys are appended to metrics.json
  under "phase15_operational" and "phase15_bootstrap" and "phase15_calibration".

USAGE
  python experiments/run_phase15_extend.py --config configs/phase1_lightgbm.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Project root on sys.path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.evaluation.bootstrap import bootstrap_metrics
from src.evaluation.calibration import (
    compute_calibration_metrics,
    plot_calibration_curve,
)
from src.evaluation.metrics import (
    precision_at_top_k,
    recall_at_fixed_fpr,
)
from src.utils.config import load_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _load_predictions(exp_dir: Path) -> pd.DataFrame:
    """Load the existing predictions.parquet from E1_lightgbm."""
    pred_path = exp_dir / "predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"predictions.parquet not found at {pred_path}.\n"
            "Run experiments/run_phase1.py first."
        )
    df = pd.read_parquet(pred_path)
    required = {"isFraud", "fraud_probability", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"predictions.parquet missing columns: {missing}")
    logger.info(
        "Loaded predictions.parquet: %d rows | columns: %s",
        len(df), list(df.columns),
    )
    return df


def _load_frozen_threshold(exp_dir: Path) -> float:
    """Load the frozen threshold from metrics.json."""
    metrics_path = exp_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found at {metrics_path}")
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    threshold = data["validation"]["threshold"]
    logger.info("Frozen threshold loaded from metrics.json: %.6f", threshold)
    return float(threshold)


def _append_to_metrics_json(exp_dir: Path, new_sections: dict) -> None:
    """Append new result sections to metrics.json without overwriting existing keys."""
    metrics_path = exp_dir / "metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        existing = json.load(f)

    for key, value in new_sections.items():
        if key in existing:
            logger.warning("Overwriting existing key '%s' in metrics.json", key)
        existing[key] = value

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    logger.info("metrics.json updated with keys: %s", list(new_sections.keys()))


def run_phase15b(exp_dir: Path, n_bootstrap: int = 2000, seed: int = 42) -> None:
    """Execute Phase 1.5B statistical evaluation extension."""
    logger.info("=" * 70)
    logger.info("PHASE 1.5B — Statistical Evaluation Extension")
    logger.info("E1 experiment dir: %s", exp_dir)
    logger.info("=" * 70)

    # ── Load existing artifacts ────────────────────────────────────────────────
    preds = _load_predictions(exp_dir)
    threshold = _load_frozen_threshold(exp_dir)

    test_df = preds[preds["split"] == "test"].copy()
    val_df  = preds[preds["split"] == "validation"].copy()

    logger.info(
        "Test split: %d rows (%d fraud, %d legit) | threshold=%.6f",
        len(test_df), int(test_df["isFraud"].sum()),
        int((test_df["isFraud"] == 0).sum()), threshold,
    )

    y_true_test = test_df["isFraud"].to_numpy(dtype=int)
    y_prob_test = test_df["fraud_probability"].to_numpy(dtype=float)

    y_true_val = val_df["isFraud"].to_numpy(dtype=int)
    y_prob_val = val_df["fraud_probability"].to_numpy(dtype=float)

    new_sections: dict = {}

    # ── Step 1: Recall @ Fixed FPR ─────────────────────────────────────────────
    logger.info("\n[STEP 1] Recall @ Fixed FPR (test set)")
    fpr_targets = [0.001, 0.005, 0.010, 0.020]
    recall_fpr_test = recall_at_fixed_fpr(y_true_test, y_prob_test, fpr_targets)
    recall_fpr_val  = recall_at_fixed_fpr(y_true_val,  y_prob_val,  fpr_targets)

    logger.info("Test  | " + " | ".join(
        f"Recall@{int(t*1000)/10:.1f}%FPR={v:.4f}"
        for t, v in zip(fpr_targets, recall_fpr_test.values())
    ))
    logger.info("Val   | " + " | ".join(
        f"Recall@{int(t*1000)/10:.1f}%FPR={v:.4f}"
        for t, v in zip(fpr_targets, recall_fpr_val.values())
    ))

    # ── Step 2: Precision @ Top-K ──────────────────────────────────────────────
    logger.info("\n[STEP 2] Precision @ Top-K (test set)")
    k_values = [100, 500, 1000, 5000]
    prec_topk_test = precision_at_top_k(y_true_test, y_prob_test, k_values)
    prec_topk_val  = precision_at_top_k(y_true_val,  y_prob_val,  k_values)

    logger.info("Test  | " + " | ".join(f"P@{k}={v:.4f}" for k, v in zip(k_values, prec_topk_test.values())))
    logger.info("Val   | " + " | ".join(f"P@{k}={v:.4f}" for k, v in zip(k_values, prec_topk_val.values())))

    new_sections["phase15_operational"] = {
        "test":  {**recall_fpr_test, **prec_topk_test},
        "validation": {**recall_fpr_val, **prec_topk_val},
        "fpr_definition": (
            "For each target FPR alpha, Recall@alpha = "
            "max{ TPR(t) : empirical_FPR(t) <= alpha } over all thresholds t."
        ),
        "topk_definition": (
            "Precision@K = fraction of actual fraud in the top-K "
            "transactions ranked by descending predicted fraud probability."
        ),
    }

    # ── Step 3: Bootstrap CIs ─────────────────────────────────────────────────
    logger.info("\n[STEP 3] Bootstrap 95%% CI (test set, targeting %d valid samples)", n_bootstrap)
    t0 = time.time()
    ci_test = bootstrap_metrics(y_true_test, y_prob_test, threshold=threshold,
                                n_bootstrap=n_bootstrap, seed=seed)
    ci_val  = bootstrap_metrics(y_true_val,  y_prob_val,  threshold=threshold,
                                n_bootstrap=n_bootstrap, seed=seed)
    elapsed = time.time() - t0
    logger.info("Bootstrap completed in %.1f seconds", elapsed)

    # Print a readable summary table
    logger.info("\nBootstrap CI Summary (test set):")
    logger.info("  %-12s  %10s  %10s  %10s  %8s", "Metric", "Mean", "CI Lower", "CI Upper", "N Valid")
    for metric, vals in ci_test.items():
        logger.info(
            "  %-12s  %10.4f  %10.4f  %10.4f  %8d",
            metric, vals["mean"], vals["ci_lower"], vals["ci_upper"], vals["n_valid"],
        )

    new_sections["phase15_bootstrap"] = {
        "test": ci_test,
        "validation": ci_val,
        "n_bootstrap_target": n_bootstrap,
        "seed": seed,
        "note": (
            "Bootstrap samples with zero positive-class examples were skipped "
            "and re-drawn. n_valid reports the number of usable iterations."
        ),
    }

    # ── Step 4: Calibration ───────────────────────────────────────────────────
    logger.info("\n[STEP 4] Calibration analysis (test set)")
    cal_test = compute_calibration_metrics(y_true_test, y_prob_test, n_bins=10)
    cal_val  = compute_calibration_metrics(y_true_val,  y_prob_val,  n_bins=10)

    logger.info(
        "Test  | Brier=%.4f (naive=%.4f) | ECE=%.4f",
        cal_test["brier_score"], cal_test["naive_brier"], cal_test["ece"],
    )
    logger.info(
        "Val   | Brier=%.4f (naive=%.4f) | ECE=%.4f",
        cal_val["brier_score"], cal_val["naive_brier"], cal_val["ece"],
    )

    # Save calibration curve plot
    cal_plot_path = exp_dir / "calibration_curve_test.png"
    plot_calibration_curve(y_true_test, y_prob_test, save_path=cal_plot_path,
                           title="E1 LightGBM — Calibration (Test Set)")

    new_sections["phase15_calibration"] = {
        "test": cal_test,
        "validation": cal_val,
        "plot": str(cal_plot_path),
        "note": (
            "Brier score lower is better; naive baseline predicts constant fraud_rate. "
            "ECE measures weighted mean gap between predicted probability and actual positive fraction."
        ),
    }

    # ── Save all new sections ─────────────────────────────────────────────────
    _append_to_metrics_json(exp_dir, new_sections)

    # ── Write human-readable operational metrics report ───────────────────────
    _write_operational_report(exp_dir, recall_fpr_test, recall_fpr_val,
                               prec_topk_test, prec_topk_val,
                               ci_test, cal_test, cal_val, threshold)

    logger.info("\n[DONE] Phase 1.5B complete. All outputs written to: %s", exp_dir)


def _write_operational_report(
    exp_dir: Path,
    recall_fpr_test: dict, recall_fpr_val: dict,
    prec_topk_test: dict, prec_topk_val: dict,
    ci_test: dict,
    cal_test: dict, cal_val: dict,
    threshold: float,
) -> None:
    """Write reports/phase1/operational_metrics_report.md."""
    report_dir = _ROOT / "reports" / "phase1"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / "operational_metrics_report.md"

    pr_ci = ci_test["pr_auc"]
    roc_ci = ci_test["roc_auc"]
    f1_ci = ci_test["f1"]
    rec_ci = ci_test["recall"]
    prec_ci = ci_test["precision"]

    lines = [
        "# E1 LightGBM — Phase 1.5B Operational Metrics Report",
        "",
        "**Experiment:** E1_lightgbm | **Split:** Test | **Threshold (frozen):** "
        f"`{threshold:.6f}`",
        "**Status:** Additive extension — E1 model and predictions are IMMUTABLE",
        "",
        "---",
        "",
        "## 1. Bootstrap Confidence Intervals (Test Set)",
        "",
        f"Targeting 2,000 valid bootstrap samples. "
        f"n_valid reported per metric.",
        "",
        "| Metric | Point Estimate | 95% CI Lower | 95% CI Upper | N Valid |",
        "|--------|---------------|-------------|-------------|---------|",
        f"| **PR-AUC** | {pr_ci['mean']:.4f} | {pr_ci['ci_lower']:.4f} | {pr_ci['ci_upper']:.4f} | {pr_ci['n_valid']} |",
        f"| ROC-AUC | {roc_ci['mean']:.4f} | {roc_ci['ci_lower']:.4f} | {roc_ci['ci_upper']:.4f} | {roc_ci['n_valid']} |",
        f"| F1 | {f1_ci['mean']:.4f} | {f1_ci['ci_lower']:.4f} | {f1_ci['ci_upper']:.4f} | {f1_ci['n_valid']} |",
        f"| Recall | {rec_ci['mean']:.4f} | {rec_ci['ci_lower']:.4f} | {rec_ci['ci_upper']:.4f} | {rec_ci['n_valid']} |",
        f"| Precision | {prec_ci['mean']:.4f} | {prec_ci['ci_lower']:.4f} | {prec_ci['ci_upper']:.4f} | {prec_ci['n_valid']} |",
        "",
        "> Bootstrap samples with zero positive-class examples are skipped and re-drawn.",
        "> All future experiment comparisons (E2 GRU, E1b encoding, etc.) should use CIs for significance assessment.",
        "",
        "---",
        "",
        "## 2. Recall at Fixed FPR Operating Points",
        "",
        "**Definition:** Recall@α = max{ TPR(t) : empirical_FPR(t) ≤ α } over all ROC thresholds.",
        "",
        "| FPR Target | Recall (Test) | Recall (Validation) |",
        "|-----------|--------------|---------------------|",
    ]
    fpr_keys = sorted(recall_fpr_test.keys())
    for key in fpr_keys:
        alpha_str = key.replace("recall_at_fpr_", "")
        alpha = float(alpha_str)
        lines.append(
            f"| **{alpha*100:.1f}%** | {recall_fpr_test[key]:.4f} | {recall_fpr_val[key]:.4f} |"
        )

    lines += [
        "",
        "> A 1% FPR on the test set (~74,784 legit transactions) corresponds to ~748 incorrectly flagged legitimate transactions.",
        "",
        "---",
        "",
        "## 3. Precision at Top-K",
        "",
        "**Definition:** Of the top-K transactions ranked by predicted fraud probability, "
        "what fraction are actually fraud?",
        "",
        "| K (investigations) | Precision (Test) | Precision (Validation) | Fraud in Top-K (est.) |",
        "|--------------------|-----------------|----------------------|----------------------|",
    ]
    for k_key in sorted(prec_topk_test.keys(), key=lambda x: int(x.split("_")[-1])):
        k = int(k_key.split("_")[-1])
        pt = prec_topk_test[k_key]
        pv = prec_topk_val[k_key]
        est_fraud = int(round(pt * k))
        lines.append(f"| {k:,} | {pt:.4f} | {pv:.4f} | ~{est_fraud} |")

    lines += [
        "",
        "> Useful for fraud teams with limited daily investigation capacity.",
        "> Compare against baseline (random) precision = fraud_rate ≈ 0.035.",
        "",
        "---",
        "",
        "## 4. Calibration Analysis",
        "",
        "| Metric | Test | Validation | Naive Baseline |",
        "|--------|------|-----------|----------------|",
        f"| Brier Score | {cal_test['brier_score']:.4f} | {cal_val['brier_score']:.4f} | {cal_test['naive_brier']:.4f} |",
        f"| ECE (10 bins) | {cal_test['ece']:.4f} | {cal_val['ece']:.4f} | N/A |",
        "",
        "**Brier Score:** Lower is better. Naive baseline = fraud_rate × (1 − fraud_rate).",
        "**ECE:** Expected Calibration Error — weighted mean gap between predicted probability "
        "and actual positive fraction per bin. Perfect = 0.",
        "",
        "See `experiments/E1_lightgbm/calibration_curve_test.png` for the reliability diagram.",
        "",
        "---",
        "",
        "*Generated by: experiments/run_phase15_extend.py | Phase 1.5B | E1 IMMUTABLE*",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Operational metrics report saved: %s", out_path)


logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1.5B — Statistical Evaluation Extension")
    parser.add_argument("--config", type=str, default="configs/phase1_lightgbm.yaml")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(Path(_ROOT) / args.config)
    exp_dir = Path(_ROOT) / cfg.get("experiment", {}).get("output_dir", "experiments/E1_lightgbm")

    log_path = exp_dir / "run_phase15b.log"
    exp_dir.mkdir(parents=True, exist_ok=True)
    global logger
    logger = get_logger(__name__, log_file=str(log_path))

    logger.info("Phase 1.5B Extension | E1 dir: %s", exp_dir)

    run_phase15b(exp_dir=exp_dir, n_bootstrap=args.n_bootstrap, seed=args.seed)


if __name__ == "__main__":
    main()
