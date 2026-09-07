"""
src/evaluation/bootstrap.py
Bootstrap confidence intervals for fraud-detection evaluation metrics.

DESIGN PRINCIPLE
  Bootstrap samples of a 3.5% fraud dataset can theoretically contain
  zero positive-class examples, which makes PR-AUC and recall undefined.
  This module targets a fixed number of VALID samples by skipping and
  re-drawing any iteration that contains no positive-class examples.
  n_valid is always reported alongside the computed CIs.

USAGE
  from src.evaluation.bootstrap import bootstrap_metrics

  ci = bootstrap_metrics(y_true, y_prob, threshold=0.616521, n_bootstrap=2000)
  print(ci["pr_auc"])
  # {"mean": 0.5317, "ci_lower": 0.5191, "ci_upper": 0.5443, "n_valid": 2000}
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# Maximum total draws allowed per valid-sample target (safety cap to prevent
# infinite loops on pathologically small / zero-fraud datasets).
_MAX_DRAW_MULTIPLIER = 10


def bootstrap_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """
    Compute 95% bootstrap confidence intervals for the core metric suite.

    Parameters
    ----------
    y_true      : Ground-truth binary labels (0/1). Shape (n,).
    y_prob      : Predicted fraud probabilities. Shape (n,).
    threshold   : Frozen decision threshold (selected on validation only).
    n_bootstrap : Target number of VALID bootstrap iterations.
                  Iterations with no positive-class examples are skipped
                  and re-drawn until n_bootstrap valid samples are obtained.
    seed        : Random seed for reproducibility.

    Returns
    -------
    dict mapping metric name → {"mean", "ci_lower", "ci_upper", "n_valid"}

    Metrics computed
    ----------------
    pr_auc, roc_auc, precision, recall, f1, mcc
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    metric_names = ["pr_auc", "roc_auc", "precision", "recall", "f1", "mcc"]
    accumulators: dict[str, list[float]] = {m: [] for m in metric_names}

    n_valid = 0
    n_total_draws = 0
    max_draws = n_bootstrap * _MAX_DRAW_MULTIPLIER

    logger.info(
        "Bootstrap CI: targeting %d valid samples (seed=%d, threshold=%.6f)",
        n_bootstrap, seed, threshold,
    )

    while n_valid < n_bootstrap and n_total_draws < max_draws:
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        n_total_draws += 1

        # Skip any draw that has zero positive-class examples — PR-AUC,
        # recall, F1 are undefined without at least one fraud instance.
        if yt.sum() == 0:
            logger.debug("Bootstrap draw %d skipped: zero positive examples", n_total_draws)
            continue

        # Skip draws that have only positive examples (precision undefined for
        # the legit class, though this is extremely unlikely in practice).
        if (1 - yt).sum() == 0:
            logger.debug("Bootstrap draw %d skipped: zero negative examples", n_total_draws)
            continue

        y_pred = (yp >= threshold).astype(int)

        accumulators["pr_auc"].append(float(average_precision_score(yt, yp)))
        accumulators["roc_auc"].append(float(roc_auc_score(yt, yp)))
        accumulators["precision"].append(float(precision_score(yt, y_pred, zero_division=0)))
        accumulators["recall"].append(float(recall_score(yt, y_pred, zero_division=0)))
        accumulators["f1"].append(float(f1_score(yt, y_pred, zero_division=0)))
        accumulators["mcc"].append(float(matthews_corrcoef(yt, y_pred)))

        n_valid += 1

        if n_valid % 500 == 0:
            logger.info("  Bootstrap progress: %d / %d valid", n_valid, n_bootstrap)

    if n_valid < n_bootstrap:
        logger.warning(
            "Bootstrap: only %d valid samples obtained after %d total draws "
            "(target was %d). Results may be less reliable.",
            n_valid, n_total_draws, n_bootstrap,
        )

    logger.info(
        "Bootstrap complete: %d valid samples from %d total draws",
        n_valid, n_total_draws,
    )

    results: dict[str, dict[str, Any]] = {}
    for name, values in accumulators.items():
        arr = np.array(values)
        results[name] = {
            "mean":     round(float(arr.mean()), 6),
            "ci_lower": round(float(np.percentile(arr, 2.5)), 6),
            "ci_upper": round(float(np.percentile(arr, 97.5)), 6),
            "std":      round(float(arr.std()), 6),
            "n_valid":  n_valid,
        }

    # Log summary
    pr = results["pr_auc"]
    roc = results["roc_auc"]
    logger.info(
        "PR-AUC = %.4f [95%% CI: %.4f – %.4f] | ROC-AUC = %.4f [95%% CI: %.4f – %.4f]",
        pr["mean"], pr["ci_lower"], pr["ci_upper"],
        roc["mean"], roc["ci_lower"], roc["ci_upper"],
    )

    return results
