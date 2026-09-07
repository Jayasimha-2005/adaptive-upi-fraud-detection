"""
src/evaluation/metrics.py
Fraud detection evaluation metrics for Phase 1.

PRIMARY METRIC: PR-AUC (Average Precision)
  Accuracy is NOT reported as a primary metric.
  At ~3.5% fraud rate, a trivial classifier that predicts all-legit
  achieves ~96.5% accuracy but detects zero fraud.

WHY PR-AUC?
  PR-AUC directly measures the tradeoff between precision and recall
  under class imbalance. Unlike ROC-AUC, it is sensitive to the
  minority class (fraud) and does not reward trivially low FPR.

THRESHOLD SELECTION POLICY
  The decision threshold is selected on validation data only.
  It is frozen before any test evaluation.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/CI environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    threshold: float,
    split_name: str = "",
) -> dict[str, Any]:
    """
    Compute the full fraud-detection metric suite.

    Parameters
    ----------
    y_true     : Ground-truth binary labels (0/1).
    y_prob     : Predicted fraud probabilities.
    threshold  : Decision threshold (must NOT be selected using test data).
    split_name : Human label for logging (e.g. "VALIDATION", "TEST").

    Returns
    -------
    dict with all metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (y_prob >= threshold).astype(int)

    pr_auc   = float(average_precision_score(y_true, y_prob))
    roc_auc  = float(roc_auc_score(y_true, y_prob))
    prec     = float(precision_score(y_true, y_pred, zero_division=0))
    rec      = float(recall_score(y_true, y_pred, zero_division=0))
    f1       = float(f1_score(y_true, y_pred, zero_division=0))
    mcc      = float(matthews_corrcoef(y_true, y_pred))
    bal_acc  = float(balanced_accuracy_score(y_true, y_pred))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    n_fraud   = int(y_true.sum())
    n_legit   = int(len(y_true) - n_fraud)
    n_pred_fraud = int(y_pred.sum())

    result = {
        "split":          split_name,
        "threshold":      round(threshold, 6),
        "pr_auc":         round(pr_auc,  6),
        "roc_auc":        round(roc_auc, 6),
        "precision":      round(prec,    6),
        "recall":         round(rec,     6),
        "f1":             round(f1,      6),
        "mcc":            round(mcc,     6),
        "balanced_accuracy": round(bal_acc, 6),
        "fpr":            round(fpr,     6),
        "fnr":            round(fnr,     6),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "actual_fraud":   n_fraud,
        "actual_legit":   n_legit,
        "predicted_fraud": n_pred_fraud,
        "fraud_pct":      round(n_fraud / len(y_true) * 100, 4),
    }

    logger.info(
        "[%s] PR-AUC=%.4f | ROC-AUC=%.4f | P=%.4f | R=%.4f | F1=%.4f | MCC=%.4f | FPR=%.4f | FNR=%.4f",
        split_name, pr_auc, roc_auc, prec, rec, f1, mcc, fpr, fnr,
    )
    return result


def select_threshold(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> float:
    """
    Select the optimal decision threshold on validation data by maximising
    the specified metric.

    IMPORTANT: This function must ONLY be called with validation data.
    The returned threshold is then frozen and applied to the test set.

    Parameters
    ----------
    y_true : Validation ground-truth labels.
    y_prob : Validation predicted probabilities.
    metric : One of "f1", "recall", "precision", "balanced_accuracy".

    Returns
    -------
    float : Optimal threshold.
    """
    y_true = np.asarray(y_true, dtype=int)
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_true, y_prob)

    if len(thresh_arr) == 0:
        return 0.5

    # Filter precision/recall arrays to match thresh_arr length
    p = prec_arr[:-1]
    r = rec_arr[:-1]

    if metric == "f1":
        denom = p + r
        scores = np.where(denom > 0, (2 * p * r) / denom, 0.0)
    elif metric == "precision":
        scores = p
    elif metric == "recall":
        scores = r
    elif metric == "balanced_accuracy":
        # Subsample grid for balanced accuracy if needed
        scores = np.zeros(len(thresh_arr))
        grid = np.linspace(0, len(thresh_arr) - 1, min(1000, len(thresh_arr)), dtype=int)
        best_t, best_s = 0.5, -1.0
        for idx in grid:
            t = thresh_arr[idx]
            y_pred = (y_prob >= t).astype(int)
            s = balanced_accuracy_score(y_true, y_pred)
            if s > best_s:
                best_s, best_t = s, float(t)
        logger.info("Threshold selected on validation: %.6f (max %s=%.4f)", best_t, metric, best_s)
        return best_t
    else:
        raise ValueError(f"Unknown threshold metric: {metric}")

    best_idx = np.argmax(scores)
    best_thresh = float(thresh_arr[best_idx])
    best_score = float(scores[best_idx])

    logger.info("Threshold selected on validation: %.6f (max %s=%.4f)", best_thresh, metric, best_score)
    return best_thresh


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    title: str,
    save_path: str | Path,
) -> None:
    """Plot and save the Precision-Recall curve."""
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    auc = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec, prec, lw=2, label=f"PR curve (AUC = {auc:.4f})")
    ax.axhline(y_true.mean(), color="gray", linestyle="--", label="Baseline (fraud rate)")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("PR curve saved: %s", save_path)


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str,
    save_path: str | Path,
) -> None:
    """Plot and save the ROC curve."""
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr_arr, tpr_arr, lw=2, label=f"ROC curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random classifier")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("ROC curve saved: %s", save_path)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: str | Path,
) -> None:
    """Plot and save a confusion matrix heatmap."""
    import matplotlib.patches as mpatches

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Legit", "Predicted Fraud"])
    ax.set_yticklabels(["Actual Legit", "Actual Fraud"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved: %s", save_path)


def save_metrics(
    val_metrics: dict,
    test_metrics: dict,
    out_dir: str | Path,
) -> None:
    """Save combined metrics JSON."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = {"validation": val_metrics, "test": test_metrics}
    out_path = out_dir / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    logger.info("Metrics saved: %s", out_path)


def recall_at_fixed_fpr(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    fpr_targets: list[float] | None = None,
) -> dict[str, float]:
    """
    Compute Recall (TPR) at fixed False Positive Rate operating points.

    DEFINITION
      For each target FPR α, select the highest achievable TPR among all
      thresholds whose empirical FPR ≤ α.
      Formally:
          Recall@α = max{ TPR(t) : FPR(t) ≤ α }
      where the maximisation is over all thresholds t on the ROC curve.

    WHY THIS MATTERS FOR FRAUD DETECTION
      A 1% FPR means ~748 legitimate transactions incorrectly flagged per day
      on a 74,784-transaction test set. Knowing recall at fixed FPR lets
      fraud operations teams choose an operating point that stays within their
      investigation capacity.

    Parameters
    ----------
    y_true      : Ground-truth binary labels (0/1).
    y_prob      : Predicted fraud probabilities.
    fpr_targets : List of FPR levels. Defaults to [0.001, 0.005, 0.01, 0.02].

    Returns
    -------
    dict mapping string key e.g. "recall_at_fpr_0.01" → float recall value.
    """
    if fpr_targets is None:
        fpr_targets = [0.001, 0.005, 0.01, 0.02]

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    # roc_curve returns (fpr, tpr, thresholds) sorted by ascending threshold
    # i.e. descending fpr/tpr.
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob)

    results: dict[str, float] = {}
    for alpha in fpr_targets:
        # All indices where empirical FPR ≤ alpha
        valid_mask = fpr_arr <= alpha
        if valid_mask.any():
            # Among those, pick the highest TPR
            best_tpr = float(tpr_arr[valid_mask].max())
        else:
            # Cannot achieve FPR ≤ alpha with any threshold (extremely rare)
            best_tpr = 0.0

        key = f"recall_at_fpr_{alpha:.3f}"
        results[key] = round(best_tpr, 6)
        logger.info("Recall @ FPR ≤ %.3f: %.4f", alpha, best_tpr)

    return results


def precision_at_top_k(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """
    Precision at Top-K predicted fraud transactions.

    Rank all transactions by predicted fraud probability (descending).
    For each K, compute: precision@K = (# true fraud in top-K) / K.

    WHY THIS MATTERS
      Fraud investigation teams have finite capacity. If a team can investigate
      1,000 flagged transactions per day, the relevant question is:
      "Of my top-1,000 model predictions, how many are actually fraud?"
      This models the real operational constraint much better than accuracy.

    Parameters
    ----------
    y_true   : Ground-truth binary labels (0/1).
    y_prob   : Predicted fraud probabilities.
    k_values : List of K thresholds. Defaults to [100, 500, 1000, 5000].

    Returns
    -------
    dict mapping string key e.g. "precision_at_top_1000" → float precision.
    """
    if k_values is None:
        k_values = [100, 500, 1000, 5000]

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)

    # Rank by descending predicted probability
    sorted_idx = np.argsort(-y_prob)

    results: dict[str, float] = {}
    for k in k_values:
        if k > n:
            logger.warning("precision_at_top_k: K=%d exceeds dataset size %d; using n.", k, n)
            k_eff = n
        else:
            k_eff = k

        top_k_labels = y_true[sorted_idx[:k_eff]]
        prec = float(top_k_labels.sum()) / k_eff
        key = f"precision_at_top_{k}"
        results[key] = round(prec, 6)
        logger.info("Precision @ Top-%d: %.4f (%d fraud in top-%d)",
                    k, prec, int(top_k_labels.sum()), k_eff)

    return results
