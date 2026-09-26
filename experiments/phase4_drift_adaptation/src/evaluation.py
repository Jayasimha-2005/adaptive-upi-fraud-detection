"""
evaluation.py — Metric Computation for Phase 4
================================================
Computes all protocol-defined metrics for the Static vs Adaptive comparison.
Applies the FROZEN threshold for threshold-dependent metrics.
PR-AUC and ROC-AUC are always computed threshold-independently.

Protocol metrics (Section 11):
  Primary:    PR-AUC
  Secondary:  ROC-AUC, Precision, Recall, F1, MCC, Balanced Accuracy,
              FPR, FNR, Brier Score,
              Recall@FPR=0.1%, 0.5%, 1%, 2%,
              Precision@Top-100, @Top-500, @Top-1000
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    brier_score_loss,
    precision_recall_curve,
    roc_curve,
)


def _recall_at_fpr(y_true: np.ndarray, proba: np.ndarray, target_fpr: float) -> float:
    """Recall when FPR is constrained to <= target_fpr."""
    fpr_arr, tpr_arr, _ = roc_curve(y_true, proba)
    idx = np.searchsorted(fpr_arr, target_fpr, side="right") - 1
    idx = max(0, min(idx, len(tpr_arr) - 1))
    return float(tpr_arr[idx])


def _precision_at_top_k(y_true: np.ndarray, proba: np.ndarray, k: int) -> float:
    """Precision among the top-k highest-probability predictions."""
    if k <= 0 or k > len(y_true):
        return float("nan")
    top_k_idx = np.argsort(proba)[::-1][:k]
    return float(y_true[top_k_idx].sum() / k)


def compute_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    period_label: str,
    model_label: str,
) -> dict:
    """
    Compute all protocol-defined metrics for one model on one period.

    Parameters
    ----------
    y_true       : Ground-truth binary labels
    proba        : Fraud probability predictions
    threshold    : The PERMANENTLY FROZEN Month-4 v1 threshold
    period_label : 'month6' or 'month7'
    model_label  : 'static_v1' or 'adaptive_vX'

    Returns full metric dict.
    """
    y_pred = (proba >= threshold).astype(int)

    pr_auc   = float(average_precision_score(y_true, proba))
    roc_auc  = float(roc_auc_score(y_true, proba))
    prec     = float(precision_score(y_true, y_pred, zero_division=0))
    rec      = float(recall_score(y_true, y_pred, zero_division=0))
    f1       = float(f1_score(y_true, y_pred, zero_division=0))
    mcc      = float(matthews_corrcoef(y_true, y_pred))
    bal_acc  = float(balanced_accuracy_score(y_true, y_pred))
    brier    = float(brier_score_loss(y_true, proba))

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr_val = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "period":          period_label,
        "model":           model_label,
        "threshold_used":  threshold,
        "n_obs":           len(y_true),
        "n_fraud":         int(y_true.sum()),
        "fraud_rate":      float(y_true.mean()),
        # Primary
        "PR_AUC":          pr_auc,
        # Secondary
        "ROC_AUC":         roc_auc,
        "Precision":       prec,
        "Recall":          rec,
        "F1":              f1,
        "MCC":             mcc,
        "Balanced_Acc":    bal_acc,
        "FPR":             fpr_val,
        "FNR":             fnr_val,
        "Brier":           brier,
        # Recall @ FPR
        "Recall_at_FPR_0001": _recall_at_fpr(y_true, proba, 0.001),
        "Recall_at_FPR_0005": _recall_at_fpr(y_true, proba, 0.005),
        "Recall_at_FPR_001":  _recall_at_fpr(y_true, proba, 0.010),
        "Recall_at_FPR_002":  _recall_at_fpr(y_true, proba, 0.020),
        # Precision @ Top-K
        "P_at_100":  _precision_at_top_k(y_true, proba, 100),
        "P_at_500":  _precision_at_top_k(y_true, proba, 500),
        "P_at_1000": _precision_at_top_k(y_true, proba, 1000),
        # Confusion
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
    }


def compare_periods(
    y_true: np.ndarray,
    static_proba: np.ndarray,
    adaptive_proba: np.ndarray,
    threshold: float,
    period_label: str,
    adaptive_version: str,
) -> dict:
    """
    Compute metrics for both static and adaptive models on the same period rows.
    Both models are evaluated on the IDENTICAL observation indices.
    """
    assert len(y_true) == len(static_proba) == len(adaptive_proba), (
        "Row count mismatch between static and adaptive predictions — "
        "both must be evaluated on the exact same rows."
    )
    static_metrics   = compute_metrics(
        y_true, static_proba, threshold, period_label, "static_v1"
    )
    adaptive_metrics = compute_metrics(
        y_true, adaptive_proba, threshold, period_label, f"adaptive_{adaptive_version}"
    )
    delta_pr_auc = adaptive_metrics["PR_AUC"] - static_metrics["PR_AUC"]
    return {
        "period":           period_label,
        "adaptive_version": adaptive_version,
        "static":           static_metrics,
        "adaptive":         adaptive_metrics,
        "delta_PR_AUC":     delta_pr_auc,
    }
