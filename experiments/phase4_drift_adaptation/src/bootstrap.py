"""
bootstrap.py — Paired Bootstrap for Phase 4 Statistical Testing
================================================================
Implements the paired bootstrap comparison from Protocol v1.1 Section 25.

LOCKED PARAMETERS:
    n_resamples = 2000
    seed        = 42
    metric      = PR-AUC
    method      = paired (same row indices for both models)
    CI level    = 95%
"""
from __future__ import annotations
import json
import numpy as np
from sklearn.metrics import average_precision_score

N_RESAMPLES = 2000
SEED        = 42
CI_LEVEL    = 0.95


def paired_bootstrap_pr_auc(
    y_true: np.ndarray,
    static_proba: np.ndarray,
    adaptive_proba: np.ndarray,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
    period_label: str = "unknown",
) -> dict:
    """
    Compute paired bootstrap confidence interval for delta PR-AUC
    (adaptive - static) using identical row indices.

    Parameters
    ----------
    y_true          : Ground-truth labels (same for both models)
    static_proba    : Static model fraud probabilities
    adaptive_proba  : Adaptive model fraud probabilities
    n_resamples     : Number of bootstrap resamples (2000)
    seed            : Random seed (42)
    period_label    : Which evaluation period

    Returns
    -------
    dict with: observed_delta, ci_lower, ci_upper, n_resamples, seed,
               static_pr_auc, adaptive_pr_auc, bootstrap_deltas
    """
    assert len(y_true) == len(static_proba) == len(adaptive_proba), (
        "Row count mismatch — paired bootstrap requires identical indices."
    )
    assert n_resamples == N_RESAMPLES, f"Bootstrap must use {N_RESAMPLES} resamples"
    assert seed == SEED, f"Bootstrap must use seed={SEED}"

    rng = np.random.default_rng(seed)
    n = len(y_true)
    y_true = np.asarray(y_true)
    static_proba   = np.asarray(static_proba)
    adaptive_proba = np.asarray(adaptive_proba)

    observed_static   = float(average_precision_score(y_true, static_proba))
    observed_adaptive = float(average_precision_score(y_true, adaptive_proba))
    observed_delta    = observed_adaptive - observed_static

    deltas = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        y_b   = y_true[idx]
        # Skip degenerate resamples (single class)
        if y_b.sum() == 0 or y_b.sum() == len(y_b):
            deltas.append(observed_delta)
            continue
        s_b = float(average_precision_score(y_b, static_proba[idx]))
        a_b = float(average_precision_score(y_b, adaptive_proba[idx]))
        deltas.append(a_b - s_b)

    deltas = np.array(deltas)
    alpha  = 1.0 - CI_LEVEL
    ci_lower = float(np.percentile(deltas, 100 * alpha / 2))
    ci_upper = float(np.percentile(deltas, 100 * (1 - alpha / 2)))

    result = {
        "period":             period_label,
        "static_PR_AUC":     observed_static,
        "adaptive_PR_AUC":   observed_adaptive,
        "observed_delta":    observed_delta,
        "ci_lower":          ci_lower,
        "ci_upper":          ci_upper,
        "ci_level":          CI_LEVEL,
        "n_resamples":       n_resamples,
        "seed":              seed,
        "n_obs":             n,
        "bootstrap_deltas":  deltas.tolist(),
    }
    return result


def format_bootstrap_result(result: dict) -> str:
    """Human-readable summary of bootstrap result."""
    obs   = result["observed_delta"]
    lo    = result["ci_lower"]
    hi    = result["ci_upper"]
    sa    = result["static_PR_AUC"]
    ad    = result["adaptive_PR_AUC"]
    period = result["period"]
    interpretation = "No significant difference" if lo <= 0 <= hi else (
        "Adaptive significantly higher" if lo > 0 else "Static significantly higher"
    )
    return (
        f"Bootstrap ({period}):\n"
        f"  Static PR-AUC:   {sa:.6f}\n"
        f"  Adaptive PR-AUC: {ad:.6f}\n"
        f"  Delta (A-S):     {obs:+.6f}\n"
        f"  95% CI:          [{lo:+.6f}, {hi:+.6f}]\n"
        f"  Interpretation:  {interpretation}\n"
        f"  Resamples: {result['n_resamples']}, seed={result['seed']}"
    )
