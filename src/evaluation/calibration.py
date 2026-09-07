"""
src/evaluation/calibration.py
Model calibration analysis for fraud-detection probability outputs.

WHAT IS CALIBRATION?
  A well-calibrated model means: "if the model assigns probability 0.7 to
  a transaction, then ~70% of such transactions should actually be fraud."
  Poor calibration means predictions are systematically over- or under-
  confident, which matters for any downstream risk-score routing system.

METRICS
  Brier Score : Mean squared error between predicted probability and true label.
                Perfect = 0. Baseline (naive fraud rate predictor) = fraud_rate * (1 - fraud_rate).
  ECE         : Expected Calibration Error — weighted mean absolute difference
                between per-bin accuracy and per-bin confidence.
  Reliability diagram (plot): Actual fraction positive per probability bucket.

NOTE
  At 3.5% fraud, calibration plots can be dominated by the majority class.
  We plot both unconditioned (all transactions) and the probability range
  [0.1, 1.0] for better visual resolution around the fraud region.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Brier score = mean squared error between predicted probability and true label.

    Lower is better. Range: [0, 1].
    Naive baseline (always predict fraud_rate) = fraud_rate * (1 - fraud_rate).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    brier = float(np.mean((y_prob - y_true) ** 2))
    fraud_rate = float(y_true.mean())
    baseline = fraud_rate * (1.0 - fraud_rate)
    logger.info("Brier Score: %.6f (naive baseline: %.6f)", brier, baseline)
    return brier


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE).

    Divides predictions into n_bins equal-width confidence buckets.
    For each bin, computes |mean_confidence - fraction_positive| weighted
    by bin size.

    Lower is better. Perfect calibration = 0.

    Parameters
    ----------
    y_true : Ground-truth binary labels.
    y_prob : Predicted fraud probabilities.
    n_bins : Number of equal-width confidence bins. Default 10.

    Returns
    -------
    float : ECE value.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)  # include 1.0 in last bin

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        mean_conf = float(y_prob[mask].mean())
        frac_pos = float(y_true[mask].mean())
        ece += (n_bin / n) * abs(mean_conf - frac_pos)

    logger.info("ECE (%d bins): %.6f", n_bins, ece)
    return float(ece)


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str | Path,
    n_bins: int = 10,
    title: str = "Calibration Curve (Reliability Diagram)",
) -> None:
    """
    Plot and save a reliability diagram.

    Two panels:
    1. Full probability range [0, 1] — shows overall calibration.
    2. Zoomed range [0, 0.5] — resolution around the fraud decision region.

    Parameters
    ----------
    y_true    : Ground-truth binary labels.
    y_prob    : Predicted fraud probabilities.
    save_path : Output path for the PNG file.
    n_bins    : Number of confidence bins.
    title     : Plot title.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    save_path = Path(save_path)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers, fraction_pos, mean_conf_per_bin, bin_counts = [], [], [], []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        bin_centers.append((lo + hi) / 2.0)
        fraction_pos.append(float(y_true[mask].mean()))
        mean_conf_per_bin.append(float(y_prob[mask].mean()))
        bin_counts.append(int(n_bin))

    bin_centers = np.array(bin_centers)
    fraction_pos = np.array(fraction_pos)
    mean_conf_per_bin = np.array(mean_conf_per_bin)

    fraud_rate = float(y_true.mean())
    brier = compute_brier_score(y_true, y_prob)
    ece = compute_ece(y_true, y_prob, n_bins=n_bins)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax_idx, (ax, xlim, title_suffix) in enumerate(zip(
        axes,
        [(0, 1), (0, 0.5)],
        ["Full range", "Zoomed [0, 0.5]"],
    )):
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")
        ax.axhline(fraud_rate, color="gray", linestyle=":", lw=1.2,
                   label=f"Fraud rate ({fraud_rate:.4f})")
        ax.plot(mean_conf_per_bin, fraction_pos, "o-", color="#2196F3", lw=2,
                markersize=6, label="Model calibration")
        ax.set_xlim(xlim)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"{title}\n({title_suffix})\nBrier={brier:.4f} | ECE={ece:.4f}")
        ax.legend(loc="upper left", fontsize=8)

        # Histogram inset showing prediction distribution
        ax2 = ax.twinx()
        ax2.hist(y_prob, bins=50, range=xlim, alpha=0.15, color="#FF5722", label="Prediction hist")
        ax2.set_ylabel("Count", color="#FF5722", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="#FF5722", labelsize=7)
        ax2.set_ylim(0, len(y_prob) * 2)

    plt.suptitle(f"E1 LightGBM Calibration | Brier={brier:.4f} | ECE={ece:.4f}", fontsize=11)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Calibration curve saved: %s", save_path)


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """
    Return Brier score and ECE as a summary dict.
    """
    return {
        "brier_score": round(compute_brier_score(y_true, y_prob), 6),
        "ece":         round(compute_ece(y_true, y_prob, n_bins=n_bins), 6),
        "fraud_rate":  round(float(np.asarray(y_true, dtype=float).mean()), 6),
        "naive_brier": round(
            float(np.asarray(y_true, dtype=float).mean()) *
            (1 - float(np.asarray(y_true, dtype=float).mean())), 6
        ),
        "n_bins": n_bins,
    }
