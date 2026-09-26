"""
threshold.py — Month-4 Threshold Selection for Phase 4
========================================================
Implements the PERMANENTLY FROZEN threshold selection policy from
Protocol v1.1 Section 10.

RULES (non-negotiable):
1. Threshold is selected EXACTLY ONCE using v1 predictions on Month 4.
2. It is NEVER re-derived for v2 or v3 (Month 4 enters their training data).
3. Sweep: 200 thresholds, 0.01 to 0.99, F1-max selection.
4. The frozen threshold is saved as an immutable artifact.
5. No test-period labels (months 5/6/7) are used.
"""
from __future__ import annotations
import hashlib
import json
import pathlib
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


# Protocol constants
N_THRESHOLDS  = 200
THRESH_MIN    = 0.01
THRESH_MAX    = 0.99
SELECTION_RULE = "F1-max"


def select_threshold(
    y_val: np.ndarray,
    proba_val: np.ndarray,
    validation_period: int = 4,
    model_version: str = "v1",
    output_path: str | pathlib.Path | None = None,
) -> dict:
    """
    Select the F1-max classification threshold from Month-4 validation data.

    Parameters
    ----------
    y_val             : Ground-truth labels for Month 4
    proba_val         : Model v1 fraud probabilities for Month 4
    validation_period : Must be 4 (asserted)
    model_version     : Must be 'v1' (asserted — only v1 threshold valid)
    output_path       : Where to save the frozen threshold artifact

    Returns
    -------
    dict with: threshold, validation_f1, sweep_results, selection_rule,
               model_version, validation_period, artifact_path
    """
    assert validation_period == 4, (
        f"Threshold must be selected on Month 4, not Month {validation_period}"
    )
    assert model_version == "v1", (
        f"Threshold must be selected using v1 predictions, not {model_version}. "
        "Month 4 becomes part of v2/v3 training data — re-selection is invalid."
    )
    assert len(y_val) == len(proba_val)

    thresholds = np.linspace(THRESH_MIN, THRESH_MAX, N_THRESHOLDS)
    sweep = []
    best_f1, best_thresh, best_idx = -1.0, 0.5, 0
    for i, t in enumerate(thresholds):
        preds = (proba_val >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        prec = precision_score(y_val, preds, zero_division=0)
        rec  = recall_score(y_val, preds, zero_division=0)
        sweep.append({"threshold": float(t), "f1": float(f1),
                       "precision": float(prec), "recall": float(rec)})
        if f1 > best_f1:
            best_f1, best_thresh, best_idx = f1, float(t), i

    result = {
        "threshold":         best_thresh,
        "validation_f1":     best_f1,
        "validation_period": validation_period,
        "model_version":     model_version,
        "selection_rule":    SELECTION_RULE,
        "n_thresholds_swept": N_THRESHOLDS,
        "threshold_min":     THRESH_MIN,
        "threshold_max":     THRESH_MAX,
        "sweep_results":     sweep,
        "frozen":            True,
        "note": (
            "This threshold is selected exactly once from v1 predictions on "
            "Month 4 and is permanently applied to v1, v2, and v3. "
            "Re-selection after v2/v3 retraining is prohibited because "
            "Month 4 becomes part of their training data."
        ),
    }

    if output_path is not None:
        out = pathlib.Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        # Hash the artifact
        h = hashlib.sha256(out.read_bytes()).hexdigest()
        result["artifact_path"] = str(out)
        result["artifact_hash"] = h
        print(f"  Threshold frozen: {best_thresh:.4f} (val F1={best_f1:.4f})")
        print(f"  Artifact saved -> {out} (SHA256: {h[:16]})")

    return result


def load_frozen_threshold(path: str | pathlib.Path) -> float:
    """Load and return the single frozen threshold value."""
    data = json.loads(pathlib.Path(path).read_text())
    assert data.get("frozen") is True, "Threshold artifact not marked as frozen"
    assert data.get("model_version") == "v1", "Threshold must be from v1"
    return float(data["threshold"])


def apply_threshold(proba: np.ndarray, threshold: float) -> np.ndarray:
    """Apply the frozen threshold to produce binary predictions."""
    return (proba >= threshold).astype(int)
