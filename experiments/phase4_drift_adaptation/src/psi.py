"""
psi.py — PSI Drift Detector for Phase 4
=========================================
Implements Population Stability Index computation exactly as specified
in Protocol v1.1 Section 8 and PHASE4_PROTOCOL_MANIFEST.json.

LOCKED SPECIFICATION:
- 10 quantile bins for numerical features
- Bin edges fitted on months 0-3 ONLY — never refit
- Out-of-range values clipped to outermost reference bin
- epsilon = 1e-6 for smoothing (prevents log(0))
- Categorical: frequency-based PSI vs months 0-3 reference
- __UNSEEN__ bucket for categories not in reference
- Trigger: count >= 6 of 29 features with PSI >= 0.10
- Reference distribution NEVER reset after retraining
"""
from __future__ import annotations
import json
import pathlib
import numpy as np
import pandas as pd
from typing import Optional

# --- Protocol constants (locked) ---
N_BINS          = 10
PSI_THRESHOLD   = 0.10
TRIGGER_COUNT   = 6
FEATURE_DENOM   = 29
EPSILON         = 1e-6
UNSEEN_LABEL    = "__UNSEEN__"


class PSIDetector:
    """
    Computes PSI for all 29 model-input features against a fixed reference
    distribution built from months 0-3.

    Usage:
        detector = PSIDetector()
        detector.fit(X_train)                  # fit on months 0-3 only
        result = detector.compute(X_monitor)   # any monitoring window
        triggered = result["triggered"]
    """

    def __init__(
        self,
        numerical_cols: list[str],
        categorical_cols: list[str],
        n_bins: int = N_BINS,
        psi_threshold: float = PSI_THRESHOLD,
        trigger_count: int = TRIGGER_COUNT,
        epsilon: float = EPSILON,
    ):
        self.numerical_cols   = numerical_cols
        self.categorical_cols = categorical_cols
        self.n_bins           = n_bins
        self.psi_threshold    = psi_threshold
        self.trigger_count    = trigger_count
        self.epsilon          = epsilon
        self.is_fitted        = False

        # Reference storage (set at fit time, never changed again)
        self._num_bin_edges: dict[str, np.ndarray] = {}
        self._num_ref_props: dict[str, np.ndarray] = {}
        self._cat_ref_props: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Fitting — months 0-3 ONLY
    # ------------------------------------------------------------------
    def fit(self, X_ref: pd.DataFrame) -> "PSIDetector":
        """
        Build reference distributions from training data (months 0-3).
        This method must be called EXACTLY ONCE on the initial training
        data and never called again, even after retraining.
        """
        assert not self.is_fitted, (
            "PSIDetector.fit() called more than once. "
            "Reference must remain months 0-3 permanently."
        )

        # Numerical: quantile bin edges
        for col in self.numerical_cols:
            vals = X_ref[col].dropna().values.astype(float)
            # linspace over quantiles to get n_bins bin edges
            quantiles = np.linspace(0, 100, self.n_bins + 1)
            edges = np.percentile(vals, quantiles)
            # Deduplicate edges (edge case: constant-like features)
            edges = np.unique(edges)
            # Extend to (-inf, +inf) to capture all future values
            edges[0]  = -np.inf
            edges[-1] = +np.inf
            self._num_bin_edges[col] = edges
            # Reference bin proportions
            counts, _ = np.histogram(vals, bins=edges)
            props = (counts + self.epsilon) / (counts.sum() + self.epsilon * len(counts))
            self._num_ref_props[col] = props

        # Categorical: category frequency distribution
        for col in self.categorical_cols:
            vc = X_ref[col].astype(str).value_counts(normalize=True)
            self._cat_ref_props[col] = vc.to_dict()

        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # PSI computation for a monitoring window
    # ------------------------------------------------------------------
    def _psi_numerical(self, col: str, X_mon: pd.DataFrame) -> float:
        edges   = self._num_bin_edges[col]
        ref_p   = self._num_ref_props[col]
        vals    = X_mon[col].dropna().values.astype(float)
        # Clip out-of-range values to outermost reference bins
        vals    = np.clip(vals, edges[1], edges[-2])  # use inner bounds
        counts, _ = np.histogram(vals, bins=edges)
        cur_p = (counts + self.epsilon) / (counts.sum() + self.epsilon * len(counts))
        psi = float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))
        return psi

    def _psi_categorical(self, col: str, X_mon: pd.DataFrame) -> float:
        ref_dist = self._cat_ref_props[col]
        cur_vc   = X_mon[col].astype(str).value_counts(normalize=False)
        total    = cur_vc.sum()
        categories = set(ref_dist.keys()) | {UNSEEN_LABEL}
        psi = 0.0
        unseen_count = 0
        for cat, ref_p in ref_dist.items():
            cur_count = cur_vc.get(cat, 0)
            cur_p = (cur_count + self.epsilon) / (total + self.epsilon * len(categories))
            ref_ps = (ref_p + self.epsilon / len(categories))
            psi += (cur_p - ref_ps) * np.log(cur_p / ref_ps)
        # Unseen categories
        unseen_count = sum(
            cnt for cat, cnt in cur_vc.items() if cat not in ref_dist
        )
        if unseen_count > 0:
            cur_p = (unseen_count + self.epsilon) / (total + self.epsilon * len(categories))
            ref_ps = self.epsilon / len(categories)
            if ref_ps > 0 and cur_p > 0:
                psi += (cur_p - ref_ps) * np.log(cur_p / ref_ps)
        return float(psi)

    def compute(
        self,
        X_mon: pd.DataFrame,
        window_label: str = "unknown",
    ) -> dict:
        """
        Compute PSI for all 29 features against the fixed months 0-3 reference.

        Returns dict with:
            psi_values        — {feature: psi_value}
            triggered_features — list of features with PSI >= threshold
            triggered_count   — int
            triggered         — bool (count >= TRIGGER_COUNT)
            window_label      — str
            n_observations    — int
            psi_threshold     — float
            trigger_count_required — int
        """
        assert self.is_fitted, "Call fit() before compute()"
        assert len(X_mon) >= 50000, (
            f"Monitoring window too small: {len(X_mon)} < 50000"
        )

        psi_values: dict[str, float] = {}
        for col in self.numerical_cols:
            psi_values[col] = self._psi_numerical(col, X_mon)
        for col in self.categorical_cols:
            psi_values[col] = self._psi_categorical(col, X_mon)

        triggered_features = [
            col for col, v in psi_values.items() if v >= self.psi_threshold
        ]
        triggered_count = len(triggered_features)
        triggered = triggered_count >= self.trigger_count

        return {
            "window_label":          window_label,
            "psi_values":            psi_values,
            "triggered_features":    triggered_features,
            "triggered_count":       triggered_count,
            "triggered":             triggered,
            "n_observations":        len(X_mon),
            "psi_threshold":         self.psi_threshold,
            "trigger_count_required": self.trigger_count,
            "feature_denominator":   FEATURE_DENOM,
            "trigger_fraction":      f"{triggered_count}/{FEATURE_DENOM}",
        }

    # ------------------------------------------------------------------
    # Serialisation (reference distributions only — for audit trail)
    # ------------------------------------------------------------------
    def save_reference(self, path: str | pathlib.Path) -> None:
        """Save the fitted reference distributions for audit purposes."""
        assert self.is_fitted
        payload = {
            "n_bins":         self.n_bins,
            "psi_threshold":  self.psi_threshold,
            "trigger_count":  self.trigger_count,
            "epsilon":        self.epsilon,
            "numerical_bin_edges": {
                col: edges.tolist()
                for col, edges in self._num_bin_edges.items()
            },
            "numerical_ref_props": {
                col: props.tolist()
                for col, props in self._num_ref_props.items()
            },
            "categorical_ref_props": self._cat_ref_props,
        }
        pathlib.Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load_reference(
        cls,
        path: str | pathlib.Path,
        numerical_cols: list[str],
        categorical_cols: list[str],
    ) -> "PSIDetector":
        """Reload a saved reference (for verification — never to refit)."""
        payload = json.loads(pathlib.Path(path).read_text())
        detector = cls(
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
            n_bins=payload["n_bins"],
            psi_threshold=payload["psi_threshold"],
            trigger_count=payload["trigger_count"],
            epsilon=payload["epsilon"],
        )
        detector._num_bin_edges = {
            col: np.array(v) for col, v in payload["numerical_bin_edges"].items()
        }
        detector._num_ref_props = {
            col: np.array(v) for col, v in payload["numerical_ref_props"].items()
        }
        detector._cat_ref_props = payload["categorical_ref_props"]
        detector.is_fitted = True
        return detector
