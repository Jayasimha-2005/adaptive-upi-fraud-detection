"""
Phase 2 — E2b Feature Preprocessing
======================================
StandardScaler applied to E2 sequence features for the E2b experiment.

Design principles (locked):
- Fit statistics on TRAINING sequences ONLY.
  Validation and test are transformed using frozen train statistics.
  No val/test data influences the fitted parameters.
- Applied feature-wise across all 4 history time steps equally.
  The scaler treats [N, 4, 406] as [N*4, 406] for fitting.
  This is consistent because each time step is a transaction with the
  same 406 features — scale statistics should be the same per feature
  regardless of sequence position.
- The gap feature (index 405, `time_since_previous_transaction`) is
  already log1p-transformed in sequence_builder.py. StandardScaler
  will additionally z-score it, which is appropriate.
- Zero-variance features (if any) are handled by setting std=1 to
  avoid division-by-zero (standard sklearn behavior).
- Feature count remains exactly 406. Feature ordering unchanged.

NOT applied to:
- Target labels (y)
- card1 (not in input)
- isFraud / TransactionID / TransactionDT (not in input)

CRITICAL:
  The scaler is fitted ONCE on X_train and then serialized.
  The same fitted scaler must be used for X_val, X_test, and any
  future inference. Never refit on val or test.
"""

from __future__ import annotations

import json
import logging
import pathlib
import pickle

import numpy as np

log = logging.getLogger(__name__)

SCALER_PATH = pathlib.Path("phase2_gru/artifacts/E2b_scaled/preprocessing/standard_scaler.pkl")
SCALER_STATS_PATH = pathlib.Path("phase2_gru/artifacts/E2b_scaled/preprocessing/scaler_stats.json")


class SequenceStandardScaler:
    """
    Feature-wise StandardScaler for E2 sequence tensors.

    Fits on training data [N_train, 4, 406] → computes per-feature
    mean and std across all (N_train * 4) observations.
    Transforms any split [N, 4, 406] using the frozen train statistics.

    Attributes:
        mean_   : np.ndarray [406]  per-feature mean (train only)
        scale_  : np.ndarray [406]  per-feature std  (train only, min 1e-8)
        n_features : int  (406)
        is_fitted  : bool
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.n_features: int = 0
        self.is_fitted: bool = False

    def fit(self, X_train: np.ndarray) -> "SequenceStandardScaler":
        """
        Compute mean and std from TRAINING sequences only.

        Args:
            X_train : float32 array [N_train, 4, 406]

        Returns:
            self (fitted)
        """
        assert X_train.ndim == 3, f"Expected 3D [N,4,406], got {X_train.ndim}D"
        N, T, F = X_train.shape
        # Reshape to [N*T, F] — treat all time steps as independent observations
        X_flat = X_train.reshape(-1, F).astype(np.float64)
        self.mean_  = X_flat.mean(axis=0).astype(np.float32)
        std         = X_flat.std(axis=0).astype(np.float32)
        # Guard against zero-variance features: set std to 1.0
        # (zero-variance features become all-zeros after centering, which is fine)
        self.scale_ = np.where(std < 1e-8, np.float32(1.0), std)
        self.n_features = F
        self.is_fitted = True
        n_zero_var = int((std < 1e-8).sum())
        log.info(
            "SequenceStandardScaler fitted on %d samples (%d×%d). "
            "Features: %d total, %d near-zero-variance (std set to 1.0).",
            N * T, N, T, F, n_zero_var
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply fitted scaling to a sequence array.

        Args:
            X : float32 array [N, 4, 406]

        Returns:
            X_scaled : float32 array [N, 4, 406]
        """
        self._check_fitted()
        assert X.shape[2] == self.n_features, \
            f"Feature dim mismatch: expected {self.n_features}, got {X.shape[2]}"
        # Broadcast [N, 4, 406] - [406] works directly via numpy broadcasting
        X_scaled = (X.astype(np.float32) - self.mean_) / self.scale_
        return X_scaled.astype(np.float32)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Fit on X_train and return transformed X_train."""
        self.fit(X_train)
        return self.transform(X_train)

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("SequenceStandardScaler.fit() must be called before transform().")

    def save(self, path: pathlib.Path | str = SCALER_PATH) -> None:
        """Serialize the fitted scaler to disk."""
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info("Scaler saved: %s", path)
        # Also save human-readable stats JSON
        stats = {
            "n_features": self.n_features,
            "n_near_zero_variance": int((self.scale_ == 1.0).sum()),
            "mean_range":  [float(self.mean_.min()),  float(self.mean_.max())],
            "scale_range": [float(self.scale_.min()), float(self.scale_.max())],
            "fitted_on": "train sequences only",
        }
        with open(SCALER_STATS_PATH, "w") as f:
            json.dump(stats, f, indent=2)
        log.info("Scaler stats saved: %s", SCALER_STATS_PATH)

    @staticmethod
    def load(path: pathlib.Path | str = SCALER_PATH) -> "SequenceStandardScaler":
        """Load a previously saved scaler."""
        with open(path, "rb") as f:
            scaler = pickle.load(f)
        log.info("Scaler loaded from: %s", path)
        return scaler

    def summary(self) -> str:
        self._check_fitted()
        n_large  = int((self.scale_ > 100).sum())
        n_medium = int(((self.scale_ >= 1) & (self.scale_ <= 100)).sum())
        n_small  = int((self.scale_ < 1).sum())
        return (
            f"SequenceStandardScaler(fitted=True, features={self.n_features})\n"
            f"  mean  range : [{self.mean_.min():.3f}, {self.mean_.max():.3f}]\n"
            f"  scale range : [{self.scale_.min():.3f}, {self.scale_.max():.3f}]\n"
            f"  scale > 100 : {n_large} features\n"
            f"  scale 1-100 : {n_medium} features\n"
            f"  scale < 1   : {n_small} features"
        )
