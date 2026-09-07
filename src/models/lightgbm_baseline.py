"""
src/models/lightgbm_baseline.py
LightGBM baseline trainer for Phase 1 (Experiment E1).

DESIGN PRINCIPLES
-----------------
- Fixed random seed for reproducibility.
- Class imbalance handled via is_unbalance=True (no SMOTE in Phase 1).
- Early stopping monitored on validation set only.
- Model is never re-evaluated on test during training.
- Feature importance (gain + split count) is recorded.
- Model is serialised in native LightGBM .txt format (portable).

WHY LightGBM?
  Strong gradient-boosting baseline for tabular, heterogeneous,
  high-dimensional fraud data. Fast training at 590K rows.
  Native support for missing values. SHAP-compatible.
  Established as top performer in similar IEEE-CIS competition solutions.

WHY NOT SMOTE?
  SMOTE is not used in Phase 1 because:
  (a) temporal ordering must be preserved — SMOTE creates synthetic
      samples from random pairs, which can create impossible temporal
      mixing.
  (b) is_unbalance=True already adjusts class weights.
  (c) SMOTE effects should be evaluated in a dedicated ablation experiment.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LightGBMBaseline:
    """
    Wraps LightGBM training with early stopping, feature importance logging,
    and serialisation.
    """

    def __init__(self, cfg: dict[str, Any], random_seed: int = 42):
        self.cfg         = cfg
        self.random_seed = random_seed
        self.model: lgb.Booster | None = None
        self._feature_names: list[str] = []
        self._train_metadata: dict[str, Any] = {}

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> "LightGBMBaseline":
        """
        Train LightGBM with early stopping on validation set.

        Parameters
        ----------
        X_train, y_train : Training features and labels.
        X_val,   y_val   : Validation features and labels (for early stopping only).

        Returns
        -------
        self
        """
        m = self.cfg.get("model", {})

        params = {
            "objective":         m.get("objective",       "binary"),
            "metric":            m.get("metric",           "average_precision"),
            "boosting_type":     m.get("boosting_type",    "gbdt"),
            "learning_rate":     m.get("learning_rate",    0.05),
            "num_leaves":        m.get("num_leaves",       63),
            "max_depth":         m.get("max_depth",        -1),
            "min_child_samples": m.get("min_child_samples", 50),
            "feature_fraction":  m.get("feature_fraction",  0.8),
            "bagging_fraction":  m.get("bagging_fraction",  0.8),
            "bagging_freq":      m.get("bagging_freq",       5),
            "reg_alpha":         m.get("reg_alpha",         0.1),
            "reg_lambda":        m.get("reg_lambda",        1.0),
            "is_unbalance":      m.get("is_unbalance",     True),
            "verbose":           -1,
            "seed":              self.random_seed,
            "num_threads":       m.get("n_jobs", -1),
        }

        n_estimators        = m.get("n_estimators",         1000)
        early_stopping      = m.get("early_stopping_rounds", 50)

        logger.info("Building LightGBM datasets ...")
        dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        dval   = lgb.Dataset(X_val,   label=y_val,   reference=dtrain, free_raw_data=False)

        self._feature_names = list(X_train.columns)

        logger.info(
            "Training LightGBM: n_estimators=%d | lr=%.3f | num_leaves=%d | early_stopping=%d",
            n_estimators, params["learning_rate"], params["num_leaves"], early_stopping,
        )
        logger.info("Training set: %d rows | Fraud: %d (%.2f%%)", len(y_train),
                    int(y_train.sum()), y_train.mean() * 100)
        logger.info("Validation set: %d rows | Fraud: %d (%.2f%%)", len(y_val),
                    int(y_val.sum()), y_val.mean() * 100)

        t0 = time.time()
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping, verbose=False),
            lgb.log_evaluation(period=50),
        ]

        self.model = lgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            valid_sets=[dval],
            valid_names=["validation"],
            callbacks=callbacks,
        )
        elapsed = time.time() - t0

        best_iter = self.model.best_iteration
        best_score = self.model.best_score.get("validation", {}).get(params["metric"], None)
        logger.info(
            "Training complete in %.1fs | best_iteration=%d | best_val_%s=%.6f",
            elapsed, best_iter, params["metric"], best_score if best_score else -1,
        )

        self._train_metadata = {
            "n_estimators_max":     n_estimators,
            "best_iteration":       best_iter,
            "early_stopping_rounds": early_stopping,
            f"best_val_{params['metric']}": round(best_score, 6) if best_score else None,
            "training_time_sec":    round(elapsed, 2),
            "n_features":           len(self._feature_names),
            "params":               params,
        }

        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return fraud probabilities for rows in X."""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict(X[self._feature_names])

    # ── Feature importance ────────────────────────────────────────────────────

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Return a DataFrame with gain and split importance per feature.
        Sorted by gain (descending).
        """
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        gain  = self.model.feature_importance(importance_type="gain")
        split = self.model.feature_importance(importance_type="split")
        fi = pd.DataFrame({
            "feature":      self._feature_names,
            "gain":         gain,
            "split":        split,
        }).sort_values("gain", ascending=False).reset_index(drop=True)
        fi["gain_rank"]  = range(1, len(fi) + 1)
        return fi

    def plot_feature_importance(
        self,
        save_path: str | Path,
        top_n: int = 30,
    ) -> None:
        """Plot top-N features by gain importance."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fi = self.get_feature_importance().head(top_n)
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
        ax.barh(fi["feature"][::-1], fi["gain"][::-1])
        ax.set_xlabel("Gain Importance")
        ax.set_title(f"LightGBM Feature Importance (Top {top_n} by Gain)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("Feature importance plot saved: %s", save_path)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, model_path: str | Path, feature_path: str | Path | None = None) -> None:
        """
        Save model in LightGBM native .txt format (portable, versionless).
        Optionally save feature list as JSON.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(model_path))
        logger.info("Model saved: %s", model_path)

        if feature_path:
            feature_path = Path(feature_path)
            with open(feature_path, "w", encoding="utf-8") as f:
                json.dump({"feature_names": self._feature_names}, f, indent=2)
            logger.info("Feature list saved: %s", feature_path)

    @classmethod
    def load(cls, model_path: str | Path, feature_path: str | Path) -> "LightGBMBaseline":
        """
        Load a saved model and feature list.
        Returns an instance ready for predict_proba().
        """
        instance = cls(cfg={})
        instance.model = lgb.Booster(model_file=str(model_path))
        with open(feature_path, "r", encoding="utf-8") as f:
            instance._feature_names = json.load(f)["feature_names"]
        logger.info("Model loaded: %s", model_path)
        return instance

    @property
    def train_metadata(self) -> dict[str, Any]:
        return self._train_metadata
