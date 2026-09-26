"""
model.py — LightGBM Model Training for Phase 4
================================================
Implements the locked LightGBM configuration from Protocol v1.1.

LOCKED CONFIGURATION (cannot change during Phase 4):
    n_estimators     = 500
    learning_rate    = 0.05
    max_depth        = 6
    num_leaves       = 63
    min_child_samples = 50
    subsample        = 0.8
    colsample_bytree = 0.8
    class_weight     = "balanced"   # NOT scale_pos_weight
    random_state     = 42

CATEGORICAL: LightGBM native (categorical_feature parameter).
             NO OrdinalEncoder / LabelEncoder used.
"""
from __future__ import annotations
import hashlib
import json
import pathlib
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from data_loader import CATEGORICAL_COLS, FEATURE_COLS, TARGET_COL

# ---------------------------------------------------------------------------
# Locked hyperparameter configuration — DO NOT modify during Phase 4
# ---------------------------------------------------------------------------
LOCKED_PARAMS: dict = {
    "n_estimators":      500,
    "learning_rate":     0.05,
    "max_depth":         6,
    "num_leaves":        63,
    "min_child_samples": 50,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "class_weight":      "balanced",
    "random_state":      42,
    "verbose":           -1,
}


def _verify_config(params: dict) -> None:
    """Assert that configuration exactly matches locked values."""
    for key, expected in LOCKED_PARAMS.items():
        actual = params.get(key)
        assert actual == expected, (
            f"Config violation: {key} = {actual!r} (expected {expected!r}). "
            "Hyperparameters are frozen in Phase 4."
        )
    assert "scale_pos_weight" not in params, (
        "scale_pos_weight is PROHIBITED. Use class_weight='balanced'."
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    version_label: str,
    training_months: list[int],
    output_dir: str | pathlib.Path | None = None,
) -> dict:
    """
    Train a single LightGBM model using the locked configuration.

    Parameters
    ----------
    X_train        : Feature matrix (must contain exactly FEATURE_COLS)
    y_train        : Target vector
    version_label  : 'v1', 'v2', or 'v3'
    training_months: List of months in training data (for metadata)
    output_dir     : Where to save the model artifact (None = don't save)

    Returns
    -------
    dict with keys: model, params, version_label, training_months,
                    n_train, n_fraud, artifact_path (if saved), model_hash
    """
    # -- Feature integrity checks (mandatory) --
    assert list(X_train.columns) == FEATURE_COLS, (
        f"Feature mismatch. Expected {FEATURE_COLS}, got {list(X_train.columns)}"
    )
    assert TARGET_COL not in X_train.columns, "Target leakage: fraud_bool in X"
    assert "month" not in X_train.columns,    "Temporal leakage: month in X"
    assert "device_fraud_count" not in X_train.columns, "Excluded col in X"
    assert len(X_train) >= 50000, f"Training set too small: {len(X_train)}"

    # -- Config verification --
    params = dict(LOCKED_PARAMS)
    _verify_config(params)

    # -- Convert categoricals to pandas Categorical if not already --
    for col in CATEGORICAL_COLS:
        if not hasattr(X_train[col], "cat"):
            X_train = X_train.copy()
            X_train[col] = pd.Categorical(X_train[col])

    # -- Train --
    model = lgb.LGBMClassifier(**params)
    t0 = time.time()
    model.fit(
        X_train, y_train,
        categorical_feature=CATEGORICAL_COLS,
    )
    elapsed = time.time() - t0

    n_train = len(X_train)
    n_fraud = int(y_train.sum())
    fraud_rate = n_fraud / n_train

    result = {
        "model":           model,
        "params":          params,
        "version_label":   version_label,
        "training_months": training_months,
        "n_train":         n_train,
        "n_fraud":         n_fraud,
        "fraud_rate":      fraud_rate,
        "training_time_s": elapsed,
        "lgb_version":     lgb.__version__,
        "artifact_path":   None,
        "model_hash":      None,
    }

    # -- Save if requested --
    if output_dir is not None:
        out = pathlib.Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        model_path = out / f"lgbm_{version_label}.txt"
        model.booster_.save_model(str(model_path))

        # Compute hash of saved model
        h = hashlib.sha256()
        with open(model_path, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
        model_hash = h.hexdigest()

        # Save metadata alongside
        meta = {
            "version_label":   version_label,
            "training_months": training_months,
            "n_train":         n_train,
            "n_fraud":         n_fraud,
            "fraud_rate":      fraud_rate,
            "training_time_s": elapsed,
            "params":          params,
            "lgb_version":     lgb.__version__,
            "model_hash":      model_hash,
            "model_file":      str(model_path),
        }
        meta_path = out / f"lgbm_{version_label}_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        result["artifact_path"] = str(model_path)
        result["model_hash"]    = model_hash

        print(f"  Model {version_label} saved -> {model_path} (SHA256: {model_hash[:16]})")
        print(f"  Training: {n_train:,} rows | {n_fraud:,} fraud ({fraud_rate:.4%}) | {elapsed:.1f}s")

    return result


def predict_proba(model_result: dict, X: pd.DataFrame) -> np.ndarray:
    """
    Generate fraud probability predictions.
    Returns 1-D array of P(fraud=1).
    """
    model = model_result["model"]
    # Ensure categoricals
    for col in CATEGORICAL_COLS:
        if col in X.columns and not hasattr(X[col], "cat"):
            X = X.copy()
            X[col] = pd.Categorical(X[col])
    proba = model.predict_proba(X)[:, 1]
    return proba
