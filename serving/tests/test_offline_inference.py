"""
tests/test_offline_inference.py
Automated tests for Phase 1 Offline Inference in fraud-model-serving.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Add project root and serving directory to sys.path
SERVING_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SERVING_DIR.parent
for p in [str(REPO_ROOT), str(SERVING_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import lightgbm as lgb
import numpy as np
import pandas as pd

from inference.offline_inference import (
    E1_DECISION_THRESHOLD,
    OfflineInferenceEngine,
)
from preprocessing.serving_wrapper import ServingPreprocessor
from src.features.ieee_cis_features import IEEECISPreprocessor


def test_artifact_loading():
    """Verify model, preprocessor, and feature names load cleanly."""
    engine = OfflineInferenceEngine(
        model_path="models/E1/model.txt",
        preprocessor_path="models/E1/preprocessing.joblib",
        feature_names_path="models/E1/feature_names.json",
    )
    assert engine.model is not None
    assert engine.serving_preprocessor is not None
    assert engine.threshold == E1_DECISION_THRESHOLD


def test_feature_count_and_order():
    """Verify preprocessor outputs exactly 406 features matching schema."""
    sp = ServingPreprocessor(
        preprocessor_path="models/E1/preprocessing.joblib",
        feature_names_path="models/E1/feature_names.json",
    )
    assert len(sp.expected_feature_names) == 406
    assert sp.preprocessor.feature_names == sp.expected_feature_names


def test_serving_wrapper_without_isfraud():
    """Verify ServingPreprocessor handles data without isFraud column."""
    sp = ServingPreprocessor()

    # Create dummy raw transaction matching minimum schema
    dummy_data = {
        "TransactionID": [999001],
        "TransactionDT": [14000000],
        "TransactionAmt": [100.0],
        "ProductCD": ["W"],
        "card1": [1000],
    }
    df_raw = pd.DataFrame(dummy_data)

    # Ensure 'isFraud' is NOT in df_raw
    assert "isFraud" not in df_raw.columns

    X, meta = sp.transform(df_raw)

    assert X.shape == (1, 406)
    assert "isFraud" not in X.columns
    assert "TransactionID" not in X.columns
    assert meta["transaction_ids"] == [999001]
    assert meta["actual_labels"] is None


def test_leakage_safeguards():
    """Verify isFraud and TransactionID are stripped from model feature matrix X."""
    sp = ServingPreprocessor()
    dummy_data = {
        "TransactionID": [999002],
        "isFraud": [1],
        "TransactionDT": [14000000],
        "TransactionAmt": [250.0],
        "ProductCD": ["C"],
        "card1": [2000],
    }
    df_raw = pd.DataFrame(dummy_data)

    X, meta = sp.transform(df_raw)

    assert "isFraud" not in X.columns
    assert "TransactionID" not in X.columns
    assert meta["actual_labels"] == [1]


def test_threshold_decision_logic():
    """Verify decision rule: probability >= 0.616521 -> FRAUD, else LEGIT."""
    threshold = 0.616521

    # Test cases above and below threshold
    probs = [0.0, 0.5, 0.616520, 0.616521, 0.85, 1.0]
    expected = ["LEGIT", "LEGIT", "LEGIT", "FRAUD", "FRAUD", "FRAUD"]

    for prob, exp in zip(probs, expected):
        decision = "FRAUD" if prob >= threshold else "LEGIT"
        assert decision == exp


def test_end_to_end_single_prediction():
    """Verify end-to-end inference execution on sample row."""
    engine = OfflineInferenceEngine()

    dummy_data = {
        "TransactionID": [999003],
        "TransactionDT": [15000000],
        "TransactionAmt": [50.0],
        "ProductCD": ["W"],
        "card1": [1357],
    }
    df_raw = pd.DataFrame(dummy_data)

    res_df, meta = engine.predict_transaction(df_raw)

    assert len(res_df) == 1
    row = res_df.iloc[0]
    assert row["transaction_id"] == 999003
    assert 0.0 <= row["fraud_probability"] <= 1.0
    assert row["decision"] in ("FRAUD", "LEGIT")
    assert row["preprocessing_time_ms"] >= 0.0
    assert row["model_prediction_time_ms"] >= 0.0
    assert row["total_inference_time_ms"] >= 0.0


if __name__ == "__main__":
    print("Running automated unit tests for Phase 1 Offline Inference...")
    test_artifact_loading()
    print("[PASS] test_artifact_loading")
    test_feature_count_and_order()
    print("[PASS] test_feature_count_and_order")
    test_serving_wrapper_without_isfraud()
    print("[PASS] test_serving_wrapper_without_isfraud")
    test_leakage_safeguards()
    print("[PASS] test_leakage_safeguards")
    test_threshold_decision_logic()
    print("[PASS] test_threshold_decision_logic")
    test_end_to_end_single_prediction()
    print("[PASS] test_end_to_end_single_prediction")
    print("\nALL 6 TESTS PASSED SUCCESSFULLY!")

