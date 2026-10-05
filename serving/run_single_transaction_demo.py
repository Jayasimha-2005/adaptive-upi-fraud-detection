"""
run_single_transaction_demo.py
Demonstration script to execute offline inference on a single unseen transaction
and print a detailed, formatted summary to the terminal.

Usage:
    python run_single_transaction_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.offline_inference import (
    OfflineInferenceEngine,
    load_e1_test_transactions,
)


def main():
    print("=" * 65)
    print("  PHASE 1 — SINGLE TRANSACTION OFFLINE INFERENCE DEMO")
    print("=" * 65)

    # 1. Initialize inference engine
    print("\n[1/3] Initializing Offline Inference Engine ...")
    engine = OfflineInferenceEngine()

    # 2. Load 1 unseen transaction from test split
    print("[2/3] Loading 1 unseen test transaction from dataset ...")
    df_single = load_e1_test_transactions(sample_size=1, random_seed=42)

    # 3. Predict transaction
    print("[3/3] Running end-to-end inference & measuring latency ...\n")
    results_df, summary_meta = engine.predict_transaction(df_single)
    row = results_df.iloc[0]

    print("+" + "-" * 50 + "+")
    print(f"| TRANSACTION INFERENCE RESULT                    |")
    print("+" + "-" * 50 + "+")
    print(f"  Transaction ID:          {row['transaction_id']}")
    print(f"  Fraud Probability:       {row['fraud_probability']:.6f} ({row['fraud_probability']*100:.3f}%)")
    print(f"  Decision Threshold:      {engine.threshold:.6f}")
    print(f"  Final Decision:          {row['decision']}")
    print(f"  Ground Truth (Actual):   {row['actual_label']}")
    print("-" * 52)
    print("  LATENCY BREAKDOWN:")
    print(f"  - Preprocessing Time:    {row['preprocessing_time_ms']:.3f} ms")
    print(f"  - Model Prediction Time: {row['model_prediction_time_ms']:.3f} ms")
    print(f"  - Total Inference Time:  {row['total_inference_time_ms']:.3f} ms")
    print("+" + "-" * 50 + "+\n")


if __name__ == "__main__":
    main()
