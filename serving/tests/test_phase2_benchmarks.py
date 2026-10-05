"""
tests/test_phase2_benchmarks.py
Automated tests for Phase 2 Performance Benchmarking in fraud-model-serving.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root and serving directory to sys.path
SERVING_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SERVING_DIR.parent
for p in [str(REPO_ROOT), str(SERVING_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd

from inference.phase2_benchmarking import Phase2BenchmarkRunner
from preprocessing.serving_wrapper import ServingPreprocessor


def test_cold_start_measurement():
    """Verify cold-start measurement logic records load times."""
    runner = Phase2BenchmarkRunner()
    cold_start_ms = runner.measure_cold_start()

    assert cold_start_ms > 0.0
    assert runner.cold_start_prep_ms > 0.0
    assert runner.cold_start_model_ms > 0.0
    assert runner.serving_preprocessor is not None
    assert runner.model is not None


def test_percentile_calculation_logic():
    """Verify percentile calculation math for P50, P95, P99."""
    latencies = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))

    assert p50 == 5.5
    assert p95 > p50
    assert p99 >= p95


def test_throughput_tps_calculation():
    """Verify throughput (TPS) calculation formula."""
    n_transactions = 1000
    duration_sec = 0.5

    tps = n_transactions / duration_sec
    assert tps == 2000.0


def test_phase2_vectorized_batch_execution():
    """Verify Phase 2 vectorized batch benchmark on dummy raw transactions."""
    runner = Phase2BenchmarkRunner()
    runner.measure_cold_start()

    # Create 5 dummy raw transactions matching minimum schema
    dummy_rows = []
    for i in range(5):
        dummy_rows.append({
            "TransactionID": 990000 + i,
            "TransactionDT": 14000000 + i * 100,
            "TransactionAmt": 50.0 + i * 10,
            "ProductCD": "W",
            "card1": 1000 + i,
        })
    df_raw = pd.DataFrame(dummy_rows)

    df_records, summary = runner.run_vectorized_batch_benchmark(df_raw)

    assert len(df_records) == 5
    assert summary["success_count"] == 5
    assert summary["fail_count"] == 0
    assert summary["throughput_tps"] > 0.0
    assert summary["prep_mean_ms"] >= 0.0
    assert summary["model_mean_ms"] >= 0.0
    assert summary["total_mean_ms"] >= 0.0


def test_phase2_sequential_single_execution():
    """Verify Phase 2 sequential single-transaction benchmark on dummy raw transactions."""
    runner = Phase2BenchmarkRunner()
    runner.measure_cold_start()

    dummy_rows = []
    for i in range(3):
        dummy_rows.append({
            "TransactionID": 991000 + i,
            "TransactionDT": 14000000 + i * 100,
            "TransactionAmt": 75.0,
            "ProductCD": "C",
            "card1": 2000,
        })
    df_raw = pd.DataFrame(dummy_rows)

    df_records, summary = runner.run_sequential_single_benchmark(df_raw)

    assert len(df_records) == 3
    assert summary["success_count"] == 3
    assert summary["fail_count"] == 0
    assert summary["prep_p50_ms"] >= 0.0
    assert summary["model_p50_ms"] >= 0.0
    assert summary["total_p50_ms"] >= 0.0
    assert summary["total_p95_ms"] >= summary["total_p50_ms"]
    assert summary["total_p99_ms"] >= summary["total_p95_ms"]


if __name__ == "__main__":
    print("Running Phase 2 Performance Benchmarking Unit Tests...")
    test_cold_start_measurement()
    print("[PASS] test_cold_start_measurement")
    test_percentile_calculation_logic()
    print("[PASS] test_percentile_calculation_logic")
    test_throughput_tps_calculation()
    print("[PASS] test_throughput_tps_calculation")
    test_phase2_vectorized_batch_execution()
    print("[PASS] test_phase2_vectorized_batch_execution")
    test_phase2_sequential_single_execution()
    print("[PASS] test_phase2_sequential_single_execution")
    print("\nALL PHASE 2 UNIT TESTS PASSED SUCCESSFULLY!")
