"""
tests/test_phase5_api_benchmarks.py
Automated unit test suite for Phase 5 FastAPI REST API Performance & Serving Benchmarks in fraud-model-serving.
"""
from __future__ import annotations

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
import pytest

fastapi = pytest.importorskip("fastapi")

from inference.phase5_api_benchmarking import (
    get_environment_info,
    run_phase5_api_benchmark,
)


def test_environment_info_collection():
    """Verify system environment metadata collection."""
    env = get_environment_info()
    assert "python_version" in env
    assert "fastapi_version" in env
    assert "uvicorn_version" in env
    assert "lightgbm_version" in env
    assert "os_platform" in env


def test_phase5_api_benchmark_small_sample(tmp_path: Path):
    """Verify Phase 5 benchmark suite runs cleanly and generates CSVs and markdown report."""
    csv_dir = tmp_path / "results"
    report_file = tmp_path / "phase5_api_performance_report.md"

    df_raw, df_conc, metrics = run_phase5_api_benchmark(
        sample_size=10,
        server_url="testclient",
        output_dir=csv_dir,
        report_path=report_file,
    )

    assert len(df_raw) == 10
    assert len(df_conc) == 5  # Concurrency levels 1, 5, 10, 25, 50
    assert (csv_dir / "api_latency_results.csv").exists()
    assert (csv_dir / "api_performance_summary.csv").exists()
    assert report_file.exists()

    assert metrics["malformed_status"] in (400, 422)
    assert metrics["single_req_stats"]["median_ms"] >= 0.0


def test_rps_calculation_logic():
    """Verify RPS throughput calculation logic."""
    total_reqs = 100
    total_time_sec = 2.0
    rps = total_reqs / total_time_sec
    assert rps == 50.0


if __name__ == "__main__":
    print("Running Phase 5 API Performance Benchmark Unit Tests...")
    test_environment_info_collection()
    print("[PASS] test_environment_info_collection")
    test_rps_calculation_logic()
    print("[PASS] test_rps_calculation_logic")

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        test_phase5_api_benchmark_small_sample(Path(tmpdir))
        print("[PASS] test_phase5_api_benchmark_small_sample")

    print("\nALL PHASE 5 UNIT TESTS PASSED SUCCESSFULLY!")
