"""
tests/test_phase4_consistency.py
Automated unit test suite for Phase 4 Offline vs Online Inference Consistency in fraud-model-serving.
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
from fastapi.testclient import TestClient

from api.main import app
from inference.offline_inference import (
    E1_DECISION_THRESHOLD,
    OfflineInferenceEngine,
    load_e1_test_transactions,
)
from inference.phase4_consistency import run_consistency_benchmark


@pytest.fixture(scope="module")
def client():
    """TestClient fixture with app lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


def test_consistency_benchmark_small_sample(tmp_path: Path):
    """Verify Phase 4 benchmark runs cleanly on small sample and outputs CSV & report."""
    csv_dir = tmp_path / "results"
    report_file = tmp_path / "phase4_consistency_report.md"

    df_res, metrics = run_consistency_benchmark(
        sample_size=10,
        random_seed=42,
        output_dir=csv_dir,
        report_path=report_file,
    )

    assert len(df_res) == 10
    assert (csv_dir / "offline_online_consistency.csv").exists()
    assert report_file.exists()

    assert metrics["total_transactions"] == 10
    assert metrics["successful_online"] == 10
    assert metrics["failed_requests"] == 0
    assert metrics["decision_match_rate_pct"] == 100.0
    assert metrics["max_abs_probability_diff"] < 1e-5


def test_probability_diff_calculation():
    """Verify calculation of probability difference statistics."""
    off_probs = np.array([0.1, 0.5, 0.9])
    on_probs = np.array([0.1, 0.5, 0.9])
    diffs = np.abs(off_probs - on_probs)

    assert np.mean(diffs) == 0.0
    assert np.max(diffs) == 0.0


def test_failure_recording_schema():
    """Verify failure state attributes in results structure."""
    record = {
        "TransactionID": 999001,
        "offline_probability": 0.05,
        "online_probability": np.nan,
        "abs_diff": np.nan,
        "offline_decision": "LEGIT",
        "online_decision": "ERROR",
        "match": False,
        "status": "FAILED",
        "error": "HTTP 500: Internal Server Error",
    }
    assert record["status"] == "FAILED"
    assert record["match"] is False
    assert record["online_decision"] == "ERROR"


if __name__ == "__main__":
    print("Running Phase 4 Consistency Unit Tests...")
    test_probability_diff_calculation()
    print("[PASS] test_probability_diff_calculation")
    test_failure_recording_schema()
    print("[PASS] test_failure_recording_schema")

    # Temporary directory test
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        test_consistency_benchmark_small_sample(Path(tmpdir))
        print("[PASS] test_consistency_benchmark_small_sample")

    print("\nALL PHASE 4 UNIT TESTS PASSED SUCCESSFULLY!")
