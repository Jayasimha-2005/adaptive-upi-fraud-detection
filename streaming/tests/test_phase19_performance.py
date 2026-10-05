"""
streaming/tests/test_phase19_performance.py
Phase 19 Test Suite: Throughput & Latency Profiling.

Validates Acceptance Gates P19.1 – P19.6:
- P19.1: Component-level profiling (Kafka, Flink, Bridge, Hydration, Preprocessing, E1)
- P19.2: Latency percentiles computed (p50, p95, p99, min, max, mean, std)
- P19.3: Throughput metrics computed for all stages
- P19.4: Hardware & environment metadata recorded without fabrication
- P19.5: Zero negative latencies and physically valid timings
- P19.6: Local controlled environment claim boundary explicitly documented
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVING_DIR = REPO_ROOT / "serving"
for p in [str(REPO_ROOT), str(SERVING_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(scope="module")
def performance_benchmark_results():
    """Load or run Phase 19 performance profiling results."""
    json_path = REPO_ROOT / "reports" / "integration" / "phase19_benchmark_results.json"
    if not json_path.exists():
        from streaming.performance_profiling_benchmark import run_performance_profiling
        return run_performance_profiling(count=100)
    with open(json_path, mode="r", encoding="utf-8") as f:
        return json.load(f)


def test_p19_1_component_level_profiling(performance_benchmark_results):
    """P19.1: Verify component latency profiling covers all stages."""
    lat = performance_benchmark_results["latencies_ms"]
    required_stages = [
        "kafka_ingestion",
        "flink_processing",
        "bridge_correlation",
        "hydration",
        "preprocessing",
        "e1_inference",
        "end_to_end",
    ]
    for stage in required_stages:
        assert stage in lat
        assert "mean" in lat[stage]
        assert "p50" in lat[stage]
        assert "p95" in lat[stage]
        assert "p99" in lat[stage]


def test_p19_2_latency_percentiles(performance_benchmark_results):
    """P19.2: Verify percentiles follow monotonic ordering min <= p50 <= p95 <= p99 <= max."""
    lat = performance_benchmark_results["latencies_ms"]
    for stage_name, s in lat.items():
        assert s["min"] <= s["p50"] <= s["p95"] <= s["p99"] <= s["max"]
        assert s["min"] >= 0.0


def test_p19_3_throughput_metrics(performance_benchmark_results):
    """P19.3: Verify throughput metrics exist and are non-zero."""
    tp = performance_benchmark_results["throughputs"]
    assert tp["end_to_end_tx_per_sec"] > 0.0
    assert tp["kafka_ingestion_events_per_sec"] > 0.0
    assert tp["flink_processing_events_per_sec"] > 0.0
    assert tp["bridge_correlation_events_per_sec"] > 0.0
    assert tp["hydration_events_per_sec"] > 0.0
    assert tp["preprocessing_events_per_sec"] > 0.0
    assert tp["e1_model_inference_events_per_sec"] > 0.0


def test_p19_4_environment_metadata(performance_benchmark_results):
    """P19.4: Verify complete execution environment specification."""
    env = performance_benchmark_results["environment"]
    assert "os" in env and env["os"]
    assert "python_version" in env and env["python_version"]
    assert "cpu_architecture" in env and env["cpu_architecture"]
    assert "lightgbm_version" in env and env["lightgbm_version"]
    assert "kafka_implementation" in env and env["kafka_implementation"]
    assert "flink_implementation" in env and env["flink_implementation"]


def test_p19_5_zero_error_rate(performance_benchmark_results):
    """P19.5: Zero unexplained pipeline errors during profiling."""
    assert performance_benchmark_results["error_count"] == 0
    assert performance_benchmark_results["throughputs"]["error_rate"] == 0.0
    assert performance_benchmark_results["passed"] is True


def test_p19_6_claim_discipline(performance_benchmark_results):
    """P19.6: Report explicitly states 'local controlled environment'."""
    env = performance_benchmark_results["environment"]
    assert "local controlled environment" in env["benchmark_context"].lower()
