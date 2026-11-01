"""
phase4_api/benchmarks/single_benchmark.py
=========================================
Single-Transaction Sequential Benchmark Runner.

Part of Member 3 — API Serving Layer for Adaptive UPI / BAF Fraud Detection.
Phase 8: Performance Testing, Benchmarking & Parity Validation.

Strictly enforces:
- Warm-up requests discarded from final steady-state stats
- Clear distinction between client-observed HTTP latency and server-reported execution_time_ms
- High-resolution monotonic timers
- Comprehensive percentile calculations (P50, P90, P95, P99)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from phase4_api.benchmarks.api_client import BenchmarkApiClient, SingleRequestResult
from phase4_api.benchmarks.dataset_loader import ValidatedTransactionRecord

logger = logging.getLogger("phase4_api.benchmarks.single_benchmark")


@dataclass
class SingleBenchmarkSummary:
    """Summary metrics for single transaction benchmarking."""
    total_requests: int
    warmup_requests: int
    measured_requests: int
    successful_requests: int
    failed_requests: int
    error_rate: float
    total_duration_seconds: float
    requests_per_second: float
    transactions_per_second: float

    # Client-observed HTTP Latency (ms)
    client_latency_avg_ms: float
    client_latency_median_ms: float
    client_latency_min_ms: float
    client_latency_max_ms: float
    client_latency_p50_ms: float
    client_latency_p90_ms: float
    client_latency_p95_ms: float
    client_latency_p99_ms: float

    # Server-reported Execution Time (ms)
    server_time_avg_ms: float
    server_time_median_ms: float
    server_time_min_ms: float
    server_time_max_ms: float
    server_time_p50_ms: float
    server_time_p90_ms: float
    server_time_p95_ms: float
    server_time_p99_ms: float


class SingleBenchmarkRunner:
    """Executes single-request sequential benchmarking against POST /api/v1/predict/single."""

    def __init__(self, api_client: BenchmarkApiClient) -> None:
        self.client = api_client

    @staticmethod
    def _calc_percentiles(values: List[float]) -> Dict[str, float]:
        """Calculates percentiles and central tendencies for a series of float values."""
        if not values:
            return {
                "avg": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }
        arr = np.array(values, dtype=float)
        return {
            "avg": round(float(np.mean(arr)), 3),
            "median": round(float(np.median(arr)), 3),
            "min": round(float(np.min(arr)), 3),
            "max": round(float(np.max(arr)), 3),
            "p50": round(float(np.percentile(arr, 50)), 3),
            "p90": round(float(np.percentile(arr, 90)), 3),
            "p95": round(float(np.percentile(arr, 95)), 3),
            "p99": round(float(np.percentile(arr, 99)), 3),
        }

    def run(
        self,
        records: List[ValidatedTransactionRecord],
        warmup_count: int = 10,
    ) -> Tuple[SingleBenchmarkSummary, List[Dict[str, Any]]]:
        """
        Runs single prediction benchmark.
        Discards warm-up requests from metrics.
        Returns summary dataclass and list of detailed per-request records.
        """
        if not records:
            raise ValueError("No transaction records provided for single benchmarking.")

        total_tx = len(records)
        warmup_tx = min(warmup_count, total_tx)

        logger.info(
            "Starting Single Transaction Benchmark: %d total transactions (warm-up: %d, measured: %d)",
            total_tx,
            warmup_tx,
            total_tx - warmup_tx,
        )

        # 1. Warm-up Phase
        if warmup_tx > 0:
            logger.info("Executing %d warm-up requests...", warmup_tx)
            for i in range(warmup_tx):
                rec = records[i]
                self.client.predict_single(rec.transaction_id, rec.features)

        # 2. Steady-State Measured Phase
        measured_records = records[warmup_tx:]
        if not measured_records:
            # Fall back to using all if total_tx <= warmup_count
            measured_records = records

        detailed_results: List[Dict[str, Any]] = []
        client_latencies: List[float] = []
        server_latencies: List[float] = []
        success_count = 0
        fail_count = 0

        bench_start = time.perf_counter()

        for idx, rec in enumerate(measured_records):
            res: SingleRequestResult = self.client.predict_single(rec.transaction_id, rec.features)
            
            is_success = res.is_success
            if is_success:
                success_count += 1
                client_latencies.append(res.client_latency_ms)
                if res.server_execution_time_ms is not None:
                    server_latencies.append(res.server_execution_time_ms)
            else:
                fail_count += 1

            detailed_results.append({
                "request_index": idx + 1,
                "transaction_id": res.transaction_id,
                "status_code": res.status_code,
                "is_success": is_success,
                "client_latency_ms": round(res.client_latency_ms, 3),
                "server_execution_time_ms": round(res.server_execution_time_ms, 3) if res.server_execution_time_ms else None,
                "prediction": res.prediction,
                "fraud_probability": round(res.fraud_probability, 6) if res.fraud_probability is not None else None,
                "risk_level": res.risk_level,
                "error_message": res.error_message,
            })

        duration = time.perf_counter() - bench_start
        measured_count = len(measured_records)
        rps = round(measured_count / duration, 2) if duration > 0 else 0.0
        tps = rps  # 1 transaction per request
        error_rate = round(fail_count / measured_count, 4) if measured_count > 0 else 0.0

        c_stats = self._calc_percentiles(client_latencies)
        s_stats = self._calc_percentiles(server_latencies)

        summary = SingleBenchmarkSummary(
            total_requests=total_tx,
            warmup_requests=warmup_tx,
            measured_requests=measured_count,
            successful_requests=success_count,
            failed_requests=fail_count,
            error_rate=error_rate,
            total_duration_seconds=round(duration, 3),
            requests_per_second=rps,
            transactions_per_second=tps,
            client_latency_avg_ms=c_stats["avg"],
            client_latency_median_ms=c_stats["median"],
            client_latency_min_ms=c_stats["min"],
            client_latency_max_ms=c_stats["max"],
            client_latency_p50_ms=c_stats["p50"],
            client_latency_p90_ms=c_stats["p90"],
            client_latency_p95_ms=c_stats["p95"],
            client_latency_p99_ms=c_stats["p99"],
            server_time_avg_ms=s_stats["avg"],
            server_time_median_ms=s_stats["median"],
            server_time_min_ms=s_stats["min"],
            server_time_max_ms=s_stats["max"],
            server_time_p50_ms=s_stats["p50"],
            server_time_p90_ms=s_stats["p90"],
            server_time_p95_ms=s_stats["p95"],
            server_time_p99_ms=s_stats["p99"],
        )

        logger.info(
            "Single Benchmark Complete: %d reqs in %.2fs (RPS: %.2f, Client P50: %.2fms, P95: %.2fms, P99: %.2fms)",
            measured_count,
            duration,
            rps,
            c_stats["p50"],
            c_stats["p95"],
            c_stats["p99"],
        )

        return summary, detailed_results
