"""
phase4_api/benchmarks/concurrency_benchmark.py
==============================================
Controlled Bounded Concurrency Benchmark Runner.

Part of Member 3 — API Serving Layer for Adaptive UPI / BAF Fraud Detection.
Phase 8: Performance Testing, Benchmarking & Parity Validation.

Strictly enforces:
- Bounded concurrency with ThreadPoolExecutor (no unbounded requests)
- Separate evaluation of Single Request vs Batch Request concurrency
- Latency percentile analysis and throughput scaling measurements
- Transparent reporting of host hardware constraints
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from phase4_api.benchmarks.api_client import BenchmarkApiClient, SingleRequestResult
from phase4_api.benchmarks.dataset_loader import ValidatedTransactionRecord

logger = logging.getLogger("phase4_api.benchmarks.concurrency_benchmark")


@dataclass
class ConcurrencyLevelMetrics:
    """Performance metrics for a specific tested concurrency level."""
    test_mode: str  # "single" or "batch"
    concurrency_level: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    error_rate: float
    total_duration_seconds: float
    requests_per_second: float
    transactions_per_second: float

    # Client-observed latency percentiles (ms)
    client_latency_avg_ms: float
    client_latency_p50_ms: float
    client_latency_p90_ms: float
    client_latency_p95_ms: float
    client_latency_p99_ms: float

    # Hardware notice
    hardware_disclaimer: str = (
        "Performance metrics reflect local hardware environment and concurrency boundaries. "
        "Local benchmarks should not be extrapolated directly to distributed production clusters."
    )


class ConcurrencyBenchmarkRunner:
    """Executes bounded multi-threaded load tests against the API."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout_seconds: float = 30.0) -> None:
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _calc_percentiles(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"avg": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
        arr = np.array(values, dtype=float)
        return {
            "avg": round(float(np.mean(arr)), 3),
            "p50": round(float(np.percentile(arr, 50)), 3),
            "p90": round(float(np.percentile(arr, 90)), 3),
            "p95": round(float(np.percentile(arr, 95)), 3),
            "p99": round(float(np.percentile(arr, 99)), 3),
        }

    def _worker_single(self, record: ValidatedTransactionRecord) -> SingleRequestResult:
        """Isolated worker thread execution with dedicated client session."""
        with BenchmarkApiClient(base_url=self.base_url, timeout_seconds=self.timeout_seconds) as client:
            return client.predict_single(record.transaction_id, record.features)

    def run_single_concurrency(
        self,
        records: List[ValidatedTransactionRecord],
        concurrency_levels: List[int],
    ) -> Tuple[List[ConcurrencyLevelMetrics], List[Dict[str, Any]]]:
        """Runs concurrent single-transaction requests across specified concurrency levels."""
        all_metrics: List[ConcurrencyLevelMetrics] = []
        all_details: List[Dict[str, Any]] = []

        for conc in concurrency_levels:
            logger.info("Running Single Concurrency Benchmark: concurrency=%d, total_tx=%d", conc, len(records))
            
            client_latencies: List[float] = []
            success_count = 0
            fail_count = 0

            start_time = time.perf_counter()

            # Execute with bounded ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=conc) as executor:
                future_map = {executor.submit(self._worker_single, rec): rec for rec in records}
                for future in as_completed(future_map):
                    rec = future_map[future]
                    try:
                        res: SingleRequestResult = future.result()
                        if res.is_success:
                            success_count += 1
                            client_latencies.append(res.client_latency_ms)
                        else:
                            fail_count += 1

                        all_details.append({
                            "mode": "single",
                            "concurrency_level": conc,
                            "transaction_id": rec.transaction_id,
                            "is_success": res.is_success,
                            "client_latency_ms": round(res.client_latency_ms, 3),
                            "status_code": res.status_code,
                            "error_message": res.error_message,
                        })
                    except Exception as e:
                        fail_count += 1
                        all_details.append({
                            "mode": "single",
                            "concurrency_level": conc,
                            "transaction_id": rec.transaction_id,
                            "is_success": False,
                            "client_latency_ms": 0.0,
                            "status_code": 0,
                            "error_message": str(e),
                        })

            duration = time.perf_counter() - start_time
            total_req = len(records)
            rps = round(total_req / duration, 2) if duration > 0 else 0.0
            tps = rps
            err_rate = round(fail_count / total_req, 4) if total_req > 0 else 0.0

            stats = self._calc_percentiles(client_latencies)

            metric = ConcurrencyLevelMetrics(
                test_mode="single",
                concurrency_level=conc,
                total_requests=total_req,
                successful_requests=success_count,
                failed_requests=fail_count,
                error_rate=err_rate,
                total_duration_seconds=round(duration, 3),
                requests_per_second=rps,
                transactions_per_second=tps,
                client_latency_avg_ms=stats["avg"],
                client_latency_p50_ms=stats["p50"],
                client_latency_p90_ms=stats["p90"],
                client_latency_p95_ms=stats["p95"],
                client_latency_p99_ms=stats["p99"],
            )
            all_metrics.append(metric)

        return all_metrics, all_details
