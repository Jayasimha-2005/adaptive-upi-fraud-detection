"""
phase4_api/benchmarks/batch_benchmark.py
========================================
Batch Inference Endpoint Benchmark Runner.

Part of Member 3 — API Serving Layer for Adaptive UPI / BAF Fraud Detection.
Phase 8: Performance Testing, Benchmarking & Parity Validation.

Strictly verifies:
- Output count matches input count exactly
- Transaction IDs and ordering are strictly preserved
- Accurate tracking of partial failures vs full failures
- Latency percentiles (P50, P90, P95, P99) and TPS throughput
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from phase4_api.benchmarks.api_client import BatchRequestResult, BenchmarkApiClient
from phase4_api.benchmarks.dataset_loader import ValidatedTransactionRecord

logger = logging.getLogger("phase4_api.benchmarks.batch_benchmark")


@dataclass
class BatchSizeMetrics:
    """Summary metrics for a specific tested batch size."""
    batch_size: int
    batch_requests_count: int
    total_transactions_submitted: int
    successful_transactions: int
    failed_transactions: int
    error_rate: float
    total_duration_seconds: float
    requests_per_second: float
    transactions_per_second: float

    # Client-observed batch latency (ms)
    client_latency_avg_ms: float
    client_latency_p50_ms: float
    client_latency_p90_ms: float
    client_latency_p95_ms: float
    client_latency_p99_ms: float

    # Server execution time (ms)
    server_time_avg_ms: float
    server_time_p50_ms: float
    server_time_p95_ms: float
    server_time_p99_ms: float

    # Verification checks
    count_parity_verified: bool
    order_parity_verified: bool
    id_preservation_verified: bool


class BatchBenchmarkRunner:
    """Benchmarks the batch prediction endpoint across multiple batch sizes."""

    def __init__(self, api_client: BenchmarkApiClient) -> None:
        self.client = api_client

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

    def run_for_batch_size(
        self,
        records: List[ValidatedTransactionRecord],
        batch_size: int,
    ) -> Tuple[BatchSizeMetrics, List[Dict[str, Any]]]:
        """
        Executes batch requests for chunks of `batch_size`.
        Validates order, count, and ID preservation.
        """
        if not records:
            raise ValueError("No records provided for batch benchmarking.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        # Chunk records into batches
        batches: List[List[ValidatedTransactionRecord]] = [
            records[i : i + batch_size] for i in range(0, len(records), batch_size)
        ]

        logger.info(
            "Running Batch Benchmark: batch_size=%d across %d batches (%d total records)",
            batch_size,
            len(batches),
            len(records),
        )

        client_latencies: List[float] = []
        server_latencies: List[float] = []
        total_submitted = 0
        total_success = 0
        total_failed = 0
        detailed_rows: List[Dict[str, Any]] = []

        all_counts_match = True
        all_orders_match = True
        all_ids_match = True

        start_wall = time.perf_counter()

        for b_idx, batch_chunk in enumerate(batches):
            # Prepare payload preserving transaction_id
            payload_chunk = []
            for rec in batch_chunk:
                tx_dict = dict(rec.features)
                tx_dict["transaction_id"] = rec.transaction_id
                payload_chunk.append(tx_dict)

            res: BatchRequestResult = self.client.predict_batch(payload_chunk)

            total_submitted += len(batch_chunk)

            if res.is_success:
                client_latencies.append(res.client_latency_ms)
                if res.server_execution_time_ms is not None:
                    server_latencies.append(res.server_execution_time_ms)
                total_success += res.successful_transactions
                total_failed += res.failed_transactions

                # Verify count parity
                if len(res.records) != len(batch_chunk):
                    all_counts_match = False

                # Verify ID and order preservation
                for rec_i, expected_rec in enumerate(batch_chunk):
                    if rec_i < len(res.records):
                        actual_rec = res.records[rec_i]
                        if actual_rec.input_index != rec_i:
                            all_orders_match = False
                        if actual_rec.transaction_id != expected_rec.transaction_id:
                            all_ids_match = False
            else:
                total_failed += len(batch_chunk)
                all_counts_match = False

            detailed_rows.append({
                "batch_index": b_idx + 1,
                "batch_size": len(batch_chunk),
                "is_success": res.is_success,
                "status_code": res.status_code,
                "client_latency_ms": round(res.client_latency_ms, 3),
                "server_execution_time_ms": round(res.server_execution_time_ms, 3) if res.server_execution_time_ms else None,
                "successful_transactions": res.successful_transactions,
                "failed_transactions": res.failed_transactions,
                "error_message": res.error_message,
            })

        duration = time.perf_counter() - start_wall
        rps = round(len(batches) / duration, 2) if duration > 0 else 0.0
        tps = round(total_submitted / duration, 2) if duration > 0 else 0.0
        error_rate = round(total_failed / total_submitted, 4) if total_submitted > 0 else 0.0

        c_stats = self._calc_percentiles(client_latencies)
        s_stats = self._calc_percentiles(server_latencies)

        summary = BatchSizeMetrics(
            batch_size=batch_size,
            batch_requests_count=len(batches),
            total_transactions_submitted=total_submitted,
            successful_transactions=total_success,
            failed_transactions=total_failed,
            error_rate=error_rate,
            total_duration_seconds=round(duration, 3),
            requests_per_second=rps,
            transactions_per_second=tps,
            client_latency_avg_ms=c_stats["avg"],
            client_latency_p50_ms=c_stats["p50"],
            client_latency_p90_ms=c_stats["p90"],
            client_latency_p95_ms=c_stats["p95"],
            client_latency_p99_ms=c_stats["p99"],
            server_time_avg_ms=s_stats["avg"],
            server_time_p50_ms=s_stats["p50"],
            server_time_p95_ms=s_stats["p95"],
            server_time_p99_ms=s_stats["p99"],
            count_parity_verified=all_counts_match,
            order_parity_verified=all_orders_match,
            id_preservation_verified=all_ids_match,
        )

        return summary, detailed_rows

    def run_all(
        self,
        records: List[ValidatedTransactionRecord],
        batch_sizes: List[int],
    ) -> Tuple[List[BatchSizeMetrics], List[Dict[str, Any]]]:
        """Runs the benchmark across all configured batch sizes."""
        all_metrics: List[BatchSizeMetrics] = []
        all_details: List[Dict[str, Any]] = []

        for bs in batch_sizes:
            m, det = self.run_for_batch_size(records, batch_size=bs)
            all_metrics.append(m)
            all_details.extend(det)

        return all_metrics, all_details
