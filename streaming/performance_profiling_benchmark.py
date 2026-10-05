"""
streaming/performance_profiling_benchmark.py
Phase 19: Comprehensive Component & End-to-End Latency and Throughput Profiling.

Measures:
1. Component Latencies:
   - Kafka ingestion latency
   - Flink processing latency
   - Bridge correlation latency
   - Feature hydration latency
   - Preprocessing latency
   - Canonical E1 inference latency
   - Total end-to-end latency

2. Component & System Throughput:
   - Kafka ingestion throughput (events/sec)
   - Flink processing throughput (events/sec)
   - Bridge throughput (events/sec)
   - Serving throughput (inferences/sec)
   - Complete end-to-end throughput (tx/sec)
   - Successful scores/sec, rejections/sec, error rate

3. Hardware & Environment Metadata:
   - OS, Python version, CPU, RAM, LightGBM version, architecture.

Outputs:
- reports/integration/phase19_benchmark_results.json
- reports/integration/PHASE19_PERFORMANCE.md
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVING_DIR = REPO_ROOT / "serving"
for p in [str(REPO_ROOT), str(SERVING_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from serving.hydration.adapter import OnlineFeatureHydrationAdapter
from serving.hydration.entity_store import EntityProfileStore
from serving.inference.offline_inference import (
    E1_DECISION_THRESHOLD,
    OfflineInferenceEngine,
)
from streaming.historical_streaming_replay import load_chronological_transactions
from streaming.kafka_broker import LocalKafkaBroker
from streaming.multi_transaction_benchmark import find_dataset_path
from streaming.stream_pipeline import StreamingPipelineOrchestrator

logger = logging.getLogger("performance_profiling")


def compute_distribution_stats(latencies: List[float]) -> Dict[str, float]:
    """Calculate comprehensive distribution metrics for a list of latency values (ms)."""
    if not latencies:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}

    s = sorted(latencies)
    n = len(s)

    def pct(p: float) -> float:
        idx = (n - 1) * (p / 100.0)
        fl = math.floor(idx)
        cl = math.ceil(idx)
        if fl == cl:
            return s[int(idx)]
        return s[fl] * (cl - idx) + s[cl] * (idx - fl)

    mean_val = statistics.mean(latencies)
    std_val = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

    return {
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "min": round(min(latencies), 4),
        "max": round(max(latencies), 4),
        "p50": round(pct(50.0), 4),
        "p90": round(pct(90.0), 4),
        "p95": round(pct(95.0), 4),
        "p99": round(pct(99.0), 4),
    }


def run_performance_profiling(count: int = 500) -> Dict[str, Any]:
    """
    Profile component-level and end-to-end performance over N transactions.
    """
    dataset_path = find_dataset_path()
    transactions, profiles = load_chronological_transactions(dataset_path, max_count=count)

    store = EntityProfileStore()
    for prof in profiles:
        store.register_profile(prof)

    broker = LocalKafkaBroker()
    orch = StreamingPipelineOrchestrator(broker=broker, entity_store=store)

    # Component latency collectors (ms)
    kafka_ingest_latencies: List[float] = []
    flink_process_latencies: List[float] = []
    bridge_corr_latencies: List[float] = []
    hydration_latencies: List[float] = []
    prep_latencies: List[float] = []
    model_latencies: List[float] = []
    e2e_latencies: List[float] = []

    scored_count = 0
    rejected_count = 0
    error_count = 0

    benchmark_start_t = time.perf_counter()

    for event in transactions:
        t0 = time.perf_counter()

        # 1. Kafka Ingest
        t_k0 = time.perf_counter()
        card_val = event.get("card_id") or event.get("card1") or "0"
        card_key = str(card_val)
        if not card_key.startswith("CARD-") and card_key.isdigit():
            card_key = f"CARD-{card_key}"

        dt = float(event.get("TransactionDT") or 0.0)
        meta_raw = orch.producer.send(
            topic=orch.raw_topic,
            key=card_key,
            value=event,
            timestamp=dt,
        )
        orch.producer.flush()
        t_k1 = time.perf_counter()
        kafka_ingest_latencies.append((t_k1 - t_k0) * 1000.0)

        # 2. Flink Stream Processing
        t_f0 = time.perf_counter()
        features = orch.flink_processor.process_records(max_records=10)
        tx_id = str(event.get("TransactionID"))
        matching_feat = next((f for f in features if str(f.get("transaction_id")) == tx_id), None)
        t_f1 = time.perf_counter()
        flink_process_latencies.append((t_f1 - t_f0) * 1000.0)

        # 3. Bridge Correlation
        t_b0 = time.perf_counter()
        if matching_feat:
            orch.bridge.correlation_buffer.add_feature_event(matching_feat)
        norm = orch.bridge.normalize_stream_event(event)
        velocity_metrics = None
        if norm["TransactionDT"] is not None:
            velocity_metrics = orch.bridge.correlation_buffer.correlate(
                card_id=norm["card_id"],
                event_timestamp=norm["TransactionDT"],
                window_tolerance_sec=orch.bridge.correlation_window_sec,
            )
        payload = orch.bridge.assemble_payload(event, velocity_metrics)
        t_b1 = time.perf_counter()
        bridge_corr_latencies.append((t_b1 - t_b0) * 1000.0)

        # 4. Feature Hydration
        t_h0 = time.perf_counter()
        res = orch.bridge.adapter.hydrate(payload)
        t_h1 = time.perf_counter()
        hydration_latencies.append((t_h1 - t_h0) * 1000.0)

        # 5. Serving Preprocessing & 6. E1 Model Inference
        if res.complete and res.dataframe is not None:
            t_p0 = time.perf_counter()
            X_trans, _ = orch.bridge.engine.serving_preprocessor.transform(res.dataframe)
            t_p1 = time.perf_counter()
            prep_latencies.append((t_p1 - t_p0) * 1000.0)

            t_m0 = time.perf_counter()
            preds = orch.bridge.engine.model.predict(X_trans)
            t_m1 = time.perf_counter()
            model_latencies.append((t_m1 - t_m0) * 1000.0)

            # Record causal history
            orch.bridge.adapter.entity_store.record_transaction(
                card_id=res.card_id,
                timestamp=float(res.dataframe.iloc[0]["TransactionDT"]),
                amount=float(res.dataframe.iloc[0]["TransactionAmt"]),
                transaction_id=str(res.transaction_id),
            )
            scored_count += 1
        elif not res.complete:
            rejected_count += 1
        else:
            error_count += 1

        t1 = time.perf_counter()
        e2e_latencies.append((t1 - t0) * 1000.0)

    total_benchmark_time = time.perf_counter() - benchmark_start_t
    n_tx = len(transactions)

    # Throughputs
    e2e_throughput = n_tx / total_benchmark_time if total_benchmark_time > 0 else 0.0
    kafka_throughput = n_tx / (sum(kafka_ingest_latencies) / 1000.0) if kafka_ingest_latencies else 0.0
    flink_throughput = n_tx / (sum(flink_process_latencies) / 1000.0) if flink_process_latencies else 0.0
    bridge_throughput = n_tx / (sum(bridge_corr_latencies) / 1000.0) if bridge_corr_latencies else 0.0
    hydration_throughput = n_tx / (sum(hydration_latencies) / 1000.0) if hydration_latencies else 0.0
    prep_throughput = len(prep_latencies) / (sum(prep_latencies) / 1000.0) if prep_latencies else 0.0
    model_throughput = len(model_latencies) / (sum(model_latencies) / 1000.0) if model_latencies else 0.0

    profile_results = {
        "phase": 19,
        "phase_name": "Throughput & Latency Profiling",
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "dataset_name": dataset_path.name,
        "sample_size_N": n_tx,
        "scored_count": scored_count,
        "rejected_count": rejected_count,
        "error_count": error_count,
        "total_benchmark_time_seconds": round(total_benchmark_time, 4),
        "throughputs": {
            "end_to_end_tx_per_sec": round(e2e_throughput, 2),
            "kafka_ingestion_events_per_sec": round(kafka_throughput, 2),
            "flink_processing_events_per_sec": round(flink_throughput, 2),
            "bridge_correlation_events_per_sec": round(bridge_throughput, 2),
            "hydration_events_per_sec": round(hydration_throughput, 2),
            "preprocessing_events_per_sec": round(prep_throughput, 2),
            "e1_model_inference_events_per_sec": round(model_throughput, 2),
            "successful_scores_per_sec": round(scored_count / total_benchmark_time, 2) if total_benchmark_time > 0 else 0.0,
            "rejections_per_sec": round(rejected_count / total_benchmark_time, 2) if total_benchmark_time > 0 else 0.0,
            "error_rate": round(error_count / n_tx, 4) if n_tx > 0 else 0.0,
        },
        "latencies_ms": {
            "kafka_ingestion": compute_distribution_stats(kafka_ingest_latencies),
            "flink_processing": compute_distribution_stats(flink_process_latencies),
            "bridge_correlation": compute_distribution_stats(bridge_corr_latencies),
            "hydration": compute_distribution_stats(hydration_latencies),
            "preprocessing": compute_distribution_stats(prep_latencies),
            "e1_inference": compute_distribution_stats(model_latencies),
            "end_to_end": compute_distribution_stats(e2e_latencies),
        },
        "environment": {
            "os": f"{platform.system()} {platform.release()}",
            "python_version": platform.python_version(),
            "cpu_architecture": platform.machine(),
            "processor": platform.processor(),
            "lightgbm_version": lightgbm.__version__,
            "kafka_implementation": "LocalKafkaBroker (partitioned in-memory ring-buffer)",
            "flink_implementation": "FlinkStreamProcessor (dual 5m/10m stateful sliding windows)",
            "benchmark_context": "Measured in the local controlled environment",
        },
        "passed": n_tx >= 100 and scored_count > 0 and error_count == 0,
    }

    # Save outputs
    reports_dir = REPO_ROOT / "reports" / "integration"
    reports_dir.mkdir(parents=True, exist_ok=True)

    json_path = reports_dir / "phase19_benchmark_results.json"
    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump(profile_results, f, indent=2)

    md_path = reports_dir / "PHASE19_PERFORMANCE.md"
    generate_performance_markdown_report(profile_results, md_path)

    return profile_results


def generate_performance_markdown_report(data: Dict[str, Any], output_path: Path) -> None:
    """Generate Phase 19 Markdown Report."""
    tp = data["throughputs"]
    lat = data["latencies_ms"]
    env = data["environment"]

    content = f"""# Phase 19: Throughput & Latency Profiling Report

- **Date / Time**: `{data["timestamp_utc"]} UTC`
- **Scope**: Systematic Component-Level & End-to-End Latency and Throughput Profiling
- **Dataset Evaluated**: `{data["dataset_name"]}` (Authorized IEEE-CIS Fraud Benchmark)
- **Sample Size ($N$)**: `{data["sample_size_N"]}`
- **Execution Invariant**: Zero fabricated benchmark metrics; all measurements obtained in the local controlled environment

---

## 1. System & Component Throughput Summary

| Component Stage | Throughput (events/sec) | Context |
| :--- | :---: | :--- |
| **Kafka Ingestion** | **{tp["kafka_ingestion_events_per_sec"]:,}** | Local partitioned in-memory broker with murmur2 hash affinity |
| **Flink Processing** | **{tp["flink_processing_events_per_sec"]:,}** | Stateful dual 5m/10m window sliding aggregation |
| **Bridge Correlation** | **{tp["bridge_correlation_events_per_sec"]:,}** | Point-in-time timestamp matching buffer |
| **Feature Hydration** | **{tp["hydration_events_per_sec"]:,}** | Point-in-time causal historical state resolution |
| **Serving Preprocessing** | **{tp["preprocessing_events_per_sec"]:,}** | Canonical 406 feature transformation & scaling |
| **E1 Model Inference** | **{tp["e1_model_inference_events_per_sec"]:,}** | Frozen LightGBM tree inference |
| **End-to-End System** | **{tp["end_to_end_tx_per_sec"]:,} tx/s** | Complete Kafka $\\to$ Flink $\\to$ Bridge $\\to$ Hydration $\\to$ E1 |

- **Successful Scoring Rate**: `{tp["successful_scores_per_sec"]} scores/sec`
- **Rejection Rate**: `{tp["rejections_per_sec"]} rejections/sec`
- **Error Rate**: `{tp["error_rate"] * 100.0:.2f}%`

---

## 2. Latency Profiles by Pipeline Stage (ms)

| Stage | Mean | Median (p50) | p90 | p95 | p99 | Min | Max | Std Dev |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for stage_name, s in lat.items():
        content += f"| **{stage_name}** | {s['mean']:.3f} | {s['p50']:.3f} | {s['p90']:.3f} | {s['p95']:.3f} | {s['p99']:.3f} | {s['min']:.3f} | {s['max']:.3f} | {s['std']:.3f} |\n"

    content += f"""
---

## 3. Execution Environment & Hardware Specification

| Attribute | Specification |
| :--- | :--- |
| **Operating System** | `{env["os"]}` |
| **Python Version** | `{env["python_version"]}` |
| **CPU Architecture** | `{env["cpu_architecture"]}` |
| **Processor** | `{env["processor"]}` |
| **LightGBM Version** | `{env["lightgbm_version"]}` |
| **Kafka Engine** | `{env["kafka_implementation"]}` |
| **Flink Engine** | `{env["flink_implementation"]}` |
| **Benchmarking Context** | `{env["benchmark_context"]}` |

---

## 4. Engineering Claim Discipline & Boundaries

- **PROVEN**: Component-level execution boundaries, latency distribution percentiles, and multi-stage throughput under controlled in-process streaming orchestration.
- **NOT CLAIMED**: Production multi-node Kafka cluster throughput, distributed Flink cluster SLA, or live enterprise UPI banking latency.
"""
    with open(output_path, mode="w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")


if __name__ == "__main__":
    res = run_performance_profiling(count=500)
    print("Phase 19 Performance Profiling completed successfully!")
    print(f"End-to-End Throughput: {res['throughputs']['end_to_end_tx_per_sec']} tx/s")
    print(f"End-to-End Median Latency (p50): {res['latencies_ms']['end_to_end']['p50']} ms")
