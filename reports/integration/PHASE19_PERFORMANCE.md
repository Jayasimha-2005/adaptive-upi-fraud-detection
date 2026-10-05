# Phase 19: Throughput & Latency Profiling Report

- **Date / Time**: `2026-10-05 22:05:33 UTC`
- **Scope**: Systematic Component-Level & End-to-End Latency and Throughput Profiling
- **Dataset Evaluated**: `train_transaction.csv` (Authorized IEEE-CIS Fraud Benchmark)
- **Sample Size ($N$)**: `500`
- **Execution Invariant**: Zero fabricated benchmark metrics; all measurements obtained in the local controlled environment

---

## 1. System & Component Throughput Summary

| Component Stage | Throughput (events/sec) | Context |
| :--- | :---: | :--- |
| **Kafka Ingestion** | **20,203.16** | Local partitioned in-memory broker with murmur2 hash affinity |
| **Flink Processing** | **16,477.23** | Stateful dual 5m/10m window sliding aggregation |
| **Bridge Correlation** | **20,480.22** | Point-in-time timestamp matching buffer |
| **Feature Hydration** | **1,375.0** | Point-in-time causal historical state resolution |
| **Serving Preprocessing** | **11.98** | Canonical 406 feature transformation & scaling |
| **E1 Model Inference** | **361.16** | Frozen LightGBM tree inference |
| **End-to-End System** | **11.45 tx/s** | Complete Kafka $\to$ Flink $\to$ Bridge $\to$ Hydration $\to$ E1 |

- **Successful Scoring Rate**: `11.45 scores/sec`
- **Rejection Rate**: `0.0 rejections/sec`
- **Error Rate**: `0.00%`

---

## 2. Latency Profiles by Pipeline Stage (ms)

| Stage | Mean | Median (p50) | p90 | p95 | p99 | Min | Max | Std Dev |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **kafka_ingestion** | 0.050 | 0.049 | 0.055 | 0.058 | 0.066 | 0.041 | 0.086 | 0.005 |
| **flink_processing** | 0.061 | 0.059 | 0.066 | 0.070 | 0.086 | 0.053 | 0.096 | 0.005 |
| **bridge_correlation** | 0.049 | 0.048 | 0.054 | 0.059 | 0.067 | 0.039 | 0.188 | 0.009 |
| **hydration** | 0.727 | 0.701 | 0.817 | 0.868 | 1.034 | 0.627 | 3.440 | 0.146 |
| **preprocessing** | 83.454 | 81.558 | 87.320 | 93.215 | 117.025 | 78.742 | 147.409 | 7.335 |
| **e1_inference** | 2.769 | 2.757 | 2.956 | 3.030 | 3.408 | 1.919 | 5.679 | 0.236 |
| **end_to_end** | 87.344 | 85.445 | 91.411 | 97.631 | 120.616 | 82.461 | 152.193 | 7.401 |

---

## 3. Execution Environment & Hardware Specification

| Attribute | Specification |
| :--- | :--- |
| **Operating System** | `Windows 11` |
| **Python Version** | `3.13.13` |
| **CPU Architecture** | `AMD64` |
| **Processor** | `Intel64 Family 6 Model 186 Stepping 2, GenuineIntel` |
| **LightGBM Version** | `4.7.0` |
| **Kafka Engine** | `LocalKafkaBroker (partitioned in-memory ring-buffer)` |
| **Flink Engine** | `FlinkStreamProcessor (dual 5m/10m stateful sliding windows)` |
| **Benchmarking Context** | `Measured in the local controlled environment` |

---

## 4. Engineering Claim Discipline & Boundaries

- **PROVEN**: Component-level execution boundaries, latency distribution percentiles, and multi-stage throughput under controlled in-process streaming orchestration.
- **NOT CLAIMED**: Production multi-node Kafka cluster throughput, distributed Flink cluster SLA, or live enterprise UPI banking latency.
