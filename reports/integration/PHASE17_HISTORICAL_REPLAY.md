# Phase 17: Historical Dataset Streaming Replay Report

- **Date / Time**: `2026-10-05 21:50:29 UTC`
- **Scope**: Controlled Chronological Historical Replay & Stream-to-Serving Simulation
- **Dataset Evaluated**: `train_transaction.csv` (Authorized IEEE-CIS Fraud Benchmark)
- **Causality Invariant**: $\text{history.timestamp} < \text{current\_event.timestamp}$ strictly maintained across all stages
- **Target Invariant**: Target labels (`isFraud`, `is_fraud`) completely excluded from inference feature matrix $X$
- **Lineage Invariant**: 100% event traceability through Kafka $\to$ Flink $\to$ Bridge $\to$ Hydration $\to$ Serving $\to$ E1

---

## 1. Staged Execution Summary

| Stage | Input Events | Scored Events | Rejections | Errors | Total Time (s) | Throughput (tx/s) | Lineage Complete | Causal Valid |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Stage_100` | 100 | 100 | 0 | 0 | 8.7355 | **11.45** | 🟢 PASS | 🟢 PASS |
| `Stage_500` | 500 | 500 | 0 | 0 | 44.1492 | **11.33** | 🟢 PASS | 🟢 PASS |
| `Stage_1000` | 1000 | 1000 | 0 | 0 | 101.7474 | **9.83** | 🟢 PASS | 🟢 PASS |

---

## 2. Latency Profiles by Stage (ms)

| Stage | Mean Latency | Median (p50) | p95 Latency | p99 Latency | Min Latency | Max Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Stage_100` | 87.3431 ms | 85.5315 ms | 93.0135 ms | 120.5196 ms | 83.4882 ms | 139.7105 ms |
| `Stage_500` | 88.2867 ms | 84.4229 ms | 102.0787 ms | 116.7219 ms | 81.8251 ms | 128.986 ms |
| `Stage_1000` | 101.7331 ms | 100.6971 ms | 106.8514 ms | 129.9612 ms | 93.6506 ms | 145.058 ms |

---

## 3. Invariants & Acceptance Gates Verification

| Gate | Acceptance Criterion | Result |
| :--- | :--- | :---: |
| **P17.1** | Chronological replay ordering by `TransactionDT` | 🟢 **PASS** |
| **P17.2** | Multi-stage scalability (100 -> 500 -> 1,000 transactions) | 🟢 **PASS** |
| **P17.3** | Absolute point-in-time causality (0 causal lookahead violations) | 🟢 **PASS** |
| **P17.4** | Target isolation (0 occurrences of `isFraud` in model matrix $X$) | 🟢 **PASS** |
| **P17.5** | End-to-end lineage completeness across all stages | 🟢 **PASS** |
| **P17.6** | Zero unhandled crashes or silent event drops | 🟢 **PASS** |

---

## 4. Engineering Claim Boundary

- **PROVEN**: Controlled historical streaming simulation across partitioned Kafka, stateful dual-window Flink stream processing, bridge correlation, and online feature hydration into frozen E1 LightGBM at scale up to 1,000 transactions.
- **NOT CLAIMED**: Live production streaming on bank infrastructure; live UPI switch integration; real-world production SLA guarantees.
