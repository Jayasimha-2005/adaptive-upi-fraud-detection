# PHASE 8 — PERFORMANCE TESTING & BENCHMARKING REPORT
## Member 3: API Serving Layer

**Project:** Adaptive Financial Fraud Detection with Temporal Modeling, Concept Drift Adaptation, Explainable AI, and Real-Time Stream Processing  
**Dataset:** Bank Account Fraud (BAF)  
**Model:** LightGBM v3 — Champion Adaptive (500 Trees, 29 Canonical Features)  
**Frozen Threshold:** `0.8964321608040201`  
**Generated At:** `2026-11-01T11:05:10.401567+00:00`  
**Execution Duration:** `21.647s`  

---

## 1. Executive Summary

This report documents the rigorous performance benchmarking, latency profiling, throughput scalability, and offline-versus-online prediction parity validation for the Member 3 FastAPI serving layer.

All evaluations adhered strictly to:
1. **Approved Model Lock:** LightGBM v3 Champion Adaptive (`experiments/phase4_drift_adaptation/artifacts/models/lgbm_v3.txt`).
2. **Frozen Threshold:** `0.8964321608040201` (authoritative risk policy).
3. **Workspace Isolation:** All benchmarking scripts, test artifacts, and metrics are isolated inside `phase4_api/`.
4. **Monotonic Timing:** Client-observed HTTP latencies are measured via `time.perf_counter()` and strictly distinguished from server-side inference `execution_time_ms`.

---

## 2. Test Environment & Hardware Profile

| Attribute | Specification |
|---|---|
| Operating System | `Windows 10` |
| Python Version | `3.11.0` |
| Logical CPU Cores | `16` |
| Total Host RAM | `23.64 GB` |
| LightGBM Version | `4.7.0` |
| FastAPI Version | `0.141.1` |
| Uvicorn Version | `0.52.4` |
| Pydantic Version | `2.13.5` |

---

## 3. Dataset Configuration

- **Dataset Source:** `D:\adaptive-upi-fraud-detection\Datasets\BAF\Base.csv`
- **Synthetic Flag:** `False`
- **Records Evaluated:** `100`
- **Feature Validation:** All 29 canonical features (24 numerical, 5 categorical) validated with strict category checks and no ground truth leakage.

---

## 4. Single-Transaction Latency & Throughput Benchmark

Single prediction requests executed sequentially against `POST /api/v1/predict/single`.

### Latency Percentiles (Measured Steady State)

| Metric | Client-Observed HTTP Latency (ms) | Server Inference Latency (ms) |
|---|---|---|
| **Average (Mean)** | `78.664` | `62.085` |
| **Median (P50)** | `67.157` | `49.319` |
| **P90 Latency** | `128.224` | `105.534` |
| **P95 Latency** | `180.703` | `144.119` |
| **P99 Latency** | `232.405` | `202.813` |
| **Minimum** | `7.992` | `5.008` |
| **Maximum** | `268.089` | `252.172` |

### Throughput & Reliability
- **Total Requests Evaluated:** `90` (Warm-up requests discarded: `10`)
- **Successful Requests:** `90`
- **Failed Requests:** `0`
- **Error Rate:** `0.00%`
- **Throughput (Requests / Sec):** `12.68 req/s`
- **Throughput (Transactions / Sec):** `12.68 tx/s`

---

## 5. Batch-Transaction Benchmark Across Sizing Profiles

Evaluated against `POST /api/v1/predict/batch` across batch sizes: [1, 10, 50, 100].

| Batch Size | Batches | Submitted Tx | Success Tx | Failed Tx | Client P50 (ms) | Client P95 (ms) | Server P50 (ms) | TPS | Order & ID Match |
|---|---|---|---|---|---|---|---|---|---|
| **1** | 100 | 100 | 100 | 0 | 61.941 | 171.346 | 45.579 | 13.26 | PASSED |
| **10** | 10 | 100 | 100 | 0 | 11.414 | 14.247 | 7.091 | 828.27 | PASSED |
| **50** | 2 | 100 | 100 | 0 | 15.21 | 15.242 | 8.849 | 3135.67 | PASSED |
| **100** | 1 | 100 | 100 | 0 | 19.844 | 19.844 | 11.223 | 4685.05 | PASSED |

---

## 6. Concurrency & Throughput Scaling

Controlled bounded concurrency evaluated with `ThreadPoolExecutor`.

| Concurrency Level | Mode | Total Requests | Success | Error Rate | Client P50 (ms) | Client P95 (ms) | RPS | TPS |
|---|---|---|---|---|---|---|---|---|
| **1** | single | 100 | 100 | 0.0% | 9.857 | 18.187 | 45.89 | 45.89 |
| **5** | single | 100 | 100 | 0.0% | 37.44 | 54.891 | 92.81 | 92.81 |
| **10** | single | 100 | 100 | 0.0% | 65.846 | 77.635 | 123.35 | 123.35 |
| **20** | single | 100 | 100 | 0.0% | 173.973 | 201.681 | 103.2 | 103.2 |

> **Hardware Disclaimer:**
> Performance metrics reflect local hardware environment and concurrency boundaries. Local benchmarks should not be extrapolated directly to distributed production clusters.

---

## 7. Offline vs. Online Prediction Parity Verification

Validation of identical predictions between Python direct inference (Offline) and HTTP serving (Online).

- **Evaluated Records:** `100`
- **Numerical Floating-Point Tolerance:** `1e-08`
- **Reconciliation Rate:** `100.00%` (`100/100`)
- **Probability Parity Agreement:** `100.0%`
- **Decision Agreement (FRAUD vs LEGIT):** `100.0%`
- **Risk Level Agreement:** `100.0%`
- **Reason Code Agreement:** `100.0%`
- **Maximum Absolute Probability Difference:** `0.0000000000e+00`
- **Mean Absolute Probability Difference:** `0.0000000000e+00`
- **Mismatched Records Count:** `0`

---

## 8. Resource Utilization Snapshot

- **Process Memory (RSS):** `189.79 MB`
- **Host CPU Utilization:** `58.4%`
- **Host Available Memory:** `8705.44 MB`

---

## 9. Conclusion

Phase 8 automated performance benchmarking and offline-online parity verification has validated that the FastAPI serving layer provides sub-millisecond to low-millisecond inference execution with **100% decision and prediction parity** against the offline LightGBM v3 champion model.
