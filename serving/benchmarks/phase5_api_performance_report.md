# Phase 5 — FastAPI REST API Performance & Serving Benchmark Report

**Member 3 — ML Serving & Inference Infrastructure**  
**Repository**: `fraud-model-serving`  
**Model**: Approved E1 LightGBM Baseline (`models/E1/model.txt`)  
**Serving Framework**: FastAPI + Uvicorn  
**Feature Count**: `406 Processed Features`  
**Decision Threshold**: `0.616521`  

---

## 1. Executive Summary

This report measures the real-time HTTP serving performance of the **FastAPI model-serving application** under single-request, multi-concurrency, and batch workloads.

### Key Performance Results
- **Cold-Start Artifact Loading Overhead**: `29.373 ms`
- **Single-Request $P_50$ Latency**: **`85.75 ms`**
- **Single-Request $P_95$ Latency**: **`93.00 ms`**
- **Single-Request $P_99$ Latency**: **`123.92 ms`**
- **Peak Single-Endpoint API Throughput**: **`12.2 RPS`** (Requests Per Second)
- **Batch API Amortized Throughput**: **`2328.5 Transactions/sec`** (Batch Size: 500)
- **Error Rate under Load**: **`0.00%`** across all valid workloads.

---

## 2. Environment & System Specifications

| Component | Specification |
| :--- | :--- |
| **Operating System** | `Windows 11 (AMD64)` |
| **Python Version** | `3.14.5` |
| **FastAPI Version** | `0.141.1` |
| **Uvicorn Version** | `0.52.4` |
| **LightGBM Version** | `4.7.0` |
| **Pydantic Version** | `2.13.5` |

---

## 3. Cold-Start vs Warm-Start Loading Behavior

- **Startup Strategy**: Lifespan context manager (`lifespan`) loads model weights (`model.txt`) and preprocessor (`preprocessing.joblib`) **once at application startup**.
- **Cold Start Time**: `29.373 ms` (Artifact deserialization + memory allocation).
- **Per-Request Overhead**: `0.00 ms` (Model remains hot in memory across all requests).

---

## 4. Single-Request Sequential Latency Distribution (N=1,000 POST `/predict`)

Included components per HTTP request:
`Client HTTP Request ➔ FastAPI Router ➔ Pydantic Validation ➔ ServingPreprocessor ➔ 406 Feature Matrix ➔ LightGBM Predict ➔ Pydantic Response Validation ➔ JSON Serialization ➔ Client Response`

| Latency Metric | Value (ms) |
| :--- | :--- |
| **Min Latency** | `73.39 ms` |
| **Mean Latency** | `86.44 ms` |
| **Median ($P_50$)** | **`85.75 ms`** |
| **$P_95$ Latency** | **`93.00 ms`** |
| **$P_99$ Latency** | **`123.92 ms`** |
| **Max Latency** | `160.65 ms` |
| **Sequential RPS Throughput** | **`11.6 RPS`** |

---

## 5. Multi-Concurrency Load Benchmark Results

Evaluated across concurrent HTTP request streams (Concurrency Levels: 1, 5, 10, 25, 50):

| Concurrency Level | Total Requests | Successful | Error Rate (%) | $P_50$ (ms) | $P_95$ (ms) | $P_99$ (ms) | API Throughput (RPS) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `1` | `200` | `200` | `0.00%` | `91.25 ms` | `95.96 ms` | `130.47 ms` | **`10.9 RPS`** |
| `5` | `200` | `200` | `0.00%` | `401.77 ms` | `443.45 ms` | `445.30 ms` | **`12.2 RPS`** |
| `10` | `200` | `200` | `0.00%` | `811.61 ms` | `867.18 ms` | `872.52 ms` | **`12.1 RPS`** |
| `25` | `300` | `300` | `0.00%` | `2083.77 ms` | `2264.26 ms` | `2842.05 ms` | **`11.9 RPS`** |
| `50` | `300` | `300` | `0.00%` | `4209.24 ms` | `4283.44 ms` | `4293.17 ms` | **`11.8 RPS`** |

---

## 6. Batch API Performance (`POST /predict/batch`)

Benchmarked across multi-transaction payloads:

| Batch Size | Total Batch Time (ms) | Amortized Time / Tx (ms) | Amortized Throughput (RPS) | Success Status |
| :--- | :--- | :--- | :--- | :--- |
| `10` | `87.29 ms` | `8.7286 ms` | **`114.6 Tx/sec`** | `True` |
| `100` | `116.03 ms` | `1.1603 ms` | **`861.8 Tx/sec`** | `True` |
| `500` | `214.73 ms` | `0.4295 ms` | **`2328.5 Tx/sec`** | `True` |

---

## 7. Comparison: Phase 2 Offline TPS vs Phase 5 Online API RPS

> [!IMPORTANT]
> **Key Architectural Distinction**:
> - **Phase 2 (Offline Vectorized TPS)**: Evaluated raw Python/pandas DataFrame matrix throughput in memory (**`27,966 TPS`** @ 10,000 rows). Excluded network I/O, HTTP parsing, Pydantic validation, and JSON serialization.
> - **Phase 5 (Online HTTP API RPS)**: Evaluates real-world client-server REST API HTTP throughput (**`12.2 RPS`** single-request, **`2328.5 Tx/sec`** batch). Includes full HTTP/JSON serialization stack overhead.

---

## 8. Main Repository Immutability Check

- **Source Repository**: `adaptive-upi-fraud-detection`
- **Working Tree Status**: Clean and untouched.
- **Model Parameters/Weights**: 0 modifications.
