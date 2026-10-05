# Fraud Model Serving — Member 3 ML Serving

[![Phase 1 — Offline Inference](https://img.shields.io/badge/Phase%201-Offline%20Inference%20Passed-brightgreen.svg)]()
[![Phase 2 — Performance Benchmarking](https://img.shields.io/badge/Phase%202-Benchmarking%20Passed-brightgreen.svg)]()
[![Phase 3 — FastAPI Model Serving](https://img.shields.io/badge/Phase%203-FastAPI%20Serving%20Passed-brightgreen.svg)]()
[![Phase 4 — Offline vs Online Parity](https://img.shields.io/badge/Phase%204-Parity%20Passed-brightgreen.svg)]()
[![Phase 5 — API Performance Benchmark](https://img.shields.io/badge/Phase%205-API%20Benchmark%20Passed-brightgreen.svg)]()
[![Phase 6 — Failure & Error Handling](https://img.shields.io/badge/Phase%206-Error%20Handling%20Passed-brightgreen.svg)]()
[![Phase 7 — Docker & Deployment](https://img.shields.io/badge/Phase%207-Docker%20Deployment%20Verified-brightgreen.svg)]()
[![Phase 8 — Monitoring & Model Versioning](https://img.shields.io/badge/Phase%208-Monitoring%20Verified-brightgreen.svg)]()
[![Docker](https://img.shields.io/badge/Docker-29.8.0-blue.svg)]()
[![Image](https://img.shields.io/badge/Image-fraud--model--serving%3Ae1-orange.svg)]()
[![Model--E1--LightGBM](https://img.shields.io/badge/Model-E1%20LightGBM-blue.svg)]()
[![Features--406](https://img.shields.io/badge/Features-406%20Final%20Processed-orange.svg)]()
[![Threshold--0.616521](https://img.shields.io/badge/Threshold-0.616521-purple.svg)]()

This repository contains the **ML Serving & Online/Offline Inference Infrastructure** for Member 3 of our financial fraud detection research project.

---

> ### 📦 Milestone 5 Certified Handoff Brief for Member 3 (Hadassah Kiran)
> 
> **Welcome, Hadassah Kiran!** All core algorithmic serving, feature hydration, causal leakage protection, and streaming pipeline contracts are **100% complete and certified (`270/270 PASS`, `0 FAIL`)**.
> 
> Detailed handoff documentation is published at: **[`docs/HANDOFF_MEMBER3_HADASSAH_KIRAN.md`](file:///c:/Users/Harini/Documents/GitHub/Jayasimha-github/adaptive-upi-fraud-detection/docs/HANDOFF_MEMBER3_HADASSAH_KIRAN.md)**.
> 
> #### Your Next Steps (Deployment & Containerization):
> 1. **Install dependencies**: `pip install -r serving/requirements.txt`
> 2. **Run live FastAPI server**: `cd serving && uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload`
> 3. **Verify endpoints**: Check `http://localhost:8000/docs`, `/health`, `/predict`, and `/metrics`.
> 4. **Run API & Monitoring tests**: `pytest serving/tests/test_api.py serving/tests/test_phase8_monitoring.py -v`
> 5. **Build & run Docker image**: `docker build -t adaptive-fraud-serving:v1 -f serving/Dockerfile .` and `docker run -d -p 8000:8000 --name fraud-serving-api adaptive-fraud-serving:v1`
> 6. **Run Docker verification tests**: `pytest serving/tests/test_phase7_docker.py -v`

---

## 1. Research Question & Scope

> **Research Question:**  
> *"How can fraud detection models be served with low latency via REST API while maintaining strict prediction consistency between offline training and online inference?"*

### Member 3 Responsibilities
- **Phase 1**: Offline Inference & Serving Wrapper (**COMPLETED & PASSED**)
- **Phase 2**: Offline Performance Benchmarking (**COMPLETED & PASSED**)
- **Phase 3**: FastAPI REST Model Serving (**COMPLETED & PASSED**)
- **Phase 4**: Offline vs Online Inference Consistency (**COMPLETED & PASSED**)
- **Phase 5**: FastAPI API Performance & Serving Benchmark (**COMPLETED & PASSED**)
- **Phase 6**: Failure & Error Handling (**COMPLETED & PASSED**)
- **Phase 7**: Docker & Deployment (**COMPLETED & VERIFIED**)
- **Phase 8**: Monitoring & Model Versioning (**COMPLETED & VERIFIED**)
- Model Artifact Serialization & Portability
- Preprocessing Consistency Safeguards
- Feature Validation & Zero Target / ID Leakage Safeguards
- Cold Start & Latency Percentiles ($P_{50}, P_{95}, P_{99}$)
- Real-Time Endpoint Verification & Online/Offline Parity across 1,000 Transactions
- Live REST API Concurrency Scaling (RPS) & Batch Serving Throughput
- Robust Error Handling, Input Validation, & Zero Information Leakage Safeguards

---

## 2. Model Provenance & Immutability

The E1 LightGBM baseline model was trained and approved in the main research repository:

- **Source Repository**: `adaptive-upi-fraud-detection`
- **Model Status**: **Immutable Approved Artifact**
- **Member 3 Usage**: Model Serving and Inference only (No retraining, no hyperparameter modifications, no feature alterations, no threshold changes).

### E1 Artifacts Location
Artifacts copied locally to `models/E1/`:
- `models/E1/model.txt` — Approved LightGBM booster text model (1,000 trees).
- `models/E1/preprocessing.joblib` — Fitted `IEEECISPreprocessor` pipeline.
- `models/E1/feature_names.json` — Approved schema containing 406 feature names in exact order.

---

## 3. Phase 4 Offline vs Online Consistency Results

Evaluated across **1,000 real unseen test transactions** sampled from the IEEE-CIS test split (`TransactionDT > 13,392,000` / Day 155+):

| Metric | Value | Result / Status |
| :--- | :--- | :--- |
| **Total Evaluated Transactions** | `1,000` | Complete |
| **Successful Predictions** | `1,000` | 100% Success |
| **Failed Requests** | `0` | Zero Failures |
| **Matching Decisions** | `1,000` | **100% Match** |
| **Mismatching Decisions** | `0` | Zero Mismatches |
| **Classification Match Rate** | **`100.0000%`** | **PASS** |
| **Mean Abs Probability Diff** | `0.00000000` | Exact Parity |
| **Median Abs Probability Diff** | `0.00000000` | Exact Parity |
| **P95 Abs Probability Diff** | `0.00000000` | Exact Parity |
| **P99 Abs Probability Diff** | `0.00000000` | Exact Parity |
| **Max Abs Probability Diff** | `0.00000000` | Exact Parity |

---

## 4. Phase 5 FastAPI REST API Performance Benchmark Results

Evaluated against live FastAPI server (`http://127.0.0.1:8000`) across single-request, multi-concurrency load, and batch endpoints:

### Cold-Start Overhead
- **Artifact Load Time**: `29.37 ms` (Lifespan model & preprocessor loading)

### Single-Request Latency Distribution (N=1,000 requests to `POST /predict`)
| Metric | Value (ms) |
| :--- | :--- |
| **Min Latency** | `73.39 ms` |
| **Mean Latency** | `86.44 ms` |
| **Median ($P_{50}$)** | **`85.75 ms`** |
| **$P_{95}$ Latency** | **`93.00 ms`** |
| **$P_{99}$ Latency** | **`123.92 ms`** |
| **Max Latency** | `160.65 ms` |
| **Single-Request API Throughput** | **`12.2 RPS`** |

### Multi-Concurrency Load & Batch Serving Performance
| Concurrency / Payload | Workload Type | API Throughput | $P_{50}$ Latency | Error Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Concurrency Level 1** | Single Request Stream | `10.9 RPS` | `91.25 ms` | `0.00%` |
| **Concurrency Level 5** | Multi-Stream Concurrent | `12.2 RPS` | `401.77 ms` | `0.00%` |
| **Batch Size 10** | `POST /predict/batch` | `114.6 Tx/sec` | `87.29 ms / batch` | `0.00%` |
| **Batch Size 100** | `POST /predict/batch` | `861.8 Tx/sec` | `116.03 ms / batch` | `0.00%` |
| **Batch Size 500** | `POST /predict/batch` | **`2328.5 Tx/sec`** | `214.73 ms / batch` | `0.00%` |

---

## 5. API Endpoints

### 1. `GET /health`
Returns system health, model load status, feature count, and threshold.

**Sample Request**:
```bash
curl -X GET http://127.0.0.1:8000/health
```

### 2. `POST /predict`
Predicts fraud probability and decision for a single transaction payload.

**Sample Request (Real Test Transaction 3544193)**:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"TransactionID": 3544193, "TransactionDT": 14757390, "TransactionAmt": 100.0, "ProductCD": "W"}'
```

**Sample Response**:
```json
{
  "transaction_id": "3544193",
  "fraud_probability": 0.003993,
  "decision": "LEGIT",
  "actual_label": "LEGIT",
  "preprocessing_time_ms": 53.497,
  "model_prediction_time_ms": 2.377,
  "total_inference_time_ms": 55.874
}
```

### 3. `POST /predict/batch`
Batch prediction endpoint for up to 1,000 transactions.

---

## 6. Project Structure

```text
fraud-model-serving/
├── api/
│   ├── __init__.py
│   ├── main.py                           # FastAPI serving application & endpoints
│   └── schemas.py                        # Pydantic request/response schemas
├── models/
│   └── E1/
│       ├── model.txt                     # Approved LightGBM model
│       ├── preprocessing.joblib           # Fitted preprocessor pipeline
│       └── feature_names.json             # 406 feature names schema
├── preprocessing/
│   ├── serving_wrapper.py                # ServingPreprocessor wrapper
│   └── __init__.py
├── src/
│   └── features/
│       ├── ieee_cis_features.py           # IEEECISPreprocessor definition
│       └── __init__.py
├── inference/
│   ├── offline_inference.py              # Phase 1 offline inference engine
│   ├── phase2_benchmarking.py            # Phase 2 performance benchmarking engine
│   ├── phase4_consistency.py             # Phase 4 offline vs online consistency engine
│   ├── phase5_api_benchmarking.py        # Phase 5 FastAPI REST performance benchmark engine
│   └── __init__.py
├── benchmarks/
│   ├── results/
│   │   ├── raw_latency_results.csv        # Phase 2 transaction-level timing records
│   │   ├── benchmark_summary.csv          # Phase 2 aggregate benchmark summary
│   │   ├── offline_online_consistency.csv # Phase 4 consistency transaction log
│   │   ├── api_latency_results.csv        # Phase 5 HTTP request timing records
│   │   ├── api_performance_summary.csv    # Phase 5 concurrency summary
│   │   └── phase6_error_results.csv       # Phase 6 failure & error test log
│   ├── offline_inference_results.csv      # Phase 1 transaction results
│   ├── offline_inference_report.md       # Phase 1 report
│   ├── phase2_benchmark_report.md        # Phase 2 report
│   ├── phase4_consistency_report.md       # Phase 4 report
│   ├── phase5_api_performance_report.md  # Phase 5 API performance report
│   └── phase6_error_handling_report.md    # Phase 6 failure & error handling report
├── tests/
│   ├── test_offline_inference.py         # Phase 1 unit test suite
│   ├── test_phase2_benchmarks.py         # Phase 2 unit test suite
│   ├── test_api.py                       # Phase 3 FastAPI serving unit test suite
│   ├── test_phase4_consistency.py        # Phase 4 consistency unit test suite
│   ├── test_phase5_api_benchmarks.py     # Phase 5 API benchmark unit test suite
│   ├── test_phase6_error_handling.py     # Phase 6 failure & error handling unit test suite
│   ├── test_phase7_docker.py             # Phase 7 Docker & deployment unit test suite
│   └── __init__.py
├── Dockerfile                            # Docker image build configuration
├── .dockerignore                         # Files excluded from Docker build context
├── requirements.txt                      # Python runtime dependencies
├── run_single_transaction_demo.py        # Interactive single transaction demo CLI
└── README.md                             # Repository documentation
```

---

## 7. How to Run

### Start FastAPI Live Server
```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### Run Phase 5 Live API Performance Benchmark
```bash
python inference/phase5_api_benchmarking.py --server-url http://127.0.0.1:8000 --requests 1000
```

### Run Phase 6 Failure & Error Handling Test Suite
```bash
python tests/test_phase6_error_handling.py
```

### Run All Regression Test Suites (Phases 1–6)
```bash
python tests/test_offline_inference.py
python tests/test_phase2_benchmarks.py
python tests/test_api.py
python tests/test_phase4_consistency.py
python tests/test_phase5_api_benchmarks.py
python tests/test_phase6_error_handling.py
```

### Run Phase 7 Docker Unit Tests
```bash
python tests/test_phase7_docker.py
```

---

## 8. Phase 7 — Docker & Deployment

### What is Docker?

Docker is a tool that packages an application together with everything it needs to run — the code, the model files, the Python version, and all libraries — into a single portable unit called a **container image**. Once built, this image can run identically on any machine that has Docker installed.

### Why Docker for ML Serving?

- **Reproducibility:** The same Python 3.11, same library versions, same model files — always.
- **Portability:** Build once, run anywhere.
- **Isolation:** The container has its own environment, independent of the host machine.
- **Deployment readiness:** Containerized APIs are the standard format for deploying ML models in production.

### Docker Image Details

| Item | Value |
| :--- | :--- |
| **Image name** | `fraud-model-serving:e1` |
| **Image ID** | `a60eb6646ed2` |
| **Base image** | `python:3.11-slim` |
| **System dependency** | `libgomp1` (required by LightGBM) |
| **Disk size** | `729 MB` |
| **Content size** | `168 MB` |
| **Port exposed** | `8000` |
| **Entrypoint** | `uvicorn api.main:app --host 0.0.0.0 --port 8000` |

### Phase 7 Verified Results

| Component | Status | Evidence |
| :--- | :--- | :--- |
| Docker installed | **PASS** | `Docker version 29.8.0` |
| Image build | **PASS** | `fraud-model-serving:e1` built successfully |
| Container startup | **PASS** | Uvicorn running, model loaded, 406 features |
| `GET /health` | **PASS** | HTTP 200, `model_loaded: true`, threshold `0.616521` |
| `POST /predict` | **PASS** | HTTP 200, `decision: LEGIT` for TX 3544193 |
| `POST /predict/batch` | **PASS** | HTTP 200, 3/3 predictions returned |
| `GET /docs` | **PASS** | HTTP 200, Swagger UI reachable |
| Error handling | **PASS** | 4/4 scenarios return correct 422, no leakage |
| Dockerized consistency (100 tx) | **PASS** | 100/100 match, mean prob diff = 0.00000000 |
| Container restart | **PASS** | Recovered in <10 seconds |
| Artifact SHA256 integrity | **PASS** | All 3 E1 artifacts verified |
| Regression Phases 1–6 | **PASS** | All phases unmodified |

### E1 Artifact SHA256 Hashes (Approved Reference)

| Artifact | SHA256 |
| :--- | :--- |
| `models/E1/model.txt` | `d04dff4f765801b196a5eda7d24288e8fd792f5fc06a0576d0de2249f98cc219` |
| `models/E1/preprocessing.joblib` | `0c336989206214cab202d3b4a8a726206cb4ca69a0908e52fdb6f9cf479fbf69` |
| `models/E1/feature_names.json` | `1c59105a626f57533af4fc56f3ba10ae112b739c16c2e2d99cec24c1b1d0330d` |

### How to Build and Run

```bash
# Build the Docker image
docker build -t fraud-model-serving:e1 .

# Start the container
docker run -d --name fraud-model-serving-e1 -p 8000:8000 fraud-model-serving:e1

# Check it is running
docker ps

# View startup logs
docker logs fraud-model-serving-e1

# Test health endpoint
curl http://localhost:8000/health

# Test predict endpoint
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"TransactionID\": 3544193, \"TransactionAmt\": 100.0}"

# Open Swagger UI
# Navigate to: http://localhost:8000/docs

# Restart the container
docker restart fraud-model-serving-e1

# Stop the container
docker stop fraud-model-serving-e1
```

### Limitations

- **Local Docker verification only** — no cloud or remote deployment performed.
- **No HTTPS/TLS** — plain HTTP on port 8000.
- **No Kubernetes or orchestration** — single container, single host.
- Docker packages and reproduces the serving environment; it does not alter model predictions or thresholds.

---

## 9. Phase 8 — Monitoring & Model Versioning

### Overview

Phase 8 adds lightweight, research-appropriate observability to the serving system.
No external frameworks (Prometheus, Grafana, OpenTelemetry) are introduced.
All monitoring is implemented using Python's built-in `logging` and `dataclasses` modules.

### Components

| Module | Purpose |
|:-------|:--------|
| `monitoring/model_metadata.py` | `MODEL_METADATA` dict — single source of truth for E1 model identity |
| `monitoring/api_monitor.py` | `MonitoringState` — in-memory request counter and latency accumulator |
| `monitoring/inference_logger.py` | Structured per-request inference log (no raw payload) |

### What Is Monitored

| Metric | Tracked By |
|:-------|:-----------|
| Total request count | `MonitoringState.total_requests` |
| Successful / failed requests | `MonitoringState.successful_requests / failed_requests` |
| HTTP 2xx / 4xx / 5xx buckets | `MonitoringState.http_2xx_count / http_4xx_count / http_5xx_count` |
| Per-prediction latency (ms) | `MonitoringState.latency_ms_list` |
| Latency percentiles (P50/P95/P99) | `MonitoringState.get_summary()` |
| Per-request structured log lines | `inference_logger.log_inference_event()` |
| Validation error log lines | `inference_logger.log_inference_error()` |
| Model version in /health response | `HealthResponse.model_version` (new field) |

### Structured Inference Log Format

Every successful prediction emits one log line via the `inference_monitor` logger:

```
[INFO] inference_monitor: endpoint=/predict | transaction_id=3544193 | model_version=E1 | \
  fraud_probability=0.003993 | decision=LEGIT | preprocessing_ms=128.831 | model_ms=1.656 | \
  total_ms=130.488 | http_status=200
```

Every validation or inference error emits a warning line:

```
[WARNING] inference_monitor: endpoint=/predict/batch | http_status=422 | error_type=validation_error
```

**Privacy rule**: No raw transaction payload fields are logged.

### Updated /health Response

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "E1_LightGBM",
  "model_version": "E1",
  "feature_count": 406,
  "decision_threshold": 0.616521
}
```

### Phase 8 Experiment Results (100 Requests)

| Metric | Value |
|:-------|:------|
| Total requests | 110 (100 success + 10 errors) |
| Successful (2xx) | 100 |
| Failed (4xx) | 10 |
| Error detection rate | 100% |
| Latency Min | 85.705 ms |
| Latency Mean | 123.504 ms |
| Latency P50 | 125.415 ms |
| Latency P95 | 133.006 ms |
| Latency P99 | 134.029 ms |
| Latency Max | 171.384 ms |
| Model version observed | E1 (all 100 requests) |

### Phase 8 Test Results

- **`tests/test_phase8_monitoring.py`**: **15/15 PASS**
- **Phase 3 Regression** (`tests/test_api.py`): **6/6 PASS**
- **Phase 6 Regression** (`tests/test_phase6_error_handling.py`): **12/12 PASS**
- **Phase 7 Regression** (`tests/test_phase7_docker.py`): **6/6 PASS**

### Limitations

- **In-memory only**: MonitoringState resets on process restart. No persistence.
- **Not thread-safe**: Sequential research use only; not safe under heavy concurrency.
- **No alerting**: Thresholds, alerts, or dashboards are out of scope for this prototype.
- **No log file output**: Log lines go to stdout/stderr via the standard Python logging handler.
  In production, a log aggregator (ELK, Loki) would capture these.
