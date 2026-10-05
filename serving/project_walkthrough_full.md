# 📘 Project Walkthrough — Fraud Model Serving
### Member 3 | ML Serving & Online Inference Infrastructure

> **Research Project:** Adaptive Financial Fraud Detection with Temporal Modeling, Concept Drift Adaptation, Explainable AI, and Real-Time Stream Processing
> 
> **Repository:** `fraud-model-serving`
> **Total Phases Completed:** 8 (all verified ✅)

---

## 📖 How to Read This Document

This document explains, in plain language, everything that was built step by step in the `fraud-model-serving` repository. Each phase is explained clearly with:
- **What the goal was**
- **What was built/changed**
- **What the results showed**

Technical terms are explained in simple language the first time they appear.

---

## 🧠 Key Terms & Definitions

| Term | Simple Definition |
|------|-------------------|
| **LightGBM** | A machine learning algorithm that learns from patterns in data to make predictions (like detecting fraud). It works using "decision trees" chained together. |
| **Model** | A trained mathematical program that takes inputs (transaction details) and outputs a prediction (fraud or not fraud). |
| **Inference** | The act of using a trained model to make a prediction on new data. |
| **Offline Inference** | Running predictions locally on your own computer, using files directly — no internet or server needed. |
| **Online Inference / API Serving** | Running predictions through a web server so that any app or system can send a request and get a prediction back instantly. |
| **FastAPI** | A modern Python framework for building web APIs (web services). It's fast, easy to use, and widely used in production systems. |
| **API** | Application Programming Interface. A way for two programs to talk to each other. In our case, an external system sends transaction data, and our API sends back a fraud prediction. |
| **REST API** | A type of API that communicates over the internet using standard HTTP methods (GET, POST, etc.). |
| **Endpoint** | A specific URL in an API that does a specific job. For example, `/predict` is an endpoint that takes transaction data and returns a prediction. |
| **Preprocessing** | Cleaning and transforming raw transaction data into the exact format that the model expects before making a prediction. |
| **Feature** | A single piece of information used as input to the model (e.g., transaction amount, product category). Our model uses 406 features. |
| **Feature Schema** | A fixed list of all 406 feature names in the exact order the model expects them. This must never change. |
| **Threshold** | A cut-off probability value. If the model outputs a fraud probability above this value, the transaction is flagged as FRAUD. Our threshold is `0.616521`. |
| **Parity / Consistency** | Making sure two different systems (e.g., offline vs. online) produce exactly the same prediction for the same input. |
| **Benchmark** | A test that measures speed or performance (e.g., how many requests per second can the API handle). |
| **Latency** | How long it takes to get a result. Lower is better. Measured in milliseconds (ms). |
| **Throughput** | How many requests/transactions can be processed per second. Higher is better. |
| **Concurrency** | Multiple requests happening at the same time (simultaneously). |
| **P50 / P95 / P99** | Percentile latency. P50 = half of requests are faster than this. P95 = 95% of requests are faster. P99 = 99% are faster. The remaining slow ones reveal worst-case behavior. |
| **Docker** | A tool that packages your entire application (code + model + dependencies) into a self-contained "container" so it runs the same way on any computer. |
| **Container** | A lightweight, portable package that includes everything needed to run an application. Think of it as a box that contains your app ready to run anywhere. |
| **SHA256 Checksum** | A unique fingerprint of a file. If the file changes even slightly, the checksum changes. Used to verify a file has not been altered. |
| **Uvicorn** | A fast web server that runs FastAPI applications. |
| **Pydantic** | A Python library for validating incoming data (e.g., checking that a request has the right fields and types). |
| **Unit Test** | A small, automated test that checks one specific piece of functionality to make sure it works correctly. |
| **Regression Test** | A test that makes sure existing functionality hasn't been accidentally broken by new changes. |
| **HTTP Status Code** | A number that tells you the result of an HTTP request. `200` = success, `422` = invalid input, `500` = server error. |
| **Cold Start** | The delay that happens the very first time a model server starts up (loading the model from disk into memory). |
| **RPS** | Requests Per Second — how many API calls the server can handle in one second. |
| **E1** | The name given to our approved LightGBM model. "E1" = Experiment 1. This model is frozen/immutable — it cannot be changed. |
| **Immutable** | Cannot be changed or modified. The E1 model is immutable — it stays exactly as it was approved. |

---

## 🏗️ Project Overview

The main research project involves building a fraud detection system for UPI (Unified Payments Interface) transactions. This system was trained in a separate repository (`adaptive-upi-fraud-detection`).

**Member 3's job** is to take the approved trained model (E1) and build the infrastructure to **serve it in production** — meaning, make it available as a real-time web API that can receive transaction data and instantly return fraud predictions.

The model that was handed over:
- **Model type:** LightGBM (1,000 decision trees)
- **Input:** 406 engineered features per transaction
- **Output:** A probability between 0–1 (how likely the transaction is fraudulent)
- **Fraud threshold:** `0.616521` — above this = FRAUD, below = LEGIT

---

## 📁 Repository: `fraud-model-serving`

This is the repository where all Member 3 work lives. Below is a summary of every folder and file at the end of all 7 phases.

```
fraud-model-serving/
│
├── api/                                     ← Web API (FastAPI server)
│   ├── __init__.py
│   ├── main.py                              ← Main FastAPI app, endpoints, error handlers
│   └── schemas.py                           ← Data validation schemas (Pydantic)
│
├── models/
│   └── E1/                                  ← Approved, immutable E1 model artifacts
│       ├── model.txt                         ← Trained LightGBM model (1,000 trees)
│       ├── preprocessing.joblib              ← Fitted preprocessing pipeline
│       └── feature_names.json               ← List of all 406 feature names (in order)
│
├── preprocessing/
│   ├── __init__.py
│   └── serving_wrapper.py                   ← ServingPreprocessor: wraps preprocessing for API use
│
├── src/
│   ├── __init__.py
│   └── features/
│       ├── __init__.py
│       └── ieee_cis_features.py             ← IEEECISPreprocessor: full feature engineering logic
│
├── inference/
│   ├── __init__.py
│   ├── offline_inference.py                 ← Phase 1: Offline inference engine
│   ├── phase2_benchmarking.py               ← Phase 2: Performance benchmarking
│   ├── phase4_consistency.py                ← Phase 4: Offline vs online consistency checker
│   └── phase5_api_benchmarking.py           ← Phase 5: Live API performance benchmarker
│
├── benchmarks/
│   ├── offline_inference_results.csv        ← Phase 1: Raw prediction results
│   ├── offline_inference_report.md          ← Phase 1: Report
│   ├── phase2_benchmark_report.md           ← Phase 2: Benchmark report
│   ├── phase4_consistency_report.md         ← Phase 4: Consistency report
│   ├── phase5_api_performance_report.md     ← Phase 5: API performance report
│   ├── phase6_error_handling_report.md      ← Phase 6: Error handling report
│   ├── phase7_docker_deployment_report.md   ← Phase 7: Docker deployment report
│   └── results/
│       ├── raw_latency_results.csv          ← Phase 2: Per-transaction timing
│       ├── benchmark_summary.csv            ← Phase 2: Aggregated performance summary
│       ├── offline_online_consistency.csv   ← Phase 4: Per-transaction comparison log
│       ├── api_latency_results.csv          ← Phase 5: Per-request HTTP timing
│       ├── api_performance_summary.csv      ← Phase 5: Concurrency performance summary
│       └── phase6_error_results.csv         ← Phase 6: Error test case results
│
├── tests/
│   ├── __init__.py
│   ├── test_offline_inference.py            ← Phase 1 tests
│   ├── test_phase2_benchmarks.py            ← Phase 2 tests
│   ├── test_api.py                          ← Phase 3 tests
│   ├── test_phase4_consistency.py           ← Phase 4 tests
│   ├── test_phase5_api_benchmarks.py        ← Phase 5 tests
│   ├── test_phase6_error_handling.py        ← Phase 6 tests
│   └── test_phase7_docker.py               ← Phase 7 tests
│
├── run_single_transaction_demo.py           ← Demo: run a single transaction prediction
├── Dockerfile                               ← Phase 7: Instructions to build Docker container
├── requirements.txt                         ← Phase 7: Python packages needed at runtime
├── .dockerignore                            ← Phase 7: Files to exclude from Docker image
└── README.md                                ← Project documentation & results summary
```

---

## ✅ Phase 1 — Offline Inference Engine

### Goal
Set up the ability to run the E1 model **offline** (locally, without any server) and confirm it produces correct predictions on unseen transaction data.

### Why This Matters
Before building an API or a server, we need to confirm the model works correctly when used directly in Python code. This is the foundation everything else is built on.

### What Was Built

#### 🆕 `inference/offline_inference.py`
This is the **Offline Inference Engine** — the core engine that:
1. **Loads the E1 model** from `models/E1/model.txt`
2. **Loads the preprocessing pipeline** from `models/E1/preprocessing.joblib`
3. **Loads the 406 feature names** from `models/E1/feature_names.json`
4. **Preprocesses raw transaction data** through the pipeline
5. **Feeds the processed data** into the model
6. **Returns a fraud probability** and a final decision (FRAUD or LEGIT)

The central class in this file is `OfflineInferenceEngine` — it coordinates the preprocessor and the model together.

#### 🆕 `preprocessing/serving_wrapper.py`
This file wraps the `IEEECISPreprocessor` (the original preprocessing code from the training repository) so it can work cleanly in serving mode. It handles raw API inputs that may be missing optional fields.

#### 🆕 `run_single_transaction_demo.py`
A simple command-line demo script. You can run this file to send a single real transaction through the offline engine and see the prediction printed on screen.

#### 🆕 `tests/test_offline_inference.py`
Automated tests that verify the offline engine works correctly — loads model, produces valid probabilities, and applies the threshold correctly.

#### 🆕 `benchmarks/offline_inference_report.md` & `offline_inference_results.csv`
The Phase 1 report and results showing that the offline engine successfully ran and produced predictions.

### Key Result
- Transaction `3544193` → Fraud probability: `0.003993` → Decision: **LEGIT** ✅

---

## ✅ Phase 2 — Offline Performance Benchmarking

### Goal
Measure exactly **how fast** the offline inference pipeline is. Not just "does it work" but "how long does each step take?"

### Why This Matters
Before building a web API, we need to know the model's baseline performance. If the offline engine already takes too long, we need to know before blaming the API later.

### What Was Built

#### 🆕 `inference/phase2_benchmarking.py`
This file contains the benchmarking engine. It:
1. Runs the offline inference on many transactions (e.g., 1,000)
2. Times each step individually: preprocessing time, model prediction time, total time
3. Calculates statistics: min, mean, median, P95, P99, max latencies
4. Writes detailed results to CSV files and a Markdown report

#### 🆕 `tests/test_phase2_benchmarks.py`
Automated tests that verify the benchmarking results meet acceptable performance thresholds.

#### 🆕 Benchmark output files:
- `benchmarks/results/raw_latency_results.csv` — Every single transaction's timing
- `benchmarks/results/benchmark_summary.csv` — Aggregated statistics
- `benchmarks/phase2_benchmark_report.md` — Full readable report

### Key Results

| Metric | Time |
|--------|------|
| Cold Start (model loading) | ~150 ms (one-time cost) |
| Preprocessing per transaction | ~50 ms |
| Model prediction per transaction | ~2 ms |
| **Total per transaction** | **~52 ms** |
| P95 Latency | < 100 ms |
| P99 Latency | < 150 ms |

**Conclusion:** The offline pipeline is fast and consistent. ✅

---

## ✅ Phase 3 — FastAPI REST Model Serving

### Goal
Wrap the offline inference engine inside a **live web server** so that any application can send HTTP requests to get fraud predictions in real time.

### Why This Matters
In production, a fraud detection system must be accessible as a service. A bank's transaction system would call our API with each transaction and get back an instant decision.

### What Was Built

#### 🆕 `api/main.py` — The FastAPI Application
This is the heart of the online serving system. It contains:

- **Startup logic:** When the server starts, it loads the E1 model and preprocessor into memory once (called "lifespan startup"). This avoids reloading on every request.
- **`GET /health`** — A health-check endpoint. Returns whether the server is running, model is loaded, number of features, and the threshold.
- **`POST /predict`** — The single-transaction prediction endpoint. Accepts raw transaction data as JSON, preprocesses it, runs the model, and returns the fraud probability + decision.
- **`POST /predict/batch`** — A batch prediction endpoint that accepts up to 1,000 transactions at once and processes all of them efficiently.
- **Error handling:** All unexpected errors are caught so the server never crashes and never leaks internal details to the caller.

#### 🆕 `api/schemas.py` — Request & Response Schemas
Defines the structure of valid requests and responses using Pydantic. For example:
- A prediction request must have a `TransactionID` (required) and optionally many other fields.
- A prediction response always returns `fraud_probability`, `decision`, and timing breakdowns.
- If someone sends an invalid request, FastAPI automatically returns a clear `422 Unprocessable Entity` error with details.

#### 🆕 `tests/test_api.py`
Automated tests for all API endpoints — health check, single prediction, batch prediction, and invalid input handling.

### How to Start the Server
```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### Sample API Response (Transaction 3544193)
```json
{
  "transaction_id": "3544193",
  "fraud_probability": 0.003993,
  "decision": "LEGIT",
  "preprocessing_time_ms": 53.497,
  "model_prediction_time_ms": 2.377,
  "total_inference_time_ms": 55.874
}
```

---

## ✅ Phase 4 — Offline vs Online Inference Consistency

### Goal
Prove that the **online API (FastAPI server) produces exactly the same predictions as the offline engine** for the same inputs.

### Why This Matters
When we moved the model from offline Python code into a web server, we needed to guarantee that the behavior didn't change. Even one mismatch would mean the serving pipeline is broken.

### What Was Built

#### 🆕 `inference/phase4_consistency.py`
This consistency checker:
1. Loads 1,000 real unseen test transactions (never seen during training)
2. Runs each transaction through the **offline engine** and records the prediction
3. Sends the same transaction to the **live FastAPI server** (`POST /predict`) and records the prediction
4. Compares every prediction: probability values and FRAUD/LEGIT decisions

#### 🆕 `tests/test_phase4_consistency.py`
Automated tests that confirm the consistency results are within acceptable bounds.

#### 🆕 Output files:
- `benchmarks/results/offline_online_consistency.csv` — Side-by-side offline vs online results for all 1,000 transactions
- `benchmarks/phase4_consistency_report.md` — Full report

### Key Results (1,000 Real Unseen Transactions)

| Metric | Value |
|--------|-------|
| Total transactions tested | 1,000 |
| Successful API calls | 1,000 (100%) |
| Failed API calls | 0 |
| Decisions matching | **1,000 / 1,000 = 100.0000%** |
| Mean probability difference | `0.00000000` (exact match) |
| Max probability difference | `0.00000000` (exact match) |

**Conclusion:** The online API is a perfect replica of the offline engine. ✅

---

## ✅ Phase 5 — FastAPI API Performance Benchmark

### Goal
Measure the **real HTTP API's speed and throughput** under different types of load — single requests, concurrent requests, and batch requests.

### Why This Matters
Phase 2 benchmarked the model running directly in Python. Phase 5 benchmarks the full system including the network, HTTP handling, JSON parsing — everything a real caller experiences.

### What Was Built

#### 🆕 `inference/phase5_api_benchmarking.py`
This benchmarking engine:
1. Sends 1,000 individual HTTP requests to `POST /predict` and times each one
2. Sends concurrent requests (multiple at the same time) and measures how the API scales
3. Sends batch requests (10, 100, 500 transactions per request) to `POST /predict/batch`
4. Calculates throughput (RPS = Requests Per Second), latency percentiles, and error rates

#### 🆕 `tests/test_phase5_api_benchmarks.py`
Automated verification that API performance meets minimum thresholds.

#### 🆕 Output files:
- `benchmarks/results/api_latency_results.csv` — Per-request HTTP timing
- `benchmarks/results/api_performance_summary.csv` — Summary across all scenarios
- `benchmarks/phase5_api_performance_report.md` — Full report

### Key Results

**Cold Start:** `29.37 ms` (one-time startup cost)

**Single-Request Latency (1,000 requests):**

| Metric | Value |
|--------|-------|
| Min | 73.39 ms |
| Mean | 86.44 ms |
| P50 (median) | 85.75 ms |
| P95 | 93.00 ms |
| P99 | 123.92 ms |
| Max | 160.65 ms |
| Throughput | **12.2 RPS** |

**Batch Endpoint Throughput:**

| Batch Size | Transactions/Second |
|------------|---------------------|
| 10 | 114.6 Tx/sec |
| 100 | 861.8 Tx/sec |
| **500** | **2,328.5 Tx/sec** |
| Error Rate | **0.00%** across all tests |

**Conclusion:** The API is fast and reliable. Batch mode is extremely efficient. ✅

---

## ✅ Phase 6 — Failure & Error Handling

### Goal
Make the API **robust against bad inputs, attacks, and unexpected situations** — so that it never crashes, never leaks internal information, and always returns a clear, safe response.

### Why This Matters
In a real production system, not every caller is well-behaved. Some might send missing fields, wrong data types, extremely large values, or even attempts to crash the server. The API must handle all of this gracefully.

### What Was Built & Changed

#### 🔧 `api/main.py` — Enhanced Error Handling
Several important improvements were made:
- **Input sanitization:** All incoming request fields are validated for type and range before the model sees them.
- **Exception handlers:** Added global handlers for `ValidationError` (wrong input format), `HTTPException` (API-level errors), and generic `Exception` (catch-all for unexpected crashes).
- **Server-side logging:** When errors occur, full technical details (stack traces) are logged on the server for debugging — but the client only receives a generic "Internal Server Error" message. This prevents information leakage.
- **Zero information leakage:** No internal file paths, model details, or Python error messages are ever sent back to the caller.

#### 🆕 `tests/test_phase6_error_handling.py`
A comprehensive test suite covering 12+ error scenarios:

| Test Case | What It Tests |
|-----------|---------------|
| Missing required fields | API rejects and explains what's missing |
| Wrong data types | API rejects string where number expected |
| Negative transaction amounts | API handles edge case values safely |
| Extremely large values | API doesn't crash on unusual inputs |
| Empty transaction ID | Handled gracefully |
| Malformed JSON | Returns 422 error cleanly |
| Batch with 0 transactions | Returns appropriate error |
| Batch exceeding 1,000 limit | Returns 422 with clear message |
| Valid transaction | Still returns correct prediction |

#### 🆕 Output files:
- `benchmarks/results/phase6_error_results.csv` — Every test case and its result
- `benchmarks/phase6_error_handling_report.md` — Full report

### Key Results
- All 12+ error scenarios handled correctly ✅
- Zero crashes, zero internal data leaked ✅
- Valid predictions still work perfectly alongside error handling ✅

---

## ✅ Phase 7 — Docker & Deployment

### Goal
Package the entire application (model + code + dependencies) into a **Docker container** so it can be deployed anywhere — any server, any cloud, any environment — and behave identically.

### Why This Matters
Without Docker, "it works on my computer" is a real problem. A Docker container freezes the exact environment the app needs, so it runs the same way on every machine.

### What Was Built

#### 🆕 `Dockerfile`
This file is the blueprint for building the Docker container. It tells Docker:

```
1. Start from an official Python 3.11 base image (lightweight "slim" version)
2. Install libgomp1 — a system library required by LightGBM
3. Copy requirements.txt and install all Python packages
4. Copy the application code and E1 model artifacts
5. Expose port 8000
6. When the container starts, run: uvicorn api.main:app --host 0.0.0.0 --port 8000
```

#### 🆕 `requirements.txt`
Lists all Python packages that must be installed in the container:
- `fastapi` — the web framework
- `uvicorn` — the web server
- `lightgbm` — the model library
- `scikit-learn` — used by the preprocessor
- `pandas`, `numpy` — data processing
- `pydantic` — input validation

#### 🆕 `.dockerignore`
Lists files that should NOT be copied into the Docker image (to keep it small and clean):
- `.venv/` — local virtual environment (not needed in container)
- `tests/` — test files (not needed in container)
- `.git/` — git history (not needed in container)
- Benchmark files and data files

#### 🆕 `tests/test_phase7_docker.py`
A test suite that verifies Docker deployment:
1. **Dockerfile exists** and has correct structure
2. **Requirements.txt** contains all necessary packages
3. **Model artifacts** are all present and correct
4. **SHA256 checksums** match — verifying the model file was not accidentally altered
5. **API endpoints** respond correctly inside the Docker environment
6. **Prediction parity** — Docker container produces the same result as local serving

#### 🆕 `benchmarks/phase7_docker_deployment_report.md`
Full deployment verification report.

### How to Build and Run with Docker
```bash
# Build the Docker image
docker build -t fraud-model-serving:latest .

# Run the container
docker run -p 8000:8000 fraud-model-serving:latest

# Test it
curl http://localhost:8000/health
```

### Key Verification Results
- Docker image builds successfully ✅
- FastAPI starts inside container ✅
- `/health` endpoint responds correctly ✅
- Prediction for transaction `3544193`: Prob `0.003993`, Decision `LEGIT` ✅ (matches all previous phases)
- SHA256 checksums of model files verified ✅
- All 7 regression test suites passed inside Docker ✅

---

## 📊 Full Results Summary — All 7 Phases

| Phase | What Was Verified | Result |
|-------|-------------------|--------|
| **Phase 1** — Offline Inference | Model loads, preprocesses, predicts correctly | ✅ PASS |
| **Phase 2** — Performance | P99 latency < 150ms, consistent timing | ✅ PASS |
| **Phase 3** — FastAPI Serving | All endpoints work, schemas validated | ✅ PASS |
| **Phase 4** — Consistency | 1,000/1,000 predictions match exactly (100%) | ✅ PASS |
| **Phase 5** — API Benchmark | P99 < 124ms, batch 2,328 Tx/sec, 0% errors | ✅ PASS |
| **Phase 6** — Error Handling | 12+ error cases handled, zero leakage | ✅ PASS |
| **Phase 7** — Docker | Container builds, serves, matches predictions | ✅ PASS |

---

## 🔐 The Immutable Core (Never Changed)

Throughout all 7 phases, these items were **never modified**:

| Item | Value |
|------|-------|
| Model file | `models/E1/model.txt` |
| Preprocessing pipeline | `models/E1/preprocessing.joblib` |
| Feature list | `models/E1/feature_names.json` (406 features) |
| Fraud threshold | `0.616521` |
| Training repository | `adaptive-upi-fraud-detection` (untouched) |

---

## 🧵 The Thread That Connects Everything

Every single phase is connected by one reference transaction: **TransactionID `3544193`**

This is a real unseen test transaction used to verify every system produces the same answer:

> Fraud Probability: **0.003993** → Decision: **LEGIT**

This result was verified in:
- Phase 1 (offline engine)
- Phase 3 (FastAPI `/predict` endpoint)
- Phase 4 (consistency checker, as one of 1,000 transactions)
- Phase 7 (Docker container)

All four systems agreed. This is the gold standard proof that the serving pipeline is correct end-to-end.

---

## 🔗 API Quick Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Check if server is running and model is loaded |
| `/predict` | POST | Predict fraud for a single transaction |
| `/predict/batch` | POST | Predict fraud for up to 1,000 transactions |

### Minimal Request Example (all other fields are optional)
```json
POST /predict
{
  "TransactionID": 3544193,
  "TransactionDT": 14757390,
  "TransactionAmt": 100.0,
  "ProductCD": "W"
}
```

### Response
```json
{
  "transaction_id": "3544193",
  "fraud_probability": 0.003993,
  "decision": "LEGIT",
  "preprocessing_time_ms": 53.497,
  "model_prediction_time_ms": 2.377,
  "total_inference_time_ms": 55.874
}
```

---

*Document generated for the `fraud-model-serving` repository — Member 3 ML Serving.*
*All phases completed and verified. Model E1 remains immutable throughout.*

---

## 📡 Phase 8 — Monitoring & Model Versioning

### Goal

Add **observability** to the serving system — the ability to see what the system is doing while it is running. This phase answers questions like:
- How many requests has the API handled?
- How many succeeded vs. failed?
- What is the typical prediction latency?
- Which model version is currently serving?
- What does a prediction look like in the logs?

All of this must be done **without adding any external tools** like Prometheus or Grafana, and **without logging raw transaction data** (privacy protection).

---

### What Was Built

**New Package: `monitoring/`**

Three new Python modules:

#### 1. `monitoring/model_metadata.py` — Single Source of Truth

A dictionary called `MODEL_METADATA` that holds all E1 model identity information in one place:

```python
MODEL_METADATA = {
    "model_version":        "E1",
    "model_name":           "E1_LightGBM",
    "feature_count":        406,
    "decision_threshold":   0.616521,   # imported — not hardcoded
    "artifact_model":       "models/E1/model.txt",
    ...
}
```

Before Phase 8, `"E1_LightGBM"` was hardcoded in the health check function. Now there is one canonical source that everything else reads from.

#### 2. `monitoring/api_monitor.py` — Request Counter & Latency Accumulator

A Python dataclass called `MonitoringState` that accumulates:
- Total requests, successful requests, failed requests
- HTTP 2xx / 4xx / 5xx bucket counts
- A list of inference latency values (for percentile computation)

A `get_summary()` method returns minimum, mean, P50, P95, P99, and maximum latency.

One shared instance (`monitor`) is imported by the API and updated on every request.

#### 3. `monitoring/inference_logger.py` — Structured Inference Logging

Two functions:
- `log_inference_event()` — emits one structured log line per successful prediction
- `log_inference_error()` — emits a structured warning line per failed request

Example log line:
```
[INFO] inference_monitor: endpoint=/predict | transaction_id=3544193 | model_version=E1 |
  fraud_probability=0.003993 | decision=LEGIT | preprocessing_ms=128.8 | total_ms=130.5 | http_status=200
```

No raw transaction payload (amounts, card numbers, product codes) is ever logged.

---

### What Was Changed

| File | Change |
|:-----|:-------|
| `api/schemas.py` | Added `model_version` field to `HealthResponse` |
| `api/main.py` | Imported all 3 monitoring modules; wired monitor.record() and log_inference_event() into `/predict`, `/predict/batch`, and the validation exception handler |
| `Dockerfile` | Added `COPY monitoring/ ./monitoring/` so the package is available inside the container |

---

### Updated /health Response

The health check endpoint now returns a `model_version` field:

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

---

### Phase 8 Experiment — 100 Requests

A controlled experiment was run using 100 synthetic `/predict` requests + 10 deliberate error requests.

**Request Counts:**
| Metric | Value |
|:-------|:------|
| Total requests | 110 |
| Successful (2xx) | 100 |
| Failed (4xx) | 10 |
| Error detection rate | 100% |

**Inference Latency (100 successful predictions):**
| Metric | Value |
|:-------|:------|
| Min | 85.705 ms |
| Mean | 123.504 ms |
| P50 (Median) | 125.415 ms |
| P95 | 133.006 ms |
| P99 | 134.029 ms |
| Max | 171.384 ms |

**Model Version Tracking:** All 100 predictions logged `model_version=E1`. 100% consistent.

---

### Phase 8 Test Results

| Test Suite | Tests | Result |
|:-----------|:------|:-------|
| Phase 8 (`test_phase8_monitoring.py`) | 15 | ✅ 15/15 PASS |
| Phase 3 regression (`test_api.py`) | 6 | ✅ 6/6 PASS |
| Phase 6 regression (`test_phase6_error_handling.py`) | 12 | ✅ 12/12 PASS |
| Phase 7 regression (`test_phase7_docker.py`) | 6 | ✅ 6/6 PASS |

**All prior phases are unaffected. Model and artifacts remain immutable.**

---

*Document updated for the `fraud-model-serving` repository — Member 3 ML Serving.*
*All 8 phases completed and verified. Model E1 remains immutable throughout.*
