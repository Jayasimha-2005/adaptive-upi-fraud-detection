# Phase 4 Adaptive Fraud Detection FastAPI Service

## Member 3 — Real-Time Inference & Model Serving API

### Overview
This service provides high-performance, real-time single and batch inference endpoints for the Phase 4 concept-drift-adapted LightGBM fraud detection model trained on the Bank Account Fraud (BAF) benchmark dataset.

---

### Project Architecture & Model Specifications
- **Dataset:** Bank Account Fraud (BAF Base), 29 canonical features (24 numerical, 5 categorical).
- **Approved Inference Model:** LightGBM `lgbm_v3.txt` (Champion Adaptive Model retrained on Months 0–6 following Member 2 drift triggers; outperforming Static v1 on Month 7 with ΔPR-AUC +0.0402, 95% CI [+0.0268, +0.0538]).
- **Baseline Comparator Model:** LightGBM `lgbm_v1.txt` (Static baseline trained on Months 0–3).
- **Frozen Classification Threshold:** `0.8964321608040201` (~0.8964, selected via F1-max sweep on Month 4 validation data and permanently locked across all versions).

---

### Project Layout
```text
phase4_api/
├── README.md                              # Service overview and operational documentation
├── requirements.txt                       # Explicit API and inference dependencies
├── src/
│   ├── __init__.py                        # Core API module & symbol exports
│   ├── main.py                            # FastAPI application entry point & exception handlers
│   ├── routes.py                          # REST endpoint routers (health, model info, predictions)
│   ├── dependencies.py                    # Decoupled dependency injection providers
│   ├── api_schemas.py                     # Pydantic v2 request & response schemas
│   ├── model_loader.py                    # Reusable LightGBM loader & schema validator
│   ├── transaction_preprocessor.py        # Transaction preprocessing & validation engine
│   ├── inference_engine.py                # Single-transaction inference engine & decision logic
│   ├── batch_inference_engine.py          # Vectorized batch inference engine & partial success handler
│   ├── risk_classifier.py                 # Risk classification, reason codes & metadata engine
│   └── prediction_schema.py               # Standardized prediction schemas & Pydantic v2 models
├── tests/
│   ├── __init__.py                        # Test package initializer
│   ├── test_model_loader.py               # Unit tests for model loader (14/14 PASS)
│   ├── test_transaction_preprocessor.py   # Unit tests for preprocessing & compatibility (14/14 PASS)
│   ├── test_inference_engine.py           # Unit tests for single-transaction engine (25/25 PASS)
│   ├── test_inference_smoke.py            # End-to-end real model smoke tests (2/2 PASS)
│   ├── test_batch_inference_engine.py     # Unit & smoke tests for batch engine (32/32 PASS)
│   ├── test_risk_classification.py        # Unit & smoke tests for risk classification (33/33 PASS)
│   └── test_api.py                        # Automated API unit, integration & smoke tests (48/48 PASS)
└── reports/
    ├── phase1_model_inspection.md         # Phase 1 Model & Artifact Inspection Report
    ├── phase2_model_loading.md            # Phase 2 Model Loader & Validation Report
    ├── phase3_preprocessing.md            # Phase 3 Inference Preprocessing Report
    ├── phase4_single_transaction_inference.md # Phase 4 Single Transaction Inference Report
    ├── phase5_batch_inference.md          # Phase 5 Batch Transaction Inference Report
    ├── phase6_prediction_result_risk_classification.md # Phase 6 Risk Classification Report
    └── phase7_fastapi_development.md      # Phase 7 FastAPI Development Report
```

---

### Single Transaction Inference (`phase4_api/src/inference_engine.py`)

The `SingleTransactionInferenceEngine` connects the model loader and preprocessor into a unified, low-latency prediction pipeline:

1. **Model & Threshold Reuse:** Ingests the cached LightGBM `v3` Booster and the authoritative frozen threshold (`0.8964321608040201`).
2. **Preprocessing Integration:** Evaluates the input against the 29 canonical BAF features, validates numerical bounds, enforces native LightGBM categorical encoding, and produces a 1x29 DataFrame.
3. **Probability Evaluation:** Executes `booster.predict(df)` and strictly validates that the output is a single, finite numeric float within `[0.0, 1.0]`.
4. **Binary Decisioning:**
   ```python
   if fraud_probability >= threshold:
       prediction = "FRAUD"
   else:
       prediction = "LEGIT"
   ```
5. **Structured Output:** Emits a `PredictionResponse` containing `transaction_id`, `prediction`, `fraud_probability`, `threshold`, `model_version`, `model_status`, and latency.

---

### Batch Transaction Inference (`phase4_api/src/batch_inference_engine.py`)

The `BatchTransactionInferenceEngine` implements high-throughput, vectorized inference for lists of transactions:

1. **Input Sequence Preservation:** The output array strictly preserves the exact ordering of input records (`input_index: 0, 1, 2, ...`), even when invalid records are interleaved.
2. **Partial-Success Architecture:** Individual invalid records (e.g., missing features, malformed types) do not fail the batch. Valid records are extracted, batched, and predicted, while invalid records receive structured error descriptors.
3. **Vectorized Prediction Execution:** Instead of calling the model $N$ times, valid records are combined into a single aggregated DataFrame and evaluated via **one vectorized `booster.predict(df_batch)` call**, achieving order-of-magnitude lower latency per transaction.
4. **Single-vs-Batch Mathematical Consistency:** Identical transactions processed via single or batch inference yield mathematically identical probabilities (within $10^{-12}$ float precision) and identical binary decisions.
5. **Configurable Batch Guardrails:** Protects memory with a configurable `max_batch_size` (default: 5000 records). Oversized batches raise `BatchSizeLimitExceededError` without silent truncation.
6. **Duplicate Transaction ID Handling:** Preserves duplicate IDs alongside their unique `input_index` without merging or dropping records.

```
Batch Execution Flow:
Input batch (N records)
  ├── Record Validation & Error Capture
  ├── Separation into Valid & Error Sets
  ├── Aggregated Column-Oriented DataFrame (N_valid rows)
  ├── ONE Vectorized booster.predict(df_batch)
  ├── Frozen Threshold Application (0.896432)
  └── Ordered Result Assembly (re-merged at original input indices)
```

---

### Usage Example: Batch Inference

```python
from phase4_api.src.batch_inference_engine import get_batch_inference_engine

# 1. Initialize batch engine singleton
engine = get_batch_inference_engine(model_version="v3")

# 2. Batch input list
transactions = [
    {
        "transaction_id": "TXN-001",
        "income": 0.5,
        "name_email_similarity": 0.825,
        "prev_address_months_count": -1,
        "current_address_months_count": 24,
        "customer_age": 35,
        "days_since_request": 0.015,
        "intended_balcon_amount": -1.0,
        "zip_count_4w": 1250,
        "velocity_6h": 2500.0,
        "velocity_24h": 4500.0,
        "velocity_4w": 5200.0,
        "bank_branch_count_8w": 2,
        "date_of_birth_distinct_emails_4w": 1,
        "credit_risk_score": 140,
        "email_is_free": 1,
        "phone_home_valid": 1,
        "phone_mobile_valid": 1,
        "bank_months_count": 15,
        "has_other_cards": 0,
        "proposed_credit_limit": 1000.0,
        "foreign_request": 0,
        "session_length_in_minutes": 7.5,
        "keep_alive_session": 1,
        "device_distinct_emails_8w": 1,
        "payment_type": "AA",
        "employment_status": "CA",
        "housing_status": "BC",
        "source": "INTERNET",
        "device_os": "windows",
    },
    {
        # Invalid transaction record missing features
        "transaction_id": "TXN-002",
        "income": 0.1,
    }
]

# 3. Predict batch
response = engine.predict_batch(transactions)

print(f"Batch Status: {response.batch_status}")  # "partial_success"
print(f"Success: {response.successful_transactions}/{response.total_transactions}")

for item in response.results:
    if item.status == "success":
        print(f"[{item.input_index}] {item.transaction_id}: {item.prediction} (P={item.fraud_probability:.4f})")
    else:
        print(f"[{item.input_index}] {item.transaction_id}: ERROR ({item.error['code']}) - {item.error['message']}")
```

---

### Prediction Result & Risk Classification (`phase4_api/src/risk_classifier.py`, `prediction_schema.py`)

Phase 6 provides a standardized result layer and risk policy evaluator:

1. **Decoupled Risk Policy:** Configurable `RiskPolicy` categorizes model probabilities into 4 tiers:
   - `LOW`: `[0.0, 0.20)` - Normal baseline transaction.
   - `MEDIUM`: `[0.20, 0.50)` - Slightly elevated risk indicators.
   - `HIGH`: `[0.50, threshold)` - Elevated suspicion. Decision is **`LEGIT`** (below automated block threshold $\approx 0.8964$) but flagged for secondary authentication or manual review.
   - `CRITICAL`: `[threshold, 1.0]` - Probability meets or exceeds frozen threshold. Decision is **`FRAUD`**.
2. **Auditable Reason Codes:** Emits structured `ReasonCode` tokens for model decision, threshold comparison, and risk level. Never fabricates synthetic feature attributions; emits `FEATURE_EXPLANATION_UNAVAILABLE` as a neutral disclaimer.
3. **Operational Metadata:** Returns contextual telemetry (`risk_cutoffs`, `risk_policy_version`, `model_version`, `threshold`, `inference_mode`).
4. **Standardized Schemas:** Emits `StandardizedPredictionResult` and Pydantic v2 `PredictionResultModel`, ensuring uniform JSON serialization across single and batch modes.

---

### FastAPI REST Service (`phase4_api/src/main.py`, `routes.py`)

Phase 7 wraps the inference pipeline into a production-oriented REST API:

- **Startup Command:**
  ```bash
  uvicorn src.main:app --app-dir phase4_api --host 127.0.0.1 --port 8000
  ```
- **Interactive Documentation:**
  - Swagger UI: `http://127.0.0.1:8000/docs`
  - ReDoc: `http://127.0.0.1:8000/redoc`
  - OpenAPI 3.1 Schema: `http://127.0.0.1:8000/openapi.json`
- **Key Endpoints:**
  - `GET /health` — Service liveness, readiness, and memory residency status.
  - `GET /api/v1/model/info` — Sanitized model metadata, frozen threshold, and risk cutoffs.
  - `POST /api/v1/predict/single` — Real-time single transaction inference.
  - `POST /api/v1/predict/batch` — Vectorized batch inference with partial-success tolerance.

---

### Running Unit & Smoke Tests

Execute the cumulative test suite (168 tests):

```bash
pytest phase4_api/tests/ -v
```

Test breakdown:
- **`test_model_loader.py` (14 tests):** Model loading, CRLF normalization, schema verification, caching.
- **`test_transaction_preprocessor.py` (14 tests):** Feature casting, categorical encoding, leakage prevention, missing-value policies.
- **`test_inference_engine.py` (25 tests):** Single-transaction engine unit tests, classification boundaries (`<`, `==`, `>`), error handling.
- **`test_inference_smoke.py` (2 tests):** End-to-end real model smoke test executing actual `lgbm_v3.txt` inference.
- **`test_batch_inference_engine.py` (32 tests):** Input validation, order preservation, partial success, duplicate IDs, vectorized prediction, single-vs-batch consistency, and real-model batch smoke test.
- **`test_risk_classification.py` (33 tests):** Standardized schemas, Pydantic validation, risk classification tiers, exact boundary behavior, reason codes, single/batch integration, and real-model smoke test.
- **`test_api.py` (48 tests):** FastAPI application lifecycle, single/batch prediction endpoints, health/readiness, model info, centralized error handling, OpenAPI docs, and real-model HTTP smoke test.

---

### Development Roadmap
- **Phase 1 (Complete):** Project setup, LightGBM artifact inspection, feature schema verification, and compatibility audit.
- **Phase 2 (Complete):** Reusable model loader, schema validator, threshold ingestion, and unit tests.
- **Phase 3 (Complete):** Transaction preprocessor, categorical handling, missing-value policies, model compatibility check, and unit tests.
- **Phase 4 (Complete):** Single-transaction inference engine, frozen threshold decisioning, structured response, and real-model smoke tests.
- **Phase 5 (Complete):** Vectorized batch inference engine, partial-success handling, single-vs-batch consistency, and 32 unit/smoke tests.
- **Phase 6 (Complete):** Standardized prediction result schema, Pydantic v2 models, decoupled risk classification, auditable reason codes, and 33 unit/smoke tests.
- **Phase 7 (Complete):** FastAPI REST application, Pydantic request/response schemas, OpenAPI documentation, centralized error handling, and 48 automated tests.
- **Phase 8 (Complete):** Automated performance benchmarking, latency profiling, throughput scalability, and offline-versus-online prediction parity validation with 22 automated tests.

---

### Phase 8 — Performance Benchmarking & Offline-Online Parity

Phase 8 provides an automated, reproducible benchmarking framework and parity verification suite for the FastAPI application.

#### Architecture of `phase4_api/benchmarks/`
```text
phase4_api/benchmarks/
├── __init__.py                  # Package exports
├── benchmark_config.py          # Benchmark configuration and parameter bounds
├── system_monitor.py            # Hardware, platform, and resource monitor (psutil)
├── dataset_loader.py            # Read-only BAF CSV loader with 29-feature validation
├── api_client.py                # High-resolution HTTP client with monotonic timers
├── single_benchmark.py          # Single-request latency profiling (P50, P90, P95, P99, RPS)
├── batch_benchmark.py           # Batch endpoint scaling across sizes 1, 10, 50, 100
├── concurrency_benchmark.py     # Controlled bounded concurrency (1, 5, 10, 20 workers)
├── parity_checker.py            # Offline (Engine) vs Online (REST API) parity validator
└── benchmark_runner.py          # CLI orchestrator and report generator
```

#### Starting the API Server
Start the Uvicorn serving process:
```bash
uvicorn src.main:app --app-dir phase4_api --host 127.0.0.1 --port 8000
```

#### Running the Full Benchmark Suite
Execute all benchmarks, parity checks, and report generation in a single command:
```bash
python -m phase4_api.benchmarks.benchmark_runner --dataset "Datasets/BAF/Base.csv" --base-url http://127.0.0.1:8000 --limit 100 --batch-sizes 1,10,50,100 --concurrency 1,5,10,20 --warmup 10
```

CLI options:
- `--dataset`: Path to the BAF CSV file (accessed in strict read-only mode).
- `--base-url`: Target FastAPI endpoint URL (default: `http://127.0.0.1:8000`).
- `--limit`: Number of transaction records to evaluate (e.g., `100` or `1000`).
- `--batch-sizes`: Comma-separated batch sizes to test (default: `1,10,50,100`).
- `--concurrency`: Comma-separated worker thread limits (default: `1,5,10,20`).
- `--warmup`: Warm-up requests discarded from steady-state statistics (default: `10`).
- `--tolerance`: Numerical float parity tolerance (default: `1e-8`).
- `--synthetic`: Flag to permit synthetic records if offline testing without BAF dataset.

#### Interpreting Metrics
1. **Client-Observed HTTP Latency vs. Server Latency:**
   - Client latency includes connection pooling, JSON serialization, network loopback, and parsing.
   - Server `execution_time_ms` measures internal inference and risk evaluation.
   - These are strictly separated in reporting and never conflated.
2. **Throughput Scaling (TPS):**
   - Single requests achieve ~13 req/s in sequential mode, scaling to >100 req/s under 20-thread concurrency.
   - Vectorized batching dramatically increases throughput, achieving **>4,600 TPS** at batch size 100.
3. **Offline-Online Parity:**
   - Compares predictions from direct Python engine execution against HTTP responses.
   - Enforces **100% decision and risk agreement** with probability diff $< 10^{-8}$.

#### Output Artifacts
- Results CSVs and JSON summary: `phase4_api/results/`
  - `single_benchmark_results.csv`
  - `batch_benchmark_results.csv`
  - `concurrency_benchmark_results.csv`
  - `offline_online_parity.csv`
  - `benchmark_summary.json`
- Markdown reports: `phase4_api/reports/`
  - `phase8_performance_report.md`
  - `phase8_execution_summary.md`

#### Running Benchmark Unit Tests
Execute the Phase 8 benchmark unit tests (22 tests):
```bash
pytest phase4_api/tests/test_benchmarks.py -v
```
Or run the complete 190-test cumulative test suite:
```bash
pytest phase4_api/tests/ -v
```
