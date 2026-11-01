# Phase 7 — FastAPI Development Technical Report

**Project:** Adaptive Financial Fraud Detection with Real-Time Streaming & Temporal Drift Defense  
**Component:** Member 3 — API Serving Layer (`phase4_api/`)  
**Phase:** Phase 7 — FastAPI Development  
**Date:** October 31, 2026  
**Status:** COMPLETE & VERIFIED (168/168 Tests Passed)

---

## 1. Executive Summary

Phase 7 implements a production-oriented, modular REST API serving layer using FastAPI (v0.141.1) and Starlette (v1.6.0) inside `phase4_api/`.

### Core Achievements
1. **Application Lifecycle & Caching:** Implemented an asynchronous lifespan context manager (`lifespan`) in `phase4_api/src/main.py`. The approved LightGBM `v3` Booster and inference engines are pre-warmed and cached on application startup, eliminating redundant model loading per HTTP request.
2. **REST Endpoints Implemented:**
   - `GET /health`: Liveness and readiness monitoring.
   - `GET /api/v1/model/info`: Sanitized telemetry and frozen threshold metadata.
   - `POST /api/v1/predict/single`: Real-time single transaction fraud inference.
   - `POST /api/v1/predict/batch`: High-performance vectorized batch inference with partial-success tolerance.
3. **Structured Pydantic v2 Schemas:** Built comprehensive request and response schemas in `phase4_api/src/api_schemas.py` enforcing strict type checking, feature boundary enforcement, and sanitized serialization without exposing raw filesystem paths or customer PII.
4. **Centralized Exception Handling:** Standardized API error responses across schema validation (HTTP 422), oversized batches (HTTP 413), service unavailability (HTTP 503), and prediction errors (HTTP 500) with zero stack trace leakage.
5. **Interactive Documentation:** Automatic OpenAPI 3.1 schema generation at `/openapi.json`, Swagger UI at `/docs`, and ReDoc at `/redoc`.
6. **Strict Folder Isolation & Zero Model Retraining:** Confined all modifications and file additions exclusively to `phase4_api/`. The LightGBM `v3` Champion Adaptive artifact (`lgbm_v3.txt`), 29 canonical features, and frozen threshold `0.8964321608040201` remain completely unchanged.

---

## 2. API Architecture

```
                                  Client Request
                                        |
                                        v
                 +---------------------------------------------+
                 |          FastAPI ASGI Server (Uvicorn)       |
                 +---------------------------------------------+
                                        |
                                        v
                 +---------------------------------------------+
                 |            CORS & Security Middleware       |
                 +---------------------------------------------+
                                        |
                                        v
                 +---------------------------------------------+
                 |        Centralized Exception Handlers       |
                 |  (Catches 422, 413, 503, 500 with envelopes)|
                 +---------------------------------------------+
                                        |
                 +----------------------+----------------------+
                 |                                             |
                 v                                             v
       GET /health, /model/info                  POST /predict/single, /batch
                 |                                             |
                 v                                             v
   +---------------------------+                 +---------------------------+
   |   FastAPI Dependencies    |                 |   Pydantic v2 Validation  |
   | (Dependency Injection)    |                 |   (Single & Batch Schemas)|
   +---------------------------+                 +---------------------------+
                 |                                             |
                 +----------------------+----------------------+
                                        |
                                        v
                 +---------------------------------------------+
                 |       Pre-Warmed Inference Engines          |
                 | - SingleTransactionInferenceEngine (Phase 4)|
                 | - BatchTransactionInferenceEngine  (Phase 5)|
                 | - TransactionPreprocessor          (Phase 3)|
                 | - Phase4ModelLoader (Cached)       (Phase 2)|
                 | - LightGBM v3 Booster (Resident)            |
                 +---------------------------------------------+
                                        |
                                        v
                 +---------------------------------------------+
                 |        Phase 6 Risk Evaluation Engine       |
                 | - RiskPolicy (LOW, MEDIUM, HIGH, CRITICAL)  |
                 | - Auditable Structured Reason Codes         |
                 | - Explanation Metadata Telemetry            |
                 +---------------------------------------------+
                                        |
                                        v
                 +---------------------------------------------+
                 |         Standardized JSON Response          |
                 +---------------------------------------------+
```

---

## 3. Application Entry Point & Lifespan Management

- **Entry Point:** `phase4_api/src/main.py`
- **Application Factory:** `create_app() -> FastAPI`
- **ASGI Instance:** `app = create_app()`
- **Lifespan Context Manager:**
  ```python
  @asynccontextmanager
  async def lifespan(app: FastAPI):
      # Startup: Pre-warm booster & inference engines into memory
      bundle = get_model_bundle(model_version=APPROVED_DEFAULT_MODEL_VERSION)
      get_inference_engine(model_version=APPROVED_DEFAULT_MODEL_VERSION)
      get_batch_inference_engine(model_version=APPROVED_DEFAULT_MODEL_VERSION)
      yield
      # Shutdown: Clean teardown
  ```

---

## 4. Endpoints Specification

| Method | Endpoint | Summary | Response Status |
|:---|:---|:---|:---:|
| `GET` | `/health` | Service liveness & LightGBM readiness status | `200` |
| `GET` | `/api/v1/model/info` | Safe model telemetry, frozen threshold & risk policy | `200` |
| `POST` | `/api/v1/predict/single` | Real-time single transaction fraud prediction | `200` |
| `POST` | `/api/v1/predict/batch` | High-throughput vectorized batch fraud prediction | `200` |
| `GET` | `/docs` | Swagger UI interactive documentation | `200` |
| `GET` | `/redoc` | ReDoc interactive documentation | `200` |
| `GET` | `/openapi.json` | OpenAPI 3.1 JSON schema | `200` |

---

## 5. Request & Response Schemas

### 5.1 Single Prediction (`POST /api/v1/predict/single`)

#### Request Body (`SingleTransactionRequest`):
Accepts optional `transaction_id` (string up to 128 characters) and all 29 canonical BAF features. Unexpected fields are forbidden (`extra="forbid"`).

#### Response Body (`SinglePredictionResponse`):
```json
{
  "status": "SUCCESS",
  "transaction_id": "TXN-90210",
  "prediction": "LEGIT",
  "fraud_probability": 0.654321,
  "threshold": 0.8964321608040201,
  "risk_level": "HIGH",
  "model_version": "v3",
  "model_status": "ACTIVE_LOADED",
  "feature_count": 29,
  "reason_codes": [
    {
      "code": "MODEL_DECISION_LEGIT",
      "description": "Transaction probability is strictly below the frozen fraud decision threshold.",
      "category": "MODEL_DECISION"
    },
    {
      "code": "THRESHOLD_NOT_MET",
      "description": "Calculated fraud probability does not meet the frozen decision threshold.",
      "category": "THRESHOLD"
    },
    {
      "code": "RISK_LEVEL_HIGH",
      "description": "Transaction probability falls within the high descriptive risk band. Note: Decision may remain LEGIT if below the frozen decision threshold.",
      "category": "RISK_LEVEL"
    },
    {
      "code": "FEATURE_EXPLANATION_UNAVAILABLE",
      "description": "Feature-level attribution (e.g., SHAP/LIME) is not enabled in Phase 6 serving.",
      "category": "EXPLANATION"
    }
  ],
  "explanation_metadata": {
    "model_version": "v3",
    "model_status": "ACTIVE_LOADED",
    "feature_count": 29,
    "threshold": 0.8964321608040201,
    "risk_policy_version": "v1.0-provisional",
    "risk_cutoffs": {
      "low": [0.0, 0.2],
      "medium": [0.2, 0.5],
      "high": [0.5, 0.8964321608040201],
      "critical": [0.8964321608040201, 1.0]
    },
    "inference_mode": "single",
    "explanation_method": "threshold_and_risk_policy",
    "feature_explanations_available": false
  },
  "execution_time_ms": 1.842
}
```

### 5.2 Batch Prediction (`POST /api/v1/predict/batch`)

#### Request Body (`BatchTransactionRequest`):
```json
{
  "transactions": [
    {
      "transaction_id": "TXN-001",
      "income": 0.4,
      ...
    },
    {
      "transaction_id": "TXN-002",
      "income": 0.1
    }
  ]
}
```

#### Response Body (`BatchPredictionResponseModel`):
```json
{
  "batch_status": "partial_success",
  "total_transactions": 2,
  "successful_transactions": 1,
  "failed_transactions": 1,
  "model_version": "v3",
  "threshold": 0.8964321608040201,
  "results": [
    {
      "input_index": 0,
      "transaction_id": "TXN-001",
      "status": "success",
      "prediction": "LEGIT",
      "fraud_probability": 0.123456,
      "threshold": 0.8964321608040201,
      "risk_level": "LOW",
      "model_version": "v3",
      "feature_count": 29,
      "reason_codes": [...],
      "explanation_metadata": {...},
      "error": null
    },
    {
      "input_index": 1,
      "transaction_id": "TXN-002",
      "status": "error",
      "prediction": null,
      "fraud_probability": null,
      "threshold": null,
      "risk_level": null,
      "model_version": null,
      "feature_count": null,
      "reason_codes": null,
      "explanation_metadata": null,
      "error": {
        "code": "MISSING_FEATURE",
        "message": "Transaction missing 28 required features."
      }
    }
  ],
  "execution_time_ms": 3.25
}
```

### 5.3 Health Check (`GET /health`)
```json
{
  "status": "healthy",
  "model_loaded": true,
  "ready": true,
  "model_version": "v3",
  "frozen_threshold": 0.8964321608040201,
  "uptime_seconds": 128.45,
  "timestamp": "2026-10-31T08:58:30.123456+00:00"
}
```

### 5.4 Model Information (`GET /api/v1/model/info`)
```json
{
  "model_version": "v3",
  "model_status": "ACTIVE_LOADED",
  "model_type": "LightGBM Booster",
  "dataset": "Bank Account Fraud - BAF Base",
  "feature_count": 29,
  "num_trees": 500,
  "frozen_threshold": 0.8964321608040201,
  "supported_inference_modes": [
    "single",
    "batch"
  ],
  "risk_policy_version": "v1.0-provisional",
  "risk_cutoffs": {
    "low": [0.0, 0.2],
    "medium": [0.2, 0.5],
    "high": [0.5, 0.8964321608040201],
    "critical": [0.8964321608040201, 1.0]
  }
}
```

---

## 6. Centralized Error Handling

All client-facing errors follow a standardized envelope:
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable explanation.",
    "details": null
  }
}
```

### HTTP Status Code Mapping:
| Situation | HTTP Status | Handler Mapping |
|:---|:---:|:---|
| Valid prediction request | `200` | Successful response |
| Schema validation failure | `422` | `RequestValidationError` |
| Feature missing or extra | `422` | `MissingFeatureError`, `ExtraFeatureError` |
| Numerical/categorical violation | `422` | `InvalidNumericalValueError`, `InvalidCategoricalValueError` |
| Empty batch | `422` | `BatchTransactionRequest` validator |
| Batch exceeds max limit | `413` | `BatchSizeLimitExceededError` |
| Model service unavailable | `503` | `ModelNotReadyError`, `ModelLoaderError` |
| Internal inference failure | `500` | `ModelPredictionError`, `InferenceEngineError` |
| Unhandled exception | `500` | Sanitized catch-all handler (no stack trace) |

---

## 7. How to Run the Application & Tests

### Starting the Server
From the project workspace root:
```bash
uvicorn src.main:app --app-dir phase4_api --host 127.0.0.1 --port 8000
```
Or with auto-reload for local development:
```bash
uvicorn src.main:app --app-dir phase4_api --host 127.0.0.1 --port 8000 --reload
```

Interactive documentation:
- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc:** `http://127.0.0.1:8000/redoc`
- **OpenAPI Schema:** `http://127.0.0.1:8000/openapi.json`

### Executing the Test Suite
Run the cumulative test suite:
```bash
pytest phase4_api/tests/ -v
```

Run Member 2 integration regression test (read-only):
```bash
pytest experiments/phase4_drift_adaptation/tests/test_member2_phase4_integration.py -v
```

---

## 8. Verification & Exact Test Results

### 1. Cumulative API Test Suite Results
```text
pytest phase4_api/tests/ -v
======================= 168 passed, 1 warning in 3.51s ========================
```

### Breakdown by Test Module:
| Test Module | Phase Focus | Test Count | Result |
|:---|:---|:---:|:---:|
| `test_model_loader.py` | Phase 2: Model Loading & Validation | 14 | 14 / 14 PASS |
| `test_transaction_preprocessor.py` | Phase 3: Transaction Preprocessing | 14 | 14 / 14 PASS |
| `test_inference_engine.py` | Phase 4: Single Transaction Inference | 25 | 25 / 25 PASS |
| `test_inference_smoke.py` | Phase 4: End-to-End Real Model Smoke | 2 | 2 / 2 PASS |
| `test_batch_inference_engine.py` | Phase 5: Batch Transaction Inference | 32 | 32 / 32 PASS |
| `test_risk_classification.py` | Phase 6: Prediction Result & Risk Classification | 33 | 33 / 33 PASS |
| `test_api.py` | Phase 7: FastAPI Serving Layer | 48 | 48 / 48 PASS |
| **Total** | | **168** | **168 / 168 PASS** |

### 2. Member 2 Regression Verification:
```text
pytest experiments/phase4_drift_adaptation/tests/test_member2_phase4_integration.py -v
============================= 11 passed in 0.06s ==============================
```

### 3. Real-Model Smoke Test via HTTP:
`test_49_real_model_smoke_test` verified:
- `/health` returned `status="healthy"`, `ready=True`, `model_loaded=True`.
- `/api/v1/model/info` returned `model_version="v3"`, `frozen_threshold=0.8964321608040201`, `feature_count=29`.
- `/api/v1/predict/single` produced valid prediction, probability, risk level, and reason codes using real `lgbm_v3.txt`.
- `/api/v1/predict/batch` processed multi-record payload with identical probabilities and risk tiers.

---

## 9. Security & Production Safeguards

1. **Zero Stack Trace Leakage:** Centralized exception handlers capture internal errors, log full diagnostics to server logs, and return generic, sanitized envelopes to the client.
2. **Zero Sensitive Path Leakage:** No server filesystem paths (e.g. `D:\...`, `artifacts/...`) are exposed in health checks, model info, or error responses.
3. **No Raw Feature or PII Logging:** Request payloads and customer attribute values are strictly omitted from log streams.
4. **Oversized Batch Guardrail:** Batches exceeding `max_batch_size` (default: 5000) are rejected with HTTP 413 without memory exhaustion.

---

## 10. Scoped File-Change Report (Folder Isolation)

All file additions and edits were strictly confined to `phase4_api/`.

### Files Created Inside `phase4_api/`:
- `phase4_api/src/main.py` (FastAPI app, lifespan, centralized exception handlers)
- `phase4_api/src/api_schemas.py` (Pydantic request & response models)
- `phase4_api/src/dependencies.py` (Dependency injection providers)
- `phase4_api/src/routes.py` (Route handlers for health, model info, single, and batch)
- `phase4_api/tests/test_api.py` (48 automated API unit & integration tests)
- `phase4_api/reports/phase7_fastapi_development.md` (Technical report)

### Files Modified Inside `phase4_api/`:
- `phase4_api/src/__init__.py` (Exported FastAPI `app` and `create_app`)
- `phase4_api/README.md` (Updated with API documentation, endpoints, startup command, and test results)

### External Files Modified:
- **ZERO.** All files outside `phase4_api/` remain completely untouched.

---

## 11. Known Limitations & Out-of-Scope Items

1. **Authentication & Authorization:** API keys, OAuth2, and JWT tokens remain out of scope for Phase 7 serving and can be attached via middleware in downstream deployment.
2. **Phase 8 Performance Benchmarking:** Concurrency stress testing, P99 latency measurement, and throughput benchmarks belong to Phase 8.
