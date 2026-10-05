# Phase 8 — Monitoring & Model Versioning Report

## 1. Executive Summary

- **Objective**: Add lightweight, research-appropriate observability to the E1 LightGBM
  fraud model-serving application — without introducing any external monitoring frameworks.
- **Scope**: API request counting, per-request inference logging, model version tracking.
- **Status**: **COMPLETE — ALL VERIFIED**

---

## 2. Research Question Addressed

> "How can fraud detection models be served with low latency while maintaining consistency
> between offline training and online inference?"

Phase 8 adds the **observability layer** to this question:

- Request counts and latency distributions confirm whether low-latency SLAs are being met
  during serving.
- Structured inference logs provide per-request traceability with model version tagging,
  confirming that *only* the approved E1 model serves predictions.
- Error monitoring distinguishes inference failures from validation failures, supporting
  root-cause analysis without raw payload logging.

---

## 3. Design Decisions

### 3.1 No External Frameworks

- **Rationale**: Prometheus, Grafana, OpenTelemetry, Datadog — all require network endpoints,
  additional services, or infrastructure configuration. This project is a research prototype
  running locally without cloud dependencies.
- **Choice**: Pure Python stdlib (`dataclasses`, `statistics`, `logging`). Zero new dependencies.

### 3.2 In-Memory State (Not Persistent)

- **Rationale**: Persistence requires a database or file-rotation mechanism. For a research
  prototype, state that resets on restart is sufficient to demonstrate the design.
- **Acknowledged Limitation**: MonitoringState counters reset on process restart.

### 3.3 Structured Logging (Not JSON Logging)

- **Rationale**: JSON log lines require a log aggregation pipeline to be useful. Human-readable
  `key=value` structured lines are immediately readable in any terminal without tooling.
- **Fields logged**: endpoint, transaction_id, model_version, fraud_probability, decision,
  preprocessing_ms, model_ms, total_ms, http_status — no raw payload.

### 3.4 Single Source of Truth (MODEL_METADATA)

- **Rationale**: Before Phase 8, `"E1_LightGBM"` was hardcoded in `health_check()` and
  `E1_DECISION_THRESHOLD` was imported separately. This created the risk of drift.
- **Choice**: `monitoring/model_metadata.py` is the canonical reference. It imports
  `E1_DECISION_THRESHOLD` from the inference module — no duplication.

### 3.5 Validation Errors Tracked in Monitor

- **Implementation note**: FastAPI's `RequestValidationError` handler fires *before* the
  endpoint body runs, so monitoring calls were added directly into the exception handler.
  This ensures 422 validation failures are counted in `failed_requests`.

---

## 4. New Files Created

| File | Purpose |
|:-----|:--------|
| `monitoring/__init__.py` | Package init — re-exports all three components |
| `monitoring/model_metadata.py` | `MODEL_METADATA` dict — single source of truth for E1 identity |
| `monitoring/api_monitor.py` | `MonitoringState` dataclass — request counting + latency accumulation |
| `monitoring/inference_logger.py` | `log_inference_event()` and `log_inference_error()` — structured logging |
| `tests/test_phase8_monitoring.py` | 15-test automated unit and integration test suite |
| `inference/phase8_monitoring_experiment.py` | 110-request controlled experiment script |
| `benchmarks/results/phase8_monitoring_results.csv` | Per-request experiment output |

---

## 5. Modified Files

| File | Change |
|:-----|:-------|
| `api/schemas.py` | Added `model_version` field to `HealthResponse` |
| `api/main.py` | Imported MODEL_METADATA, monitor, loggers; wired into /predict, /predict/batch, /health, and validation exception handler |
| `Dockerfile` | Added `COPY monitoring/ ./monitoring/` |

---

## 6. Unit Test Results — Phase 8

**Suite**: `tests/test_phase8_monitoring.py`

| # | Test | Result |
|:--|:-----|:-------|
| 1 | `test_monitor_counts_requests` | ✅ PASS |
| 2 | `test_monitor_success_error_tracking` | ✅ PASS |
| 3 | `test_monitor_latency_recorded` | ✅ PASS |
| 4 | `test_monitor_get_summary_structure` | ✅ PASS |
| 5 | `test_monitor_latency_percentiles` | ✅ PASS |
| 6 | `test_model_metadata_correct` | ✅ PASS |
| 7 | `test_health_includes_model_version` | ✅ PASS |
| 8 | `test_health_regression` | ✅ PASS |
| 9 | `test_predict_response_unchanged` | ✅ PASS |
| 10 | `test_batch_response_unchanged` | ✅ PASS |
| 11 | `test_monitor_increments_on_predict` | ✅ PASS |
| 12 | `test_monitor_tracks_errors` | ✅ PASS |
| 13 | `test_inference_log_event_fields` | ✅ PASS |
| 14 | `test_inference_log_no_raw_payload` | ✅ PASS |
| 15 | `test_model_version_in_log_matches_metadata` | ✅ PASS |

**Result: 15/15 PASS**

---

## 7. Monitoring Experiment Results

**Script**: `inference/phase8_monitoring_experiment.py`  
**Output**: `benchmarks/results/phase8_monitoring_results.csv`

### 7.1 Experiment Configuration

| Parameter | Value |
|:----------|:------|
| Environment | Local host, FastAPI TestClient (in-process) |
| Request mode | Sequential |
| Successful requests | 100 |
| Deliberate error requests | 10 |
| Total requests | 110 |
| Model | E1_LightGBM |
| Model version | E1 |
| Decision threshold | 0.616521 |
| Feature count | 406 |

### 7.2 Request Count Results (MonitoringState)

| Metric | Value |
|:-------|:------|
| Total requests | 110 |
| Successful (2xx) | 100 |
| Failed (4xx+5xx) | 10 |
| HTTP 2xx count | 100 |
| HTTP 4xx count | 10 |
| HTTP 5xx count | 0 |

### 7.3 Prediction Breakdown (100 successful requests)

| Decision | Count |
|:---------|:------|
| LEGIT | 100 |
| FRAUD | 0 |

*Note: Synthetic transactions with varied amounts (10–505 USD, 5 product codes, 5 card numbers)
all scored below the 0.616521 fraud threshold. This is consistent with the general class
imbalance in the IEEE-CIS dataset (approx. 3.5% fraud rate).*

### 7.4 Inference Latency (100 successful predictions, TestClient)

| Metric | Latency |
|:-------|:--------|
| Min | 85.705 ms |
| Mean | 123.504 ms |
| P50 (Median) | 125.415 ms |
| P95 | 133.006 ms |
| P99 | 134.029 ms |
| Max | 171.384 ms |

*Latency includes full preprocessing pipeline (ServingPreprocessor with 406-feature validation)
and LightGBM prediction, measured via `total_inference_time_ms` from the OfflineInferenceEngine.*

### 7.5 Model Version Tracking

| Metric | Value |
|:-------|:------|
| model_version values observed | `{'E1'}` |
| All match MODEL_METADATA | **True** |

### 7.6 Error Monitoring

| Metric | Value |
|:-------|:------|
| Error requests sent | 10 |
| Recorded as failed | 10 |
| Error detection rate | **100.0%** |

---

## 8. Regression Test Results

All prior phases verified unmodified after Phase 8 changes:

| Suite | Tests | Result |
|:------|:------|:-------|
| `tests/test_api.py` (Phase 3) | 6 | ✅ 6/6 PASS |
| `tests/test_phase6_error_handling.py` (Phase 6) | 12 | ✅ 12/12 PASS |
| `tests/test_phase7_docker.py` (Phase 7) | 6 | ✅ 6/6 PASS |

---

## 9. Immutability Verification

The following artifacts remain unchanged in Phase 8:

| Artifact | Status |
|:---------|:-------|
| `models/E1/model.txt` | **UNCHANGED** |
| `models/E1/preprocessing.joblib` | **UNCHANGED** |
| `models/E1/feature_names.json` | **UNCHANGED** |
| `inference/offline_inference.py` | **UNCHANGED** |
| `preprocessing/serving_wrapper.py` | **UNCHANGED** |
| `adaptive-upi-fraud-detection/` (main repo) | **UNTOUCHED** |

---

## 10. Sample Inference Log Lines

The following are real structured log lines emitted during the Phase 8 experiment
(selected from `inference_monitor` logger output):

```
2026-09-23 15:35:27 [INFO] inference_monitor: endpoint=/predict | transaction_id=800000 | model_version=E1 | fraud_probability=0.091723 | decision=LEGIT | preprocessing_ms=132.367 | model_ms=2.071 | total_ms=134.441 | http_status=200

2026-09-23 15:35:56 [INFO] inference_monitor: endpoint=/predict | transaction_id=800099 | model_version=E1 | fraud_probability=0.154452 | decision=LEGIT | preprocessing_ms=124.014 | model_ms=1.649 | total_ms=125.665 | http_status=200

2026-09-23 15:35:56 [WARNING] inference_monitor: endpoint=/predict/batch | http_status=422 | error_type=validation_error
```

*Observation: Raw transaction payload fields (TransactionAmt, card1, ProductCD, etc.) are
completely absent from all log output, confirming the privacy rule is enforced.*

---

## 11. Updated /health Endpoint Response

After Phase 8, `GET /health` returns:

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

The new `model_version` field (`"E1"`) allows any caller to confirm which model experiment
version is currently serving predictions.

---

## 12. Conclusion

Phase 8 adds complete, research-appropriate observability to the fraud model-serving system:

- **Monitoring**: Every request is counted, classified (success/failure/HTTP bucket), and its
  inference latency accumulated for percentile analysis.
- **Logging**: Every successful prediction emits a structured log line with model version,
  fraud probability, decision, and latency — with zero raw payload exposure.
- **Versioning**: `MODEL_METADATA` in `monitoring/model_metadata.py` is the single source of
  truth for E1 model identity, eliminating hardcoded duplicates across the application.
- **Regression**: All 24 prior-phase tests (Phases 3, 6, 7) continue to pass without
  modification.
