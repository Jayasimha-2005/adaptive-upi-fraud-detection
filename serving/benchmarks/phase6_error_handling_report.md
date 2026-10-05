# Phase 6 — Failure & Error Handling Report

**Member 3 — ML Serving & Inference Infrastructure**  
**Repository**: `fraud-model-serving`  
**Model**: Approved E1 LightGBM Baseline (`models/E1/model.txt`)  
**Serving Framework**: FastAPI + Uvicorn  
**Feature Count**: `406 Processed Features`  
**Decision Threshold**: `0.616521`  

---

## 1. Executive Summary

This report documents the failure mode resilience, validation safety, security, and error handling performance of the **FastAPI model-serving application**.

### Key Error Handling Findings
- **Request Validation**: All malformed JSON payloads, missing required fields, and incorrect datatypes are cleanly intercepted by Pydantic / FastAPI validation layer (HTTP `422 Unprocessable Content` / HTTP `400 Bad Request`).
- **Zero Information Leakage**: Unexpected server-side runtime exceptions return sanitized client messages (`"Prediction service encountered an internal error."`) while capturing full tracebacks in server-side logs. Zero Python tracebacks or local file paths (e.g. `C:\Users\...`) appear in HTTP responses.
- **Service Resilience**: Ordinary client request errors and simulated 500 runtime inference failures do NOT crash the server. Subsequent valid requests and `/health` pre-flight checks succeed with 100% reliability.
- **Model Unavailability Handling**: When the E1 model engine is uninitialized or unavailable, endpoints safely return HTTP `503 Service Unavailable` with structured JSON error details.

---

## 2. Tested Error Scenarios & Actual Observed Results

| Test ID | Test Category | Endpoint | Input Payload / Scenario | Expected Status | Actual Status | Test Status | Traceback Leaked |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | Missing Required Fields | `POST /predict/batch` | Missing `transactions` key (`{}`) | `422` | `422` | **PASS** | `False` |
| **2** | Wrong Data Types | `POST /predict/batch` | Invalid structure (`transactions: "string"`) | `422` | `422` | **PASS** | `False` |
| **3** | Malformed JSON Syntax | `POST /predict` | Broken JSON syntax (`{"TransactionID": `) | `422 / 400` | `422` | **PASS** | `False` |
| **4** | Missing / Invalid Feature Data | `POST /predict` | Null & NaN raw feature values | `200` | `200` | **PASS** | `False` |
| **5** | Invalid Batch (Empty List) | `POST /predict/batch` | Empty list (`transactions: []`) | `422` | `422` | **PASS** | `False` |
| **6** | Invalid Batch (Bad Item) | `POST /predict/batch` | Non-object item (`transactions: [12345]`) | `422` | `422` | **PASS** | `False` |
| **7** | Model Engine Unavailable | `POST /predict` | Engine state is `None` | `503` | `503` | **PASS** | `False` |
| **8** | Preprocessor Load Failure | N/A | Missing `preprocessing.joblib` artifact | `FileNotFoundError` | `FileNotFoundError` | **PASS** | `False` |
| **9** | Runtime Inference Failure | `POST /predict` | Monkeypatched `RuntimeError` | `500` | `500` | **PASS** | `False` |
| **10** | Post-Error Health Check | `GET /health` | Health query after repeated failures | `200` | `200` | **PASS** | `False` |
| **11** | Error Response Schema | `POST /predict/batch` | Validate `ErrorResponse` JSON structure | `422` | `422` | **PASS** | `False` |
| **12** | Zero Path/Traceback Leakage | `POST /predict` | Exception with local file path (`C:\...`) | `500` | `500` | **PASS** | `False` |
| **13** | Service Resilience | `POST /predict` | Valid request after failure recovery | `200` | `200` | **PASS** | `False` |

---

## 3. Error Response Specifications

### Standard Validation Error Response (HTTP 422)
```json
{
  "error": "Request Validation Error",
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "transactions"],
      "msg": "Field required",
      "input": {}
    }
  ]
}
```

### Internal Server Error Response (HTTP 500)
```json
{
  "error": "HTTP Error",
  "detail": "Inference execution failed due to an internal error."
}
```

### Service Unavailable Error Response (HTTP 503)
```json
{
  "error": "HTTP Error",
  "detail": "E1 Model Engine is not loaded."
}
```

---

## 4. Main Repository Immutability Check

- **Source Repository**: `adaptive-upi-fraud-detection`
- **Working Tree Status**: Clean and untouched (`git status` verified).
- **Model Parameters/Weights**: 0 modifications.
