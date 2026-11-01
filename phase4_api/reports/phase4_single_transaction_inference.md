# Phase 4: Single Transaction Inference Engine Report

**Project:** Adaptive Financial Fraud Detection with Real-Time Streaming & Temporal Drift Defense  
**Subproject:** Member 3 — FastAPI Model Serving & Real-Time Inference (`phase4_api`)  
**Phase:** Phase 4 — Single Transaction Inference Engine  
**Date:** October 31, 2026  
**Status:** COMPLETE — READY FOR REVIEW  

---

## 1. Executive Summary

In Phase 4, the core single-transaction inference engine was implemented in [`phase4_api/src/inference_engine.py`](file:///d:/adaptive-upi-fraud-detection/phase4_api/src/inference_engine.py). The engine cleanly integrates the Phase 2 model loader ([`phase4_api/src/model_loader.py`](file:///d:/adaptive-upi-fraud-detection/phase4_api/src/model_loader.py)) and Phase 3 transaction preprocessor ([`phase4_api/src/transaction_preprocessor.py`](file:///d:/adaptive-upi-fraud-detection/phase4_api/src/transaction_preprocessor.py)) to execute end-to-end inference against the approved LightGBM `v3` champion model (`lgbm_v3.txt`).

The engine validates transaction inputs, evaluates probability scores with the native booster, applies the authoritative frozen threshold (`0.8964321608040201`), and returns a structured, sanitized `PredictionResponse`.

A dedicated unit test suite ([`test_inference_engine.py`](file:///d:/adaptive-upi-fraud-detection/phase4_api/tests/test_inference_engine.py)) and real-model smoke test suite ([`test_inference_smoke.py`](file:///d:/adaptive-upi-fraud-detection/phase4_api/tests/test_inference_smoke.py)) passed **27/27 tests in 1.29s**, bringing cumulative API tests to **55/55 passed**.

Zero files outside `phase4_api/` were modified.

---

## 2. Architecture & Design Decisions

```
+-------------------------------------------------------------+
| Raw Input Payload (29 Features) + Optional transaction_id    |
+-------------------------------------------------------------+
                                |
                                v
+-------------------------------------------------------------+
| Phase 3: TransactionPreprocessor                            |
| - Verifies presence of exactly 29 canonical features        |
| - Rejects unexpected keys (fraud_bool, month, extra fields)  |
| - Preserves missing numericals as NaN                       |
| - Enforces native LightGBM Categorical dtypes (5 levels)    |
| - Constructs 1x29 DataFrame in invariant canonical sequence |
+-------------------------------------------------------------+
                                |
                                v
+-------------------------------------------------------------+
| Phase 2: Cached LightGBM v3 Booster                         |
| - booster.predict(df) -> 1D fraud probability               |
| - Output validation: single finite float in [0.0, 1.0]      |
+-------------------------------------------------------------+
                                |
                                v
+-------------------------------------------------------------+
| Decision Logic: Authoritative Frozen Threshold             |
| - Threshold = 0.8964321608040201 (from frozen_threshold_v1) |
| - If fraud_probability >= threshold: "FRAUD" else: "LEGIT"  |
+-------------------------------------------------------------+
                                |
                                v
+-------------------------------------------------------------+
| Structured Output: PredictionResponse                       |
| - transaction_id: Preserved without model leakage           |
| - prediction: "FRAUD" | "LEGIT"                             |
| - fraud_probability: Display rounded to 6 decimals          |
| - threshold: 0.896432                                       |
| - execution_time_ms: Measured latency                       |
| - Safe structured logging (no raw feature values or PII)    |
+-------------------------------------------------------------+
```

### Key Design Decisions
1. **Separation of Concerns:**
   - Preprocessing logic remains exclusively in `TransactionPreprocessor`.
   - Model artifact parsing and threshold loading remain exclusively in `Phase4ModelLoader`.
   - Inference execution and classification decisioning are orchestrated by `SingleTransactionInferenceEngine`.
2. **Deterministic Threshold Decisioning:**
   - The frozen threshold (`0.8964321608040201`) is retrieved dynamically from the loaded model bundle rather than hardcoded.
   - Threshold decisioning evaluates internal full-precision probabilities before display rounding.
3. **Transaction Identifier Decoupling:**
   - `transaction_id` is treated strictly as envelope metadata. It is validated (disallowing control characters, whitespace-only, and lengths > 128 characters) and preserved in the response, but never passed to the model input matrix.
4. **Dependency Injection:**
   - `SingleTransactionInferenceEngine` accepts optional `model_bundle` and `preprocessor` arguments, enabling fast unit testing with mocks and boundary tests without touching disk.
5. **Privacy-Preserving Logging:**
   - Standard library logging records execution duration, model version, TID, and decision, without dumping transaction payloads or personal identifiers.

---

## 3. Structured Response Specification

The engine returns a `PredictionResponse` dataclass with `.to_dict()` serialization:

```json
{
  "transaction_id": "TX-BAF-2026-001",
  "prediction": "LEGIT",
  "fraud_probability": 0.445958,
  "threshold": 0.8964321608040201,
  "model_version": "v3",
  "model_status": "ACTIVE_LOADED",
  "feature_count": 29,
  "execution_time_ms": 1.745
}
```

*Guarantees: No internal server filesystem paths or sensitive feature values are included in the serialized output.*

---

## 4. Error Handling Hierarchy

The inference engine defines specific exceptions and reuses Phase 2 and Phase 3 errors without swallow-all `try/except` blocks:

| Exception Class | Trigger Condition |
|---|---|
| `InvalidTransactionIdError` | `transaction_id` is empty, contains control characters, exceeds 128 chars, or is non-string. |
| `ModelNotReadyError` | Underlying model bundle fails initialization or is missing. |
| `MissingFeatureError` | One or more of the 29 canonical features is missing from input. |
| `ExtraFeatureError` | Extra unexpected features (e.g. `fraud_bool`, `month`) present in input. |
| `InvalidNumericalValueError` | Numerical feature is non-convertible or non-finite (`inf`). |
| `InvalidCategoricalValueError` | Categorical feature contains an unrecognized level in strict mode. |
| `PreprocessedOutputValidationError`| Preprocessor DataFrame shape deviates from `(1, 29)`. |
| `ModelPredictionError` | LightGBM C++ booster raises an error during `predict()`. |
| `InvalidModelOutputError` | Prediction output is empty, multiple values, `NaN`, `inf`, or $<0$ or $>1$. |
| `ThresholdConfigurationError` | Loaded threshold is non-numeric, non-finite, or outside $(0.0, 1.0)$. |

---

## 5. Classification Boundary Verification

Using mock booster outputs, boundary conditions around the frozen threshold were verified:

| Scenario | Injected Probability | Threshold | Expected Prediction | Actual Output | Status |
|---|---:|---:|:---:|:---:|:---:|
| **Below Threshold** | `0.896430` | `0.8964321608040201` | `"LEGIT"` | `"LEGIT"` | **PASS** |
| **Exact Threshold** | `0.8964321608040201` | `0.8964321608040201` | `"FRAUD"` | `"FRAUD"` | **PASS** |
| **Above Threshold** | `0.950000` | `0.8964321608040201` | `"FRAUD"` | `"FRAUD"` | **PASS** |

---

## 6. Real-Model Smoke Test Results

Executed via [`phase4_api/tests/test_inference_smoke.py`](file:///d:/adaptive-upi-fraud-detection/phase4_api/tests/test_inference_smoke.py) against the real `lgbm_v3.txt` artifact:
- **Model Ingestion:** Loaded real `lgbm_v3.txt` (500 trees, 29 features) via Phase 2 loader.
- **Preprocessing:** Processed a valid representative transaction with authentic categorical levels (`AA`, `CA`, `BC`, `INTERNET`, `windows`).
- **Probability Output:** `0.445958`
- **Classification:** Evaluated as `LEGIT` (since $0.445958 < 0.896432$).
- **End-to-End Latency:** ~1.7ms on CPU.
- **Status:** **PASS** (Zero warnings, zero schema errors).

*Note: This smoke test confirms interface integration, not statistical model accuracy.*

---

## 7. Test Suite Summary

Executed command:
```bash
pytest phase4_api/tests/test_inference_engine.py phase4_api/tests/test_inference_smoke.py -v
```

**Results (27/27 Passed in 1.29s):**
```
phase4_api/tests/test_inference_engine.py::TestInferenceEngine::test_01_valid_transaction_produces_prediction PASSED [  3%]
phase4_api/tests/test_inference_engine.py::TestInferenceEngine::test_02_model_version_correctly_returned PASSED [  7%]
phase4_api/tests/test_inference_engine.py::TestInferenceEngine::test_03_fraud_probability_between_0_and_1 PASSED [ 11%]
phase4_api/tests/test_inference_engine.py::TestInferenceEngine::test_04_prediction_uses_frozen_threshold PASSED [ 14%]
phase4_api/tests/test_inference_engine.py::TestInferenceEngine::test_05_transaction_id_preserved PASSED [ 18%]
phase4_api/tests/test_inference_engine.py::TestInferenceEngine::test_06_missing_transaction_id_supported PASSED [ 22%]
phase4_api/tests/test_inference_engine.py::TestInferenceEngine::test_07_feature_count_remains_29 PASSED [ 25%]
phase4_api/tests/test_inference_engine.py::TestInferenceEngine::test_08_model_not_reloaded_per_inference PASSED [ 29%]
phase4_api/tests/test_inference_engine.py::TestInferenceEngine::test_09_probability_below_threshold_returns_legit PASSED [ 33%]
phase4_api/tests/test_inference_engine.py::TestInferenceEngine::test_10_probability_equal_to_threshold_returns_fraud PASSED [ 37%]
phase4_api/tests/test_inference_engine.py::TestInferenceEngine::test_11_probability_above_threshold_returns_fraud PASSED [ 40%]
phase4_api/tests/test_inference_engine.py::TestInferenceEngine::test_12_missing_features_rejected PASSED [ 44%]
phase4_api/tests/test_inference_engine.py::TestInferenceEngine::test_13_unexpected_features_rejected PASSED [ 48%]
phase4_api/tests/test_inference_engine.py::TestInferenceEngine::test_14_invalid_numerical_values_rejected PASSED [ 51%]
phase4_api/tests/test_inference_engine.py::TestInferenceEngine::test_15_invalid_categorical_values_rejected PASSED [ 55%]
phase4_api/tests/test_inference_engine.py::TestInferenceEngine::test_16_model_initialization_failure_handled PASSED [ 59%]
phase4_api/tests/test_inference_engine.py::TestInferenceEngine::test_17_model_prediction_exception_handled PASSED [ 62%]
phase4_api/tests/test_inference_engine.py::TestInferenceEngine::test_18_empty_prediction_output_rejected PASSED [ 66%]
phase4_api/tests/test_inference_engine.py::TestInferenceEngine::test_19_multiple_prediction_outputs_rejected PASSED [ 70%]
phase4_api/tests/test_inference_engine.py::TestInferenceEngine::test_20_nan_probability_rejected PASSED [ 74%]
phase4_api/tests/test_inference_engine.py::TestInferenceEngine::test_21_infinite_probability_rejected PASSED [ 77%]
phase4_api/tests/test_inference_engine.py::TestInferenceEngine::test_22_probability_below_0_rejected PASSED [ 81%]
phase4_api/tests/test_inference_engine.py::TestInferenceEngine::test_23_probability_above_1_rejected PASSED [ 85%]
phase4_api/tests/test_inference_engine.py::TestInferenceEngine::test_24_invalid_threshold_rejected PASSED [ 88%]
phase4_api/tests/test_inference_engine.py::TestInferenceEngine::test_25_incorrect_preprocessor_shape_rejected PASSED [ 92%]
phase4_api/tests/test_inference_smoke.py::TestInferenceSmoke::test_end_to_end_real_model_smoke_inference PASSED [ 96%]
phase4_api/tests/test_inference_smoke.py::TestInferenceSmoke::test_singleton_engine_factory PASSED [100%]

============================= 27 passed in 1.29s ==============================
```

Cumulative test run across all phases:
```bash
pytest phase4_api/tests/ -v
# Result: 55 passed in 1.42s
```

---

## 8. Folder Isolation & Git Verification

**`git status --short`:**
```
 M experiments/phase4_drift_adaptation/src/pipeline.py
?? experiments/phase4_drift_adaptation/src/member2_drift_adapter.py
?? experiments/phase4_drift_adaptation/tests/test_member2_phase4_integration.py
?? phase4_api/
```

- **Strict Isolation Confirmation:** All Phase 4 file creations and edits were strictly confined to `phase4_api/`.
- No files in `experiments/` were altered during Phase 4.
- No model artifacts or datasets were modified.
- No model training or retraining was triggered.

---

## 9. Status & Readiness for Phase 5

Phase 4 acceptance criteria are complete. The inference engine is ready for Phase 5 (FastAPI application, Pydantic v2 request/response models, `/api/v1/predict` endpoint routing, and live ASGI server testing).
