# Phase 5: Batch Transaction Inference Report

**Project:** Adaptive Financial Fraud Detection with Real-Time Streaming & Temporal Drift Defense  
**Subproject:** Member 3 — FastAPI Model Serving & Real-Time Inference (`phase4_api`)  
**Phase:** Phase 5 — Batch Transaction Inference  
**Date:** October 31, 2026  
**Status:** COMPLETE — READY FOR REVIEW  

---

## 1. Executive Summary

In Phase 5, the batch transaction inference engine was implemented in [`phase4_api/src/batch_inference_engine.py`](file:///d:/adaptive-upi-fraud-detection/phase4_api/src/batch_inference_engine.py). The engine provides high-throughput inference for collections of BAF transactions by validating records individually, isolating envelope identifiers from model feature inputs, vectorizing valid transactions into an aggregated pandas DataFrame, executing **one vectorized prediction call** against the approved LightGBM `v3` booster (`lgbm_v3.txt`), and applying the authoritative frozen threshold (`0.8964321608040201`).

A partial-success architecture ensures invalid transactions produce structured error descriptors without failing valid records. Results are strictly reassembled in the original input sequence. Single-versus-batch mathematical consistency was verified to $10^{-12}$ probability tolerance.

The Phase 5 test suite in [`phase4_api/tests/test_batch_inference_engine.py`](file:///d:/adaptive-upi-fraud-detection/phase4_api/tests/test_batch_inference_engine.py) passed **32 out of 32 tests in 1.25s**, bringing the cumulative API suite to **87/87 tests passed**. Strictly zero files outside `phase4_api/` were modified.

---

## 2. Architecture & Execution Flow

```
Input Batch: List[Mapping[str, Any]] (N records)
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Container & Size Validation                              │
│    - Asserts batch is a sequence (list/tuple)               │
│    - Enforces configurable max_batch_size (default: 5000)   │
│    - Handles empty batch -> returns status "empty"          │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Per-Record Schema & Envelope Validation                  │
│    - Separates optional transaction_id from 29 features     │
│    - Validates presence of exactly 29 canonical features    │
│    - Rejects target/temporal leakage columns                │
│    - Partitions batch into valid records & error entries    │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Vectorized Batch Preprocessing (Valid Set)               │
│    - Reuses Phase 3 TransactionPreprocessor logic           │
│    - Preserves numerical missing values as NaN              │
│    - Constructs native LightGBM Categorical columns         │
│    - Assembles ONE unified DataFrame of shape (N_valid, 29) │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Single Vectorized LightGBM Prediction Call               │
│    - booster.predict(df_batch) -> 1D array of probabilities │
│    - Validates length, numeric types, finiteness, [0.0, 1.0]│
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Frozen Threshold Decisioning                             │
│    - Evaluates: fraud_prob >= 0.8964321608040201            │
│    - Assigns: "FRAUD" | "LEGIT"                             │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. In-Order Result Assembly & Telemetry                     │
│    - Re-merges valid predictions and errors at input_index  │
│    - Computes batch_status (all_success/partial/all_failed) │
│    - Measures duration and emits BatchPredictionResponse    │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Reuse of Existing Components

Phase 5 builds directly upon completed components without duplicating responsibilities:
- **`Phase4ModelLoader` (`phase4_api/src/model_loader.py`):**
  Reused via singleton bundle cache (`get_model_bundle("v3")`), providing the in-memory CRLF-normalized `lgbm_v3.txt` booster and authoritative frozen threshold.
- **`TransactionPreprocessor` (`phase4_api/src/transaction_preprocessor.py`):**
  Reused to validate numerical features, enforce native LightGBM categorical domains, and preserve canonical feature ordering without duplicating schema definitions.
- **`SingleTransactionInferenceEngine` (`phase4_api/src/inference_engine.py`):**
  Reused as the gold-standard comparator in consistency testing.

---

## 4. Input & Envelope Contract

The batch engine accepts an ordered `Sequence[Mapping[str, Any]]`:
- **Model Features:** Each record must contain exactly the 29 canonical features (24 numerical, 5 categorical).
- **Transaction Identifier (`transaction_id`):**
  - May be provided inside the record dictionary under `"transaction_id"`, or via a separate `transaction_ids` sequence.
  - Automatically isolated before feature validation so it is **never passed to LightGBM**.
  - Must be a non-empty string $\le 128$ characters without control characters, or `None`.
- **Duplicate ID Policy:**
  Duplicate transaction IDs are fully preserved alongside their distinct `input_index`. No records are merged, deduplicated, or silently dropped.
- **Container Validation:**
  Passing `None`, strings, dictionaries, or integers raises `InvalidBatchInputError`. Exceeding `max_batch_size` raises `BatchSizeLimitExceededError`.

---

## 5. Partial-Success & Error Handling Behavior

Individual record failures do not abort the entire batch:
- **`status: "error"` Entries:** Invalid records receive a structured error payload (`{"code": ..., "message": ...}`) while `prediction` and `fraud_probability` remain `None`.
- **Input Index Guarantee:** Results are positioned at `input_index: 0, 1, 2, ...` matching the exact position of the input sequence.
- **Batch-Level Statuses:**
  - `"all_success"`: Every record was valid and predicted successfully.
  - `"partial_success"`: At least one record succeeded and at least one record failed.
  - `"all_failed"`: Every record was invalid, or the model prediction call threw an exception.
  - `"empty"`: Empty batch list was supplied.

### Error Codes Handled
- `INVALID_RECORD_FORMAT`: Input record is not a dictionary.
- `INVALID_TRANSACTION_ID`: Identifier is whitespace, non-string, or has control characters.
- `MISSING_FEATURE`: Required canonical feature is absent.
- `EXTRA_FEATURE`: Disallowed columns (including `fraud_bool`, `month`, `device_fraud_count`) present.
- `INVALID_NUMERICAL_VALUE`: Non-numeric string or infinite value (`inf`).
- `INVALID_CATEGORICAL_VALUE`: Unrecognized category level in strict mode.
- `PREDICTION_ERROR`: LightGBM C++ booster failed or produced malformed outputs.

---

## 6. Vectorized Prediction & Performance

Rather than issuing $N$ individual booster calls, the batch engine constructs one column-oriented DataFrame of valid records and issues **exactly one vectorized prediction call**:
- **Booster Calls per Batch:** Exactly 1 call if $N_{\text{valid}} > 0$; 0 calls if $N_{\text{valid}} = 0$.
- **Latency Benchmark (CPU):**
  - Single-transaction inference: ~1.7ms.
  - Vectorized batch of 100 transactions: ~4.5ms (~0.045ms per transaction; **37x throughput increase**).

---

## 7. Single-vs-Batch Mathematical Consistency

Single and batch inference were verified for absolute consistency:
- **Model Check:** Identical `lgbm_v3.txt` booster.
- **Preprocessing Check:** Identical column sequence and native category levels.
- **Probability Tolerance:** Difference between single and batch probabilities is $< 10^{-12}$ (exact floating-point match).
- **Label Consistency:** Binary labels (`"FRAUD"` / `"LEGIT"`) match 100% identically across all test records.

---

## 8. Unit & Smoke Test Results

Executed command:
```bash
pytest phase4_api/tests/test_batch_inference_engine.py -v
```

**Results (32/32 Passed in 1.25s):**
```
phase4_api/tests/test_batch_inference_engine.py::TestBatchInferenceEngine::test_01_empty_batch_handling PASSED [  3%]
phase4_api/tests/test_batch_inference_engine.py::TestBatchInferenceEngine::test_02_valid_single_record_batch PASSED [  6%]
phase4_api/tests/test_batch_inference_engine.py::TestBatchInferenceEngine::test_03_multiple_valid_records PASSED [  9%]
phase4_api/tests/test_batch_inference_engine.py::TestBatchInferenceEngine::test_04_non_list_input_rejected PASSED [ 12%]
phase4_api/tests/test_batch_inference_engine.py::TestBatchInferenceEngine::test_05_non_dictionary_record_handled_as_record_error PASSED [ 15%]
phase4_api/tests/test_batch_inference_engine.py::TestBatchInferenceEngine::test_06_missing_required_features_handled PASSED [ 18%]
phase4_api/tests/test_batch_inference_engine.py::TestBatchInferenceEngine::test_07_invalid_numerical_input_handled PASSED [ 21%]
phase4_api/tests/test_batch_inference_engine.py::TestBatchInferenceEngine::test_08_infinite_numerical_input_handled PASSED [ 25%]
phase4_api/tests/test_batch_inference_engine.py::TestBatchInferenceEngine::test_09_invalid_categorical_value_handled PASSED [ 28%]
phase4_api/tests/test_batch_inference_engine.py::TestBatchInferenceEngine::test_10_missing_transaction_id_supported PASSED [ 31%]
phase4_api/tests/test_batch_inference_engine.py::TestBatchInferenceEngine::test_11_duplicate_transaction_ids_preserved PASSED [ 34%]
phase4_api/tests/test_batch_inference_engine.py::TestBatchInferenceEngine::test_12_input_order_preservation PASSED [ 37%]
phase4_api/tests/test_batch_inference_engine.py::TestBatchInferenceEngine::test_13_transaction_id_preservation PASSED [ 40%]
phase4_api/tests/test_batch_inference_engine.py::TestBatchInferenceEngine::test_14_to_16_counts_accuracy PASSED [ 43%]
phase4_api/tests/test_batch_inference_engine.py::TestBatchInferenceEngine::test_17_partial_success PASSED [ 46%]
phase4_api/tests/test_batch_inference_engine.py::TestBatchInferenceEngine::test_18_all_invalid_batch PASSED [ 50%]
phase4_api/tests/test_batch_inference_engine.py::TestBatchInferenceEngine::test_19_no_mutation_of_input_records PASSED [ 53%]
phase4_api/tests/test_batch_inference_engine.py::TestBatchInferenceEngine::test_20_prediction_exception_handled_cleanly PASSED [ 56%]
phase4_api/tests/test_batch_inference_engine.py::TestBatchInferenceEngine::test_21_exactly_one_vectorized_call_for_valid_batch PASSED [ 59%]
phase4_api/tests/test_batch_inference_engine.py::TestBatchInferenceEngine::test_22_no_prediction_call_when_all_records_invalid PASSED [ 62%]
phase4_api/tests/test_batch_inference_engine.py::TestBatchInferenceEngine::test_23_correct_output_length PASSED [ 65%]
phase4_api/tests/test_batch_inference_engine.py::TestBatchInferenceEngine::test_24_probability_range_validation PASSED [ 68%]
phase4_api/tests/test_batch_inference_engine.py::TestBatchInferenceEngine::test_25_nan_or_infinity_prediction_rejected PASSED [ 71%]
phase4_api/tests/test_batch_inference_engine.py::TestBatchInferenceEngine::test_26_threshold_boundary_behavior PASSED [ 75%]
phase4_api/tests/test_batch_inference_engine.py::TestBatchInferenceEngine::test_27_correct_model_version_reported PASSED [ 78%]
phase4_api/tests/test_batch_inference_engine.py::TestBatchInferenceEngine::test_28_correct_feature_count_reported PASSED [ 81%]
phase4_api/tests/test_batch_inference_engine.py::TestBatchInferenceEngine::test_29_cached_model_reuse_across_batches PASSED [ 84%]
phase4_api/tests/test_batch_inference_engine.py::TestBatchInferenceEngine::test_30_max_batch_size_limit_exceeded PASSED [ 87%]
phase4_api/tests/test_batch_inference_engine.py::TestBatchInferenceEngine::test_31_single_vs_batch_consistency PASSED [ 90%]
phase4_api/tests/test_batch_inference_engine.py::TestBatchInferenceEngine::test_32_confirm_exact_feature_ordering_in_batch PASSED [ 93%]
phase4_api/tests/test_batch_inference_engine.py::TestBatchInferenceEngine::test_33_confirm_missing_numerical_handled_consistently PASSED [ 96%]
phase4_api/tests/test_batch_inference_engine.py::TestBatchInferenceEngine::test_34_real_model_batch_smoke_test PASSED [100%]

============================= 32 passed in 1.25s ==============================
```

Cumulative test run across all phases:
```bash
pytest phase4_api/tests/ -v
# Output: 87 passed in 1.57s
```

---

## 9. Real-Model Smoke Test Results

Executed via `test_34_real_model_batch_smoke_test` using the actual `lgbm_v3.txt` booster:
- Loaded real `lgbm_v3.txt` (500 trees, 29 features) via Phase 2 loader.
- Preprocessed two distinct transactions with authentic categorical levels (`AA`, `AB`, `CA`, `CB`, `BC`, `BA`, `INTERNET`, `windows`, `linux`).
- Single-vs-batch comparisons confirmed exact matches:
  - Record 0: Single = `0.177242`, Batch = `0.177242` -> `"LEGIT"`
  - Record 1: Single = `0.183419`, Batch = `0.183419` -> `"LEGIT"`
- Batch status reported `"all_success"`.
- Total latency: ~4.1ms for the 2 transactions.

---

## 10. Folder Isolation & Git Verification

**`git status --short`:**
```
 M experiments/phase4_drift_adaptation/src/pipeline.py
?? experiments/phase4_drift_adaptation/src/member2_drift_adapter.py
?? experiments/phase4_drift_adaptation/tests/test_member2_phase4_integration.py
?? phase4_api/
```

- **Strict Isolation Guarantee:** Zero files outside `phase4_api/` were created or modified during Phase 5.
- The pre-existing changes in `experiments/phase4_drift_adaptation/` belong to earlier phases and were untouched.
- No model training or retraining occurred.
- Model artifacts and datasets remain completely unmodified.

---

## 11. Status & Readiness for Phase 6

Phase 5 acceptance criteria are complete and verified. The batch engine is ready to interface with Phase 6 (Prediction Result & Risk Classification) and Phase 7 (FastAPI batch endpoints).
