# Phase 3: Inference Preprocessing Report

**Project:** Adaptive Financial Fraud Detection with Real-Time Streaming & Temporal Drift Defense  
**Subproject:** Member 3 — FastAPI Model Serving & Real-Time Inference (`phase4_api`)  
**Phase:** Phase 3 — Inference Preprocessing  
**Date:** October 31, 2026  
**Status:** COMPLETE — READY FOR REVIEW  

---

## 1. Executive Summary

In Phase 3, an inference preprocessing engine was implemented in [`phase4_api/src/transaction_preprocessor.py`](file:///d:/adaptive-upi-fraud-detection/phase4_api/src/transaction_preprocessor.py). The preprocessor receives raw transaction records, validates them against the 29 canonical BAF features, enforces strict column sequence, handles numerical and native categorical types in full accordance with original LightGBM training behavior, and constructs an invariant 1x29 pandas DataFrame.

A model compatibility check confirmed that preprocessed DataFrames can be passed directly to the approved `lgbm_v3.txt` booster's `predict()` method, producing valid probability outputs.

The unit test suite in [`phase4_api/tests/test_transaction_preprocessor.py`](file:///d:/adaptive-upi-fraud-detection/phase4_api/tests/test_transaction_preprocessor.py) passed **14 out of 14 tests in 1.17s**, bringing the total API test suite to **28/28 tests passed**.

---

## 2. Preprocessing Architecture & Discoveries from Training Code

### A. Training Preprocessing Behavior Discovered
By inspecting [`experiments/phase4_drift_adaptation/src/data_loader.py:86-89`](file:///d:/adaptive-upi-fraud-detection/experiments/phase4_drift_adaptation/src/data_loader.py#L86-L89), [`experiments/phase4_drift_adaptation/src/model.py:17-19, 98-110`](file:///d:/adaptive-upi-fraud-detection/experiments/phase4_drift_adaptation/src/model.py#L17-L19), and inspecting `booster.dump_model()['pandas_categorical']`:
1. **No External Encoders:** Neither `OneHotEncoder`, `OrdinalEncoder`, nor `LabelEncoder` was used during Phase 4 training.
2. **Native LightGBM Categoricals:** Categorical features were converted strictly to `pd.Categorical` and passed to `LGBMClassifier.fit(..., categorical_feature=CATEGORICAL_COLS)`.
3. **Internal Category Dictionaries:** LightGBM stored the categorical levels in exact sorted order inside the booster metadata:
   - `payment_type`: `['AA', 'AB', 'AC', 'AD', 'AE']` (5 levels)
   - `employment_status`: `['CA', 'CB', 'CC', 'CD', 'CE', 'CF', 'CG']` (7 levels)
   - `housing_status`: `['BA', 'BB', 'BC', 'BD', 'BE', 'BF', 'BG']` (7 levels)
   - `source`: `['INTERNET', 'TELEAPP']` (2 levels)
   - `device_os`: `['linux', 'macintosh', 'other', 'windows', 'x11']` (5 levels)
4. **No Monotonic Feature Scaling:** LightGBM splits continuously on raw values; no standard scaler or min-max normalization was applied in training.

### B. Numerical Feature Handling
- Converts all 24 numerical columns to `float`.
- Disallows non-finite values (`inf`, `-inf`).
- Preserves missing values (`None` or `NaN`) as `np.nan` by default. LightGBM natively routes `NaN` to the optimal default split branch established during tree construction.

### C. Categorical Feature Handling & Unseen Category Policy
- Explicitly instantiates `pd.Categorical([val], categories=levels)` using the exact training levels.
- **Strict Mode (default):** Incoming categorical values must be members of the training levels; otherwise, `InvalidCategoricalValueError` is raised with a list of valid levels.
- **Permissive Mode (`allow_unseen_categories=True`):** Unseen categories are cast to `NaN`, which LightGBM routes to the default categorical missing bucket.
- Missing categorical values (`None`) are safely cast to `NaN` categories.

### D. Leakage & Extra Column Protection
Incoming dictionaries are strictly validated:
- Missing required canonical features raise `MissingFeatureError`.
- Unexpected extra columns (such as target `fraud_bool`, temporal `month`, or constant `device_fraud_count`) raise `ExtraFeatureError` to prevent target/temporal leakage.

### E. Immutability
The preprocessor never modifies or pops keys from the caller's input dictionary.

---

## 3. Preprocessor Output Specification

| Property | Value | Verification |
|---|---|:---:|
| **Output Type** | `pandas.DataFrame` | Confirmed |
| **Row Count** | Exactly 1 (`len(df) == 1`) | Confirmed |
| **Column Count** | Exactly 29 (`len(df.columns) == 29`) | Confirmed |
| **Column Ordering** | Indices 0–23: Numerical<br>Indices 24–28: Categorical | Confirmed |
| **Numerical Dtypes** | Subtype of `np.number` (`float64`) | Confirmed |
| **Categorical Dtypes** | `pd.CategoricalDtype` matching training levels | Confirmed |

---

## 4. Model Compatibility Test Results

The compatibility test (`test_14_model_compatibility_with_lightgbm_v3`) loaded the approved `lgbm_v3.txt` booster via `get_model_bundle("v3")` and executed:
```python
df_preprocessed = preprocessor.transform(sample_transaction)
pred_proba = booster.predict(df_preprocessed)
prob = float(pred_proba[0])
```

**Results:**
- **Inference Status:** SUCCESS (No schema mismatch or C++ casting errors)
- **Output:** Finite probability score ($0 \le P(\text{fraud}=1) \le 1$)
- **Sample Score:** `0.445958`
- **Threshold Decision against 0.896432:** Evaluated as `False` (LEGIT) without error.

---

## 5. Unit Test Suite Summary

Executed command:
```bash
pytest phase4_api/tests/test_transaction_preprocessor.py -v
```

**Test Results (14/14 Passed in 1.17s):**
```
phase4_api/tests/test_transaction_preprocessor.py::TestTransactionPreprocessor::test_01_valid_transaction_preprocessing PASSED [  7%]
phase4_api/tests/test_transaction_preprocessor.py::TestTransactionPreprocessor::test_02_exactly_29_output_columns PASSED [ 14%]
phase4_api/tests/test_transaction_preprocessor.py::TestTransactionPreprocessor::test_03_correct_numerical_feature_handling PASSED [ 21%]
phase4_api/tests/test_transaction_preprocessor.py::TestTransactionPreprocessor::test_04_correct_categorical_feature_handling PASSED [ 28%]
phase4_api/tests/test_transaction_preprocessor.py::TestTransactionPreprocessor::test_05_exact_feature_ordering PASSED [ 35%]
phase4_api/tests/test_transaction_preprocessor.py::TestTransactionPreprocessor::test_06_missing_required_feature_rejection PASSED [ 42%]
phase4_api/tests/test_transaction_preprocessor.py::TestTransactionPreprocessor::test_07_unexpected_feature_rejection PASSED [ 50%]
phase4_api/tests/test_transaction_preprocessor.py::TestTransactionPreprocessor::test_08_invalid_numerical_value_rejection PASSED [ 57%]
phase4_api/tests/test_transaction_preprocessor.py::TestTransactionPreprocessor::test_09_invalid_categorical_value_handling PASSED [ 64%]
phase4_api/tests/test_transaction_preprocessor.py::TestTransactionPreprocessor::test_10_missing_numerical_values_handling PASSED [ 71%]
phase4_api/tests/test_transaction_preprocessor.py::TestTransactionPreprocessor::test_11_missing_categorical_values_handling PASSED [ 78%]
phase4_api/tests/test_transaction_preprocessor.py::TestTransactionPreprocessor::test_12_empty_input_rejection PASSED [ 85%]
phase4_api/tests/test_transaction_preprocessor.py::TestTransactionPreprocessor::test_14_model_compatibility_with_lightgbm_v3 PASSED [ 92%]
phase4_api/tests/test_transaction_preprocessor.py::TestTransactionPreprocessor::test_15_original_input_object_unchanged PASSED [100%]

============================= 14 passed in 1.17s ==============================
```

Combined suite test execution:
```bash
pytest phase4_api/tests/ -v
# Result: 28 passed in 1.40s
```

---

## 6. Verification Commands Executed

1. **Python Syntax Compilation:**
   ```bash
   python -m py_compile phase4_api/src/transaction_preprocessor.py
   # Returncode: 0
   ```
2. **Phase 3 Test Suite:**
   ```bash
   pytest phase4_api/tests/test_transaction_preprocessor.py -v
   # Returncode: 0 (14/14 passed)
   ```
3. **Cumulative API Test Suite:**
   ```bash
   pytest phase4_api/tests/ -v
   # Returncode: 0 (28/28 passed)
   ```
4. **Regression Check on Member 2 Integration Tests:**
   ```bash
   pytest experiments/phase4_drift_adaptation/tests/test_member2_phase4_integration.py -v
   # Returncode: 0 (11/11 passed)
   ```

---

## 7. Folder Isolation & Git Status Verification

**`git status --short`:**
```
 M experiments/phase4_drift_adaptation/src/pipeline.py
?? experiments/phase4_drift_adaptation/src/member2_drift_adapter.py
?? experiments/phase4_drift_adaptation/tests/test_member2_phase4_integration.py
?? phase4_api/
```

- **Strict Isolation Check:** Zero files outside `phase4_api/` were touched during Phase 3.
- No model training or retraining occurred.
- Model artifacts remain unmodified.

---

## 8. Status & Readiness for Phase 4

Phase 3 is complete and verified. The preprocessing component is ready to be integrated into Phase 4 (Pydantic v2 schemas, FastAPI routing, `/predict` endpoints, and live serving).
