# Phase 4 — Offline vs Online Inference Consistency Report

**Member 3 — ML Serving & Inference Infrastructure**  
**Repository**: `fraud-model-serving`  
**Model**: Approved E1 LightGBM Baseline (`models/E1/model.txt`)  
**Threshold**: `0.616521`  
**Feature Count**: `406 Processed Features`  

---

## 1. Executive Summary

This report evaluates whether the **FastAPI online model serving pipeline** (`POST /predict`) produces identical predictions and decisions as the **approved offline E1 inference pipeline** across `100` real unseen test transactions.

### Key Parity Results
- **Evaluated Transactions**: `100`
- **Successful Predictions**: `100` (Offline: `100`)
- **Failed Requests**: `0`
- **Classification Decision Match Rate**: **`100.0000%`** (`100 / 100`)
- **Decision Mismatches**: `0`
- **Mean Probability Difference**: `0.00000000`
- **Median Probability Difference**: `0.00000000`
- **P95 Probability Difference**: `0.00000000`
- **P99 Probability Difference**: `0.00000000`
- **Maximum Probability Difference**: `0.00000000`

---

## 2. Consistency Metrics Summary

| Metric | Value | Status |
| :--- | :--- | :--- |
| **Total Transactions Tested** | `100` | Complete |
| **Successful Predictions** | `100` | 100% Success |
| **Failed Requests** | `0` | None |
| **Matching Decisions** | `100` | **100% Match** |
| **Mismatching Decisions** | `0` | Zero Mismatches |
| **Decision Match Rate** | **`100.0000%`** | **PASS** |
| **Mean Abs Probability Diff** | `0.00000000` | Exact Parity |
| **Median Abs Probability Diff** | `0.00000000` | Exact Parity |
| **P95 Abs Probability Diff** | `0.00000000` | Exact Parity |
| **P99 Abs Probability Diff** | `0.00000000` | Exact Parity |
| **Max Abs Probability Diff** | `0.00000000` | Exact Parity |

---

## 3. Sample Transaction Comparisons

| TransactionID | Offline Probability | Online Probability | Absolute Diff | Offline Decision | Online Decision | Match Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `3544193` | `0.003993` | `0.003993` | `0.00000000` | `LEGIT` | `LEGIT` | **`MATCH`** |
| `3553782` | `0.028656` | `0.028656` | `0.00000000` | `LEGIT` | `LEGIT` | **`MATCH`** |
| `3572950` | `0.356211` | `0.356211` | `0.00000000` | `LEGIT` | `LEGIT` | **`MATCH`** |
| `3523324` | `0.006417` | `0.006417` | `0.00000000` | `LEGIT` | `LEGIT` | **`MATCH`** |
| `3560686` | `0.020108` | `0.020108` | `0.00000000` | `LEGIT` | `LEGIT` | **`MATCH`** |
| `3513511` | `0.662079` | `0.662079` | `0.00000000` | `FRAUD` | `FRAUD` | **`MATCH`** |
| `3545668` | `0.016060` | `0.016060` | `0.00000000` | `LEGIT` | `LEGIT` | **`MATCH`** |
| `3518867` | `0.089978` | `0.089978` | `0.00000000` | `LEGIT` | `LEGIT` | **`MATCH`** |
| `3561136` | `0.000569` | `0.000569` | `0.00000000` | `LEGIT` | `LEGIT` | **`MATCH`** |
| `3565004` | `0.026372` | `0.026372` | `0.00000000` | `LEGIT` | `LEGIT` | **`MATCH`** |

---

## 4. Methodological Safeguards & Parity Guarantees

1. **Identical Preprocessing Artifacts**:
   - Both pipelines load `models/E1/preprocessing.joblib` and `models/E1/feature_names.json`.
   - Pre-alignment of raw feature schema ensures missing fields in online JSON payloads are handled identically to offline NaN values.

2. **Identical Model Booster & Threshold**:
   - Both pipelines infer using `models/E1/model.txt` with frozen decision threshold `0.616521`.

3. **Zero Target & Identifier Leakage**:
   - `isFraud` and `TransactionID` are strictly excluded from feature matrix $X$ in both paths.

---

## 5. Main Repository Immutability Check

- **Source Repository**: `adaptive-upi-fraud-detection`
- **Working Tree Status**: Clean and untouched.
- **Model Parameters/Weights**: 0 modifications.
