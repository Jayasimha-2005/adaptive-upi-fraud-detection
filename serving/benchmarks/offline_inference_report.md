# Phase 1 — Offline Inference Benchmark Report

## 1. Executive Summary
- **Objective**: Prove that unseen raw test transactions can be processed through the approved E1 LightGBM inference pipeline cleanly and accurately with low latency.
- **Member 3 Scope**: Phase 1 — Offline Inference (Model loading, feature validation, threshold decision, latency measurement).
- **Status**: **SUCCESS**

---

## 2. Model Provenance & Specification
- **Model Architecture**: LightGBM Gradient Boosted Decision Trees (E1 Baseline)
- **Source Repository**: `adaptive-upi-fraud-detection`
- **Artifacts Used**:
  - Model: `models/E1/model.txt`
  - Preprocessor: `models/E1/preprocessing.joblib`
  - Feature Names: `models/E1/feature_names.json`
- **Expected Feature Count**: `406`
- **Decision Threshold**: `0.616521` (Immutable)

---

## 3. Preprocessing & Feature Integrity
- **Processed Feature Count**: `406`
- **Feature Name & Order Match**: `PASS` (Exact match against approved `feature_names.json`)
- **Target Leakage Safeguard**: `PASS` (`isFraud` strictly excluded from model feature matrix)
- **Identifier Leakage Safeguard**: `PASS` (`TransactionID` strictly excluded from model feature matrix)

---

## 4. Single-Transaction Execution Result

```text
TransactionID:            3544193
Fraud Probability:        0.003993
Decision:                 LEGIT
Actual Label:             LEGIT

Preprocessing Time:       65.528 ms
Model Prediction Time:    1.842 ms
Total Inference Time:     67.372 ms
```

---

## 5. Multi-Transaction Latency & Throughput Benchmark (1 Transactions)

### Benchmark Summary
| Metric | Value |
| :--- | :--- |
| **Total Test Transactions** | `1` |
| **Successful Inferences** | `1` |
| **Failed Inferences** | `0` |
| **Predicted FRAUD Count** | `0` (0.00%) |
| **Predicted LEGIT Count** | `1` (100.00%) |
| **Agreement with Ground Truth** | `1/1 (100.00%)` |

### Latency Breakdown (Per Transaction Average)
| Pipeline Step | Avg Latency (ms) | Percentage of Total |
| :--- | :--- | :--- |
| **Preprocessing Time** | `55.871 ms` | `96.2%` |
| **Model Prediction Time** | `2.235 ms` | `3.8%` |
| **Total Inference Time** | `58.108 ms` | `100.0%` |

---

## 6. Sample Prediction Results

| TransactionID | Fraud Probability | Decision | Actual Label | Preprocessing (ms) | Model (ms) | Total (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `3544193` | `0.003993` | **LEGIT** | `LEGIT` | `55.871` | `2.235` | `58.108` |

---

## 7. Verification & Integrity Checklist
- [x] Model successfully loads without error
- [x] Preprocessor artifact successfully loads
- [x] `feature_names.json` schema loads and matches preprocessor
- [x] Feature matrix shape is strictly `(N, 406)`
- [x] Feature names and ordering match approved schema exactly
- [x] Zero target leakage (`isFraud` excluded from X)
- [x] Zero identifier leakage (`TransactionID` excluded from X)
- [x] Predicted probabilities are numeric and within `[0.0, 1.0]`
- [x] Decision threshold `0.616521` applied correctly
- [x] End-to-end latency measured per step
- [x] Results saved to `benchmarks/offline_inference_results.csv`
- [x] Original `adaptive-upi-fraud-detection` repository unmodified

---

## 8. Conclusion
The Phase 1 Offline Inference pipeline in `fraud-model-serving` is validated and fully operational. It demonstrates low latency and 100% feature consistency between offline training and offline inference.
