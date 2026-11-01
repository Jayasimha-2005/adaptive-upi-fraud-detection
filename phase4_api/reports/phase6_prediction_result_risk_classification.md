# Phase 6 — Prediction Result & Risk Classification Technical Report

**Project:** Adaptive Financial Fraud Detection with Real-Time Streaming & Temporal Drift Defense  
**Component:** Member 3 — API Serving Layer (`phase4_api/`)  
**Phase:** Phase 6 — Prediction Result & Risk Classification  
**Date:** October 31, 2026  
**Status:** COMPLETE & VERIFIED  

---

## 1. Executive Summary

Phase 6 introduces a decoupled, standardized prediction-result and risk-classification layer for both single-transaction and batch-transaction inference engines within `phase4_api/`.

### Core Achievements
1. **Standardized Unified Schema:** Created `StandardizedPredictionResult` and Pydantic v2 `PredictionResultModel`, ensuring uniform, validated, JSON-serializable outputs across single and batch modes.
2. **Decoupled Risk Classification:** Implemented `RiskPolicy` and `RiskClassifier`, establishing four distinct descriptive risk tiers (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).
3. **Decoupled Fraud Decisioning vs. Risk Level:** Preserved the strict separation between:
   - **Fraud Probability:** Raw model probability $p \in [0.0, 1.0]$.
   - **Fraud Decision:** Frozen-threshold binary decision ($p \ge \text{threshold} \implies \text{FRAUD}$; $p < \text{threshold} \implies \text{LEGIT}$).
   - **Risk Level:** Configurable descriptive tier independent of model retraining.
   - **Key Distinction:** A transaction can exhibit high suspicion ($0.50 \le p < \text{threshold}$) receiving a `"HIGH"` risk tier while remaining a `"LEGIT"` decision because it has not breached the conservative frozen action threshold ($\approx 0.8964$).
4. **Structured Auditable Reason Codes:** Implemented `ReasonCode` emitting auditable decision, threshold, and risk codes without fabricating synthetic feature attributions. Emits a neutral disclaimer `FEATURE_EXPLANATION_UNAVAILABLE` explaining that feature-level attribution (SHAP/LIME) is not enabled in Phase 6.
5. **Operational Explanation Metadata:** Emitted structured operational telemetry (`risk_policy_version`, `risk_cutoffs`, `inference_mode`, `model_version`, `model_status`, `feature_count`, `threshold`).
6. **Zero Model Retraining & Strict Isolation:** Retained LightGBM `v3` Champion Adaptive artifact (`lgbm_v3.txt`), 29 canonical features, and frozen threshold `0.8964321608040201`. Zero modifications occurred outside `phase4_api/`.

---

## 2. Architecture & Design

```
+---------------------------------------------------------------------------------+
|                               Incoming Request                                  |
|                 (Single Transaction or Batch of Transactions)                   |
+---------------------------------------------------------------------------------+
                                       |
                                       v
+---------------------------------------------------------------------------------+
|                       Transaction Preprocessor (Phase 3)                        |
|       - Envelope separation (transaction_id isolated)                           |
|       - Exact 29 canonical feature alignment & ordering                         |
+---------------------------------------------------------------------------------+
                                       |
                                       v
+---------------------------------------------------------------------------------+
|                          LightGBM Booster Execution                             |
|       - Single Booster Predict (Phase 4) or Vectorized Batch Predict (Phase 5)  |
|       - Validated Probability Output p in [0.0, 1.0]                            |
+---------------------------------------------------------------------------------+
                                       |
                                       v
+---------------------------------------------------------------------------------+
|                       Authoritative Decision Logic                              |
|       - Frozen Threshold: 0.8964321608040201                                    |
|       - Decision: "FRAUD" if p >= threshold else "LEGIT"                        |
+---------------------------------------------------------------------------------+
                                       |
                                       v
+---------------------------------------------------------------------------------+
|                   Phase 6: Risk Classifier & Policy Engine                      |
|       - RiskPolicy (v1.0-provisional): LOW, MEDIUM, HIGH, CRITICAL              |
|       - Reason Codes: Stable, auditable justification tokens                    |
|       - Explanation Metadata: Model version, cutoffs, explanation method        |
+---------------------------------------------------------------------------------+
                                       |
                                       v
+---------------------------------------------------------------------------------+
|                    Standardized Prediction Result Schema                        |
|       - Single: PredictionResponse / StandardizedPredictionResult               |
|       - Batch:  BatchPredictionResponse with enriched BatchRecordResult         |
|       - Both JSON-serializable, Pydantic-validated, sanitized (no PII leakage)   |
+---------------------------------------------------------------------------------+
```

---

## 3. Prediction Result Schema

### Fields Specification
| Field Name | Type | Description |
|:---|:---|:---|
| `status` | `str` | `"SUCCESS"` or `"ERROR"` |
| `transaction_id` | `Optional[str]` | Unique transaction identifier |
| `prediction` | `Optional[str]` | `"FRAUD"` if $p \ge \text{threshold}$, else `"LEGIT"` (`None` on error) |
| `fraud_probability` | `Optional[float]` | LightGBM model score rounded to 6 decimal places (`None` on error) |
| `threshold` | `Optional[float]` | Authoritative frozen threshold ($0.8964321608040201$) |
| `risk_level` | `Optional[str]` | Descriptive tier: `"LOW"`, `"MEDIUM"`, `"HIGH"`, `"CRITICAL"` (`None` on error) |
| `model_version` | `Optional[str]` | Approved model version (`"v3"`) |
| `model_status` | `Optional[str]` | Operational status (`"ACTIVE_LOADED"`) |
| `feature_count` | `Optional[int]` | Expected canonical features ($29$) |
| `reason_codes` | `List[Dict[str, str]]` | Ordered auditable explanation tokens |
| `explanation_metadata` | `Optional[Dict[str, Any]]` | Contextual metadata explaining cutoffs and operational parameters |
| `execution_time_ms` | `Optional[float]` | Inference latency in milliseconds |
| `input_index` | `Optional[int]` | 0-based batch input index (populated in batch mode) |
| `error` | `Optional[Dict[str, str]]` | Error payload `{"code": ..., "message": ...}` (populated when status is `"ERROR"`) |

### Example Serialized JSON Output (Single Prediction)
```json
{
  "status": "SUCCESS",
  "transaction_id": "TX-998821",
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
  "execution_time_ms": 2.451
}
```

---

## 4. Risk Classification Policy & Justification

### Policy Origin & Labeling
A comprehensive codebase audit confirmed that no pre-existing risk-banding policy existed in the repository. As directed by the master specification:
- This risk policy is explicitly defined as a **configurable business policy choice**, labeled `v1.0-provisional`.
- It is **not** a model-derived mathematical property or calibrated real-world posterior probability.

### Cutoffs Definition
| Risk Level | Interval | Default Cutoffs | Operational Meaning |
|:---|:---|:---|:---|
| **LOW** | $[0.0, \text{low\_threshold})$ | $[0.0, 0.20)$ | Normal baseline activity; negligible fraud suspicion. Automated pass. |
| **MEDIUM** | $[\text{low\_threshold}, \text{medium\_threshold})$ | $[0.20, 0.50)$ | Minor risk signals; moderate suspicion. Logged for telemetry. |
| **HIGH** | $[\text{medium\_threshold}, \text{high\_threshold})$ | $[0.50, \text{threshold})$ | Elevated risk. Crucially, falls below the automated block threshold ($\approx 0.8964$), so model decision remains **LEGIT**. Recommended for secondary challenge (e.g. OTP) or manual queue review. |
| **CRITICAL** | $[\text{high\_threshold}, 1.0]$ | $[\text{threshold}, 1.0]$ | Probability meets or exceeds the frozen decision threshold. Model decision is **FRAUD**. Automated block. |

### Anchoring to the Frozen Threshold
When `high_threshold` is not explicitly overridden, `RiskPolicy.resolve_high_threshold()` dynamically anchors the boundary between `HIGH` and `CRITICAL` directly to the model bundle's authoritative frozen threshold ($0.8964321608040201$).

### Boundary Rules
- Intervals are half-open $[a, b)$: the exact boundary cutoff belongs to the higher risk category.
  - $p = 0.20 \implies \text{MEDIUM}$
  - $p = 0.50 \implies \text{HIGH}$
  - $p = 0.8964321608040201 \implies \text{CRITICAL}$
  - $p = 1.0 \implies \text{CRITICAL}$ (inclusive upper endpoint)
- Invalid values (NaN, $\pm\infty$, $p < 0.0$, $p > 1.0$, non-numeric) raise `InvalidProbabilityError`.
- Misordered cutoffs ($c_{\text{low}} \ge c_{\text{med}}$ or $c_{\text{med}} \ge c_{\text{high}}$) raise `InvalidRiskPolicyError`.

---

## 5. Structured Reason Codes

Reason codes provide stable, machine-readable explanation tokens for auditable downstream routing:

| Code | Category | Meaning |
|:---|:---|:---|
| `MODEL_DECISION_FRAUD` | `MODEL_DECISION` | Transaction probability equals or exceeds the frozen fraud decision threshold. |
| `MODEL_DECISION_LEGIT` | `MODEL_DECISION` | Transaction probability is strictly below the frozen fraud decision threshold. |
| `THRESHOLD_EXCEEDED` | `THRESHOLD` | Calculated probability meets or exceeds the frozen threshold. |
| `THRESHOLD_NOT_MET` | `THRESHOLD` | Calculated probability does not meet the frozen threshold. |
| `RISK_LEVEL_LOW` | `RISK_LEVEL` | Probability falls within the low descriptive risk band ($[0.0, 0.20)$). |
| `RISK_LEVEL_MEDIUM` | `RISK_LEVEL` | Probability falls within the medium descriptive risk band ($[0.20, 0.50)$). |
| `RISK_LEVEL_HIGH` | `RISK_LEVEL` | Probability falls within the high descriptive risk band ($[0.50, \text{threshold})$). |
| `RISK_LEVEL_CRITICAL` | `RISK_LEVEL` | Probability falls within the critical descriptive risk band ($[\text{threshold}, 1.0]$). |
| `FEATURE_EXPLANATION_UNAVAILABLE` | `EXPLANATION` | Explicit neutral disclaimer stating feature-level attribution is not enabled. |

### Privacy & Governance Compliance
1. **Zero Synthetic Feature Attribution:** In strict accordance with Phase 6 instructions, feature-level attributions are NOT fabricated or hallucinated.
2. **Zero PII Exposure:** Reason codes and explanation metadata never reveal raw input values, customer identifiers, account numbers, or personal identifying information.

---

## 6. Integration with Phase 4 & Phase 5 Engines

### Single Transaction Inference Engine (`inference_engine.py`)
- Updated `SingleTransactionInferenceEngine` to instantiate and execute `RiskClassifier`.
- Enhanced `PredictionResponse` to include `risk_level`, `reason_codes`, `explanation_metadata`, and `status="SUCCESS"`.
- Backward Compatibility: All 25 Phase 4 unit tests continue to pass without modifications. Existing fields (`prediction`, `fraud_probability`, `threshold`, `model_version`, `model_status`, `feature_count`, `execution_time_ms`) remain unmodified.

### Batch Transaction Inference Engine (`batch_inference_engine.py`)
- Updated `BatchTransactionInferenceEngine` to evaluate risk for every successful vectorized record.
- Enhanced `BatchRecordResult` with `risk_level`, `reason_codes`, and `explanation_metadata`.
- **Partial-Success Preservation:**
  - Successful records receive accurate risk classifications and reason codes.
  - Failed records (missing features, invalid datatypes, corrupt inputs) retain structured error payloads and have `risk_level=None`. **Failed records are NEVER classified as LOW risk.**
- Backward Compatibility: All 32 Phase 5 unit tests continue to pass without modifications.

### Mathematical Single vs. Batch Consistency
Transactions evaluated via single-record inference or vectorized batch inference yield identical:
- Fraud probability (within float precision $< 10^{-12}$)
- Decision (`"FRAUD"` vs `"LEGIT"`)
- Risk level (`"LOW"`, `"MEDIUM"`, `"HIGH"`, `"CRITICAL"`)
- Ordered reason codes
- Frozen threshold ($0.8964321608040201$)

---

## 7. Verification & Test Coverage

### Test Execution Results
The test suite was executed via `pytest`:

```text
pytest phase4_api/tests/ -v
============================= 120 passed in 1.72s =============================
```

### Breakdown by Test Module
| Test Module | Phase Focus | Test Count | Result |
|:---|:---|:---:|:---:|
| `test_model_loader.py` | Phase 2: Model Loading & Validation | 14 | 14 / 14 PASS |
| `test_transaction_preprocessor.py` | Phase 3: Transaction Preprocessing | 14 | 14 / 14 PASS |
| `test_inference_engine.py` | Phase 4: Single Transaction Inference | 25 | 25 / 25 PASS |
| `test_inference_smoke.py` | Phase 4: End-to-End Real Model Smoke | 2 | 2 / 2 PASS |
| `test_batch_inference_engine.py` | Phase 5: Batch Transaction Inference | 32 | 32 / 32 PASS |
| `test_risk_classification.py` | Phase 6: Prediction Result & Risk Classification | 33 | 33 / 33 PASS |
| **Total** | | **120** | **120 / 120 PASS** |

### Member 2 Integration Regression Verification
Member 2's PSI drift integration was executed as a read-only regression check:
```text
pytest experiments/phase4_drift_adaptation/tests/test_member2_phase4_integration.py -v
============================= 11 passed in 0.04s ==============================
```

---

## 8. Real-Model Smoke-Test Verification

`test_33_real_model_smoke_test` validated the Phase 6 pipeline directly against the real LightGBM v3 Champion Adaptive model:
- **Model Artifact:** `experiments/phase4_drift_adaptation/artifacts/models/lgbm_v3.txt`
- **Features Tested:** Exactly 29 canonical features
- **Frozen Threshold Applied:** `0.8964321608040201`
- **Single Inference Output:** Produced valid prediction (`"LEGIT"` or `"FRAUD"`), valid risk level, and $\ge 3$ structured reason codes.
- **Batch Inference Output:** Produced identical probability, decision, and risk tier matching single inference.

---

## 9. Scoped File-Change Report (Folder Isolation)

In strict adherence to Section 1 & Section 15 of the Phase 6 prompt, all file creations and modifications occurred exclusively inside `phase4_api/`.

### Files Created Inside `phase4_api/`:
1. `phase4_api/src/risk_classifier.py` (New: RiskLevel, ReasonCode, RiskPolicy, RiskClassifier)
2. `phase4_api/src/prediction_schema.py` (New: StandardizedPredictionResult, Pydantic PredictionResultModel)
3. `phase4_api/tests/test_risk_classification.py` (New: 33 comprehensive unit tests)
4. `phase4_api/reports/phase6_prediction_result_risk_classification.md` (New: Technical report)

### Files Modified Inside `phase4_api/`:
1. `phase4_api/src/inference_engine.py` (Integrated RiskClassifier, enriched PredictionResponse)
2. `phase4_api/src/batch_inference_engine.py` (Integrated RiskClassifier, enriched BatchRecordResult)
3. `phase4_api/src/__init__.py` (Exported Phase 6 classes and exceptions)
4. `phase4_api/tests/test_inference_engine.py` (Resolved `from typing import Any` import warning)
5. `phase4_api/README.md` (Documented Phase 6 components, schema, and usage)

### External Files Modified:
**ZERO.** No files outside `phase4_api/` were created, modified, staged, committed, or deleted.

---

## 10. Known Limitations & Next Steps

1. **Feature-Level Explainability (SHAP / LIME):** Feature-level importance or SHAP values are not computed during Phase 6 serving to maintain sub-10ms latency. The neutral reason code `FEATURE_EXPLANATION_UNAVAILABLE` explicitly communicates this to downstream clients.
2. **Readiness for Phase 7:** The standardized schema (`PredictionResultModel` and `StandardizedPredictionResult`) provides direct drop-in models for FastAPI endpoints (`/api/v1/predict/single` and `/api/v1/predict/batch`) in Phase 7.
