# Phase 2 — STEP 9: E2b Test Evaluation Report

**Generated:** 2026-09-18  
**Status: STEP 9 COMPLETE — STOP. Awaiting STEP 10 review.**  
**Test population evaluated exactly once. No refitting. No test-driven decisions.**

---

## 1. Protocol Summary

| Rule | Status |
|------|--------|
| E2b checkpoint (epoch 2) loaded, NOT retrained | ✅ |
| StandardScaler loaded, NOT refitted | ✅ |
| Threshold selected on **validation only** (F1-max) | ✅ |
| Test labels used only for final metric computation | ✅ |
| E1 artifacts unmodified | ✅ |
| E2a artifacts unmodified | ✅ |
| Test set evaluated exactly once | ✅ |

---

## 2. Test Population

| Population | Count |
|-----------|-------|
| E1 full test TransactionIDs | 78,542 |
| E2b eligible test sequences | 76,520 |
| **Common (paired comparison)** | **76,520** |
| E1-only (no E2b sequence) | 2,022 |
| E2b coverage | **97.43%** |

**Fraud in common population:** 2,700 fraud / 73,820 legit (3.53%)

The 2,022 E1-only transactions lack sufficient prior history (fewer than 4 causal predecessors within the same card1 entity) and are therefore ineligible for E2b sequence construction. This is a consequence of the locked L=5 window requirement, not a data error.

---

## 3. Authoritative Scaler Statistics (Full Training Fit)

The scaler used for test transformation was fitted on 398,312 × 4 = 1,593,248 training observations. These are the authoritative statistics — the smoke-sample audit (500 entities) in the preprocessing design phase was for informational purposes only.

| Statistic | Full-train value |
|-----------|----------------|
| Scale > 100 (large-scale features) | 66 |
| Scale 1–100 (medium-scale) | 86 |
| Scale < 1 (small/binary/ordinal) | 254 |
| Near-zero-variance (scale set to 1.0) | 1 |
| Mean range | [−306.83, 134,420.22] |
| Scale range | [0.002, 77,694.42] |

Test transformation was applied using these frozen training statistics. No recomputation on validation or test.

---

## 4. Thresholds

| Model | Threshold | Selection Method |
|-------|-----------|-----------------|
| E1 | **0.616521** | Frozen from original E1 experiment (val F1-max) |
| E2b | **0.810221** | Val F1-max, selected on 75,727 validation sequences |

The E2b threshold (0.810221) is notably high, which is consistent with a model that produces relatively low fraud probabilities overall and requires a high cutoff to achieve reasonable precision.

---

## 5. Full Test Metrics — Common 76,520 Population

| Metric | E1 | E2b |
|--------|-----|------|
| **PR-AUC** | **0.5267** | **0.1683** |
| **ROC-AUC** | **0.8981** | **0.7422** |
| F1 | **0.5308** | 0.2432 |
| Precision | **0.5790** | 0.1913 |
| Recall | **0.4900** | 0.3337 |
| MCC | **0.5170** | 0.2166 |
| Balanced Accuracy | **0.7385** | 0.6411 |
| FPR | **0.0130** | 0.0516 |
| FNR | **0.5100** | 0.6663 |

> [!NOTE]
> E1 PR-AUC on the common 76,520 population (0.5267) is slightly lower than the original full-test PR-AUC (0.5317) because 2,022 of the easiest-to-classify transactions (those without sufficient sequence history) are excluded. This is expected and does not represent a change to E1.

---

## 6. Confusion Matrices — Common 76,520 Population

**E1 (threshold = 0.616521):**

|  | Predicted Legit | Predicted Fraud |
|--|----------------|----------------|
| **Actual Legit** (73,820) | TN = 72,858 | FP = 962 |
| **Actual Fraud** (2,700) | FN = 1,377 | TP = 1,323 |

**E2b (threshold = 0.810221):**

|  | Predicted Legit | Predicted Fraud |
|--|----------------|----------------|
| **Actual Legit** (73,820) | TN = 70,011 | FP = 3,809 |
| **Actual Fraud** (2,700) | FN = 1,799 | TP = 901 |

---

## 7. Operational Metrics — Common 76,520 Population

| Metric | E1 | E2b |
|--------|-----|------|
| Recall @ 0.1% FPR | 0.2081 | 0.0215 |
| Recall @ 0.5% FPR | 0.3770 | 0.0985 |
| Recall @ 1.0% FPR | 0.4578 | 0.1441 |
| Recall @ 2.0% FPR | 0.5259 | 0.2119 |
| Precision @ Top-100 | **0.9800** | 0.4800 |
| Precision @ Top-500 | **0.9040** | 0.4280 |
| Precision @ Top-1,000 | **0.8310** | 0.3650 |
| Precision @ Top-5,000 | **0.3362** | 0.1850 |

E1 substantially outperforms E2b across all operational metrics.

---

## 8. Calibration

| Metric | E1 | E2b |
|--------|-----|------|
| Brier Score | **0.0324** | 0.2377 |
| ECE (10 bins) | **0.0523** | 0.4247 |

E1 is well-calibrated (low Brier, moderate ECE). E2b is poorly calibrated (ECE = 0.4247), indicating its predicted probabilities are not reliable estimates of the true fraud rate. The high ECE is consistent with the high threshold (0.810221) required to operate E2b at any reasonable precision level.

---

## 9. Paired Bootstrap — PR-AUC (2,000 resamples, seed=42)

**Primary metric: Δ PR-AUC = PR-AUC(E2b) − PR-AUC(E1)**

| Quantity | Value |
|---------|-------|
| Observed E1 PR-AUC | 0.5267 |
| Observed E2b PR-AUC | 0.1683 |
| Observed Δ PR-AUC | **−0.3584** |
| Bootstrap mean Δ | − |
| **95% CI for Δ** | **[−0.3756, −0.3395]** |
| CI excludes zero | **Yes** |
| Valid resamples | 2,000 / 2,000 |

> [!IMPORTANT]
> The 95% paired bootstrap CI for Δ PR-AUC = [−0.3756, −0.3395] lies entirely below zero. The paired bootstrap consistently finds E1 superior to E2b on PR-AUC across all 2,000 resamples of the 76,520 common test transactions. This does not establish a causal explanation but provides a statistically consistent empirical finding.

---

## 10. Research Question Response

> *"Does historical transaction behavior provide additional predictive value beyond the tabular LightGBM baseline (E1)?"*

**Empirical finding from the test evaluation:**

The E2b GRU (StandardScaler-preprocessed, 4-step history, 406 features, locked architecture) achieves a test PR-AUC of **0.1683** on the 76,520 common test transactions, compared to E1's test PR-AUC of **0.5267** on the same population.

The 95% paired bootstrap CI for Δ PR-AUC = [−0.3756, −0.3395] excludes zero entirely.

Under the locked experimental conditions, the E2b GRU does not provide additional predictive value over the tabular LightGBM baseline. On the contrary, E1 substantially outperforms E2b on every reported metric.

**This is a legitimate and informative research finding.** It does not invalidate the approach of temporal sequence modeling — it documents what a specific locked GRU configuration achieves relative to a strong tabular baseline on this dataset and feature set.

---

## 11. Artifacts

| Artifact | Path |
|---------|------|
| Test predictions (parquet) | `phase2_gru/artifacts/E2b_scaled/test_predictions.parquet` |
| Test metrics (JSON) | `phase2_gru/artifacts/E2b_scaled/test_metrics.json` |
| Paired bootstrap (JSON) | `phase2_gru/artifacts/E2b_scaled/paired_bootstrap.json` |
| Common IDs (txt) | `phase2_gru/artifacts/E2b_scaled/common_test_transaction_ids.txt` |
| Integrity audit | `phase2_gru/reports/STEP9_integrity_audit.json` |

---

*Phase 2 STEP 9 — Completed 2026-09-18 | E1 and E2a unmodified | Single test evaluation*
