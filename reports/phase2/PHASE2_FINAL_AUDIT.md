# PHASE 2 — FINAL RESEARCH AUDIT + IMMUTABLE FREEZE
**STEP 10 | Generated:** 2026-09-18  
**Audit type:** Read-only verification of all Phase 2 artifacts, metrics, and methodology  
**Phase 2 status: 🔒 FROZEN**

---

## Audit Summary

| Category | Result |
|----------|--------|
| E1 canonical metrics | ✅ PASS — all 9 values match exactly |
| Common-population metrics (76,520) | ✅ PASS — all 20 E1+E2b values match |
| Operational metrics (8 checks) | ✅ PASS — all match |
| Calibration values | ✅ PASS — verified |
| Paired bootstrap | ✅ PASS — delta and CI verified |
| Artifact hash verification | ✅ PASS — 4 SHA256 hashes MATCH |
| E2b training configuration | ✅ PASS — all locked parameters verified |
| E2a configuration | ✅ PASS — best epoch 1, val PR-AUC 0.0345 |
| Test suite | ✅ PASS — 80/80 |
| Common IDs file | ✅ PASS — 76,520 entries |
| Known warnings documented | ✅ 2 warnings recorded |

**All audits passed. Phase 2 is formally frozen.**

---

## 1. Immutable E1 Reference

E1 is frozen. Verified by SHA256.

**E1 predictions.parquet:** `6a73279d81491437` — **MATCH**  
**E1 metrics.json:** `6f3ab9c86f9b9f47` — **MATCH**

### E1 Canonical Test Metrics (full 78,542 transactions)

| Metric | Value |
|--------|-------|
| PR-AUC | **0.531731** |
| ROC-AUC | 0.898993 |
| Precision | 0.581810 |
| Recall | 0.493511 |
| F1 | 0.534035 |
| MCC | 0.520301 |
| Balanced Accuracy | 0.740262 |
| FPR | 0.012987 |
| FNR | 0.506489 |
| Threshold (frozen) | **0.616521** |

These values must not be modified. They are the anchoring baseline for all Phase 2 comparisons.

---

## 2. E2a Audit

| Property | Value |
|---------|-------|
| Description | Unscaled GRU — control variant |
| Architecture | FraudGRU(input=406, hidden=64, layers=1, dropout=0.2) |
| Input shape | [batch, 4, 406] |
| Preprocessing | **None** (raw E1 features, unscaled) |
| Seed | 42 |
| Optimizer | Adam, lr=1e-3 |
| Best epoch | 1 |
| Best val PR-AUC | **0.0345** |
| Val fraud prevalence | ~0.0339 |
| Interpretation | Near-random validation performance |
| Test evaluated | No |
| Artifact dir | `phase2_gru/artifacts/E2a_unscaled/` |
| SHA256 (checkpoint) | *(captured in freeze manifest)* |

E2a serves as the documented control experiment demonstrating the validation behavior of the locked GRU architecture when applied to unscaled E1 features.

---

## 3. E2b Audit

### Architecture (verified)

| Parameter | Value |
|---------|-------|
| input_size | **406** |
| hidden_size | **64** |
| num_layers | **1** |
| dropout | **0.2** |
| seq_len | **4** |
| Output | Linear(64 → 1) logits |
| Trainable parameters | 90,689 |

### Training Configuration (verified)

| Parameter | Value |
|---------|-------|
| Seed | **42** |
| Optimizer | **Adam** |
| Learning rate | **0.001** |
| Batch size | **128** |
| Max epochs | **30** |
| Patience | **5** |
| Loss | **BCEWithLogitsLoss** |
| pos_weight | **27.47** |
| Monitor | validation PR-AUC |
| SMOTE | No |
| Random oversample | No |

### Preprocessing (verified)

| Property | Value |
|---------|-------|
| Scaler | StandardScaler |
| Fit population | Training sequences only (398,312 × 4 = 1,593,248 obs) |
| Val transformation | Frozen train statistics |
| Test transformation | Frozen train statistics |
| SHA256 (scaler) | **fe294184277da80f — MATCH** |

### Training Results

| Property | Value |
|---------|-------|
| Best epoch | **2** |
| Best val PR-AUC | **0.1377** |
| Epochs completed | 7 / 30 |
| Early stopped | Yes (patience=5) |
| Overfitting observed | Yes — val loss rising after epoch 2 |
| GPU | NVIDIA RTX 5050 Laptop (CUDA 13.2) |
| Training time | 78.7 sec |

**SHA256 (checkpoint epoch 2):** `31b17bb46d2c9257` — **MATCH**

---

## 4. Sequence Generation Audit

### Locked Specification (v1.1)

| Parameter | Value |
|---------|-------|
| Entity key | `card1` (grouping only, not in GRU input) |
| Min entity size | 5 |
| Window length | 5 (T1–T4 history + T5 target) |
| Stride | 1 (sliding) |
| Temporal ordering | Strictly ascending `TransactionDT` |
| Duplicate DT treatment | **Option A: exclude windows where DT(T4) == DT(T5)** |
| Windows excluded | **585** |
| Gap feature | `log1p(DT(T5) − DT(T4))` at index 405 |
| Target gap in input | **No** |
| Split assignment | By target transaction (T5) split |
| Cross-split history | Allowed (prior splits only) |

### Verified Leakage Controls

| Control | Status |
|---------|--------|
| `isFraud` absent from X | ✅ |
| Target TransactionID absent from history | ✅ |
| All history DTs < target DT | ✅ |
| No future transactions in history | ✅ |
| History sorted ascending | ✅ |
| card1 absent from GRU input | ✅ |
| T5 gap not used as feature | ✅ |

### Dataset Counts (verified)

| Split | Sequences | Fraud | Legit |
|-------|----------|-------|-------|
| Train | **398,312** | 14,397 | 383,915 |
| Validation | **75,727** | 2,567 | 73,160 |
| Test | **76,520** | 2,700 | 73,820 |

Duplicate DT excluded: **585** | Entities with windows: **6,511** / 13,553 total

---

## 5. STEP 9 Final Evaluation Audit

### Population

| Quantity | Value |
|---------|-------|
| E1 full test | 78,542 |
| E2b eligible test | 76,520 |
| **Common (paired)** | **76,520** |
| E1-only (excluded) | 2,022 |
| E2b coverage | **97.43%** |
| Common fraud | 2,700 |
| Common legit | 73,820 |
| Label alignment | **VERIFIED IDENTICAL** |

### Threshold Audit

| Model | Threshold | Method |
|-------|-----------|--------|
| E1 | **0.616521** | Frozen (Phase 1 val F1-max) |
| E2b | **0.810221** | Val F1-max — no test labels used |

### Final Test Metrics — Common 76,520 Population (verified)

| Metric | E1 | E2b |
|--------|-----|------|
| **PR-AUC** | **0.5267** | 0.1683 |
| ROC-AUC | 0.8981 | 0.7422 |
| Precision | 0.5790 | 0.1913 |
| Recall | 0.4900 | 0.3337 |
| F1 | 0.5308 | 0.2432 |
| MCC | 0.5170 | 0.2166 |
| Balanced Accuracy | 0.7385 | 0.6411 |
| Brier | 0.0324 | 0.2377 |
| ECE | 0.0523 | 0.4247 |
| FPR | 0.0130 | 0.0516 |
| FNR | 0.5100 | 0.6663 |

### Operational Metrics — Common 76,520 (verified)

| Metric | E1 | E2b |
|--------|-----|------|
| Recall @ 0.1% FPR | 0.2081 | 0.0215 |
| Recall @ 0.5% FPR | 0.3770 | 0.0985 |
| Recall @ 1.0% FPR | 0.4578 | 0.1441 |
| Recall @ 2.0% FPR | 0.5259 | 0.2119 |
| Precision @ Top-100 | 0.9800 | 0.4800 |
| Precision @ Top-500 | 0.9040 | 0.4280 |
| Precision @ Top-1,000 | 0.8310 | 0.3650 |
| Precision @ Top-5,000 | 0.3362 | 0.1850 |

### Calibration (verified)

| Model | Brier Score | ECE (10-bin) |
|-------|------------|-------------|
| E1 | 0.0324 | 0.0523 |
| E2b | 0.2377 | 0.4247 |

E1 has a lower Brier score and lower ECE than E2b on the common test population. E2b's higher ECE is consistent with the high operating threshold (0.810221) required to achieve any precision, indicating its predicted probabilities are less well-calibrated as direct probability estimates.

---

## 6. Paired Bootstrap Audit

**Convention: Δ = PR-AUC(E2b) − PR-AUC(E1)**

| Quantity | Value |
|---------|-------|
| Resamples | 2,000 |
| Seed | 42 |
| Population | 76,520 common TransactionIDs |
| Observed Δ PR-AUC | **−0.3584** |
| 95% CI | **[−0.3756, −0.3395]** |
| CI excludes zero | **Yes** |
| Valid resamples | 2,000 / 2,000 |

**Secondary (ROC-AUC):** Δ = −0.1559, 95% CI [−0.16715940175849875, −0.14454939181321735], excludes zero.

> The 95% paired bootstrap CI for Δ PR-AUC lies entirely below zero under the specified paired resampling procedure on the 76,520 common test TransactionIDs. This does not establish causality.

---

## 7. E2a → E2b Interpretation

E2a validation PR-AUC: 0.0345 (≈ fraud prevalence of 3.39%)  
E2b validation PR-AUC: 0.1377

Scaling was associated with a substantial change in validation performance between the two tested variants. E2a and E2b differ only in whether the 406 input features are standardized before being fed to the GRU. E2b's higher validation PR-AUC is empirically observed, but the experiment design does not formally isolate scaling as the sole factor — other aspects of the optimization landscape also change between scaled and unscaled inputs.

**This report does not claim that scaling caused the improvement.**

---

## 8. E1 → E2b Interpretation

Under the locked E2b configuration, standalone temporal GRU modeling did not demonstrate additional predictive value over the frozen tabular LightGBM baseline on the common test population. The 95% paired bootstrap CI [−0.3756, −0.3395] excludes zero, and the direction is consistently negative across all 2,000 resamples.

---

## 9. What Phase 2 Establishes

Phase 2 establishes evidence about the following specific configuration:

- A 4-step sliding history window grouped by card1
- The 406-feature representation derived from E1 (405 retained + 1 log1p gap)
- train-fitted StandardScaler applied to all 406 features
- A single-layer GRU (hidden=64, dropout=0.2)
- Adam optimization with lr=1e-3, pos_weight=27.47
- The IEEE-CIS transaction population

Under these conditions, the standalone GRU did not outperform the frozen tabular baseline on the common test population.

---

## 10. What Phase 2 Does NOT Establish

Phase 2 does **not** establish that:

- Temporal modeling is useless for fraud detection
- GRUs are unsuitable for fraud detection in general
- Longer history windows cannot provide additional signal
- Attention mechanisms or Transformers cannot improve performance
- Temporal features cannot complement tabular features in a hybrid model
- Another sequence architecture cannot outperform E1
- Temporal modeling cannot help under concept drift
- The E1 features are an ideal representation for sequence input

These remain open research questions, which Phase 3 (E3 Hybrid) and later phases are designed to investigate.

---

## 11. Limitations

1. E2b uses only four historical transactions (window length 4). Longer histories may contain different signal.
2. 585 duplicate-TransactionDT windows were excluded (Option A). These ~0.11% of sequences are methodologically excluded, not lost data.
3. E2b test population (76,520) is 97.43% of E1's test population. The 2,022 E1-only transactions lacked sufficient causal history for sequence construction.
4. The GRU architecture is intentionally simple (single layer, 64 units) and was not tuned.
5. The 406 input features were engineered for LightGBM and may not be an ideal representation for recurrent sequence modeling.
6. Some IEEE-CIS V-features are opaque/anonymized, limiting interpretability of what temporal patterns the GRU could detect.
7. Results apply to IEEE-CIS and the specific locked configuration — generalization to other datasets is not established here.
8. The E2b threshold RuntimeWarning is documented but not corrected (experiment frozen).

---

## 12. Known Warnings

### W1 — RuntimeWarning in threshold selection
- **Location:** `phase2_gru/src/evaluation/run_step9.py` — `f1_max_threshold()`
- **Warning:** `RuntimeWarning: invalid value encountered in divide`
- **Cause:** `sklearn.metrics.precision_recall_curve` produces 0/0 at extreme thresholds; numpy handles the resulting NaN as 0 in F1 computation
- **Effect:** None — the selected threshold (0.810221) and all test metrics are unaffected
- **Action:** Not corrected — experiment is frozen

### W2 — SMOTE/oversample fields in run_metadata.json
- **Location:** `phase2_gru/artifacts/E2b_scaled/run_metadata.json`
- **Warning:** `SMOTE` and `random_oversample` stored as `null` instead of `false`
- **Cause:** Python `None` serialized to JSON `null` rather than `false`
- **Effect:** None — no oversampling was used in either case
- **Action:** Not corrected — experiment is frozen

---

## 13. Artifact Hash Audit

| Artifact | SHA-256 (first 16 hex) | Status |
|---------|------------------------|--------|
| E1 `predictions.parquet` | `6a73279d81491437` | ✅ MATCH |
| E2b `gru_best.pt` (epoch 2) | `31b17bb46d2c9257` | ✅ MATCH |
| E2b `standard_scaler.pkl` | `fe294184277da80f` | ✅ MATCH |
| E1 `metrics.json` | `6f3ab9c86f9b9f47` | ✅ MATCH |

Full SHA-256 digests captured in `reports/phase2/phase2_freeze_manifest.json`.

---

## 14. Test Suite Audit

| Module | Tests | Category |
|--------|-------|---------|
| `test_gru_architecture.py` | AT1–AT15 | Architecture integrity |
| `test_label_isolation.py` | LT3 (8 tests) | Label isolation |
| `test_preprocessing.py` | PT1–PT8 (22 tests) | Scaler train-only fitting |
| `test_sequence_leakage.py` | LT1, LT6, LT8 (9 tests) | Temporal leakage |
| `test_split_integrity.py` | LT7 (6 tests) | Split contamination |
| `test_temporal_order.py` | LT2, LT4, LT5 (7 tests) | Causal ordering |
| **Total** | **80 / 80 PASSED** | |

---

## 15. Phase 2 Research Handoff to Phase 3

**Phase 2 finding:**  
Under the locked E2b configuration, the standalone temporal GRU did not demonstrate additional predictive value over the frozen tabular LightGBM baseline on the IEEE-CIS common test population (Δ PR-AUC = −0.3584, 95% CI [−0.3756, −0.3395]).

**Phase 3 research question:**  
*"Does temporal information provide complementary signal when combined with the strong current-transaction tabular representation?"*

**Next experiment:** E3 Hybrid (IEEE-CIS) — GRU temporal features combined with E1 tabular features.

**Status:** WAITING — requires explicit approval before any implementation begins.

**Constraint:** E1, E2a, and E2b artifacts are immutable. Phase 3 must build on them without modification.

---

## 16. Phase 2 Freeze Manifest

**Location:** `reports/phase2/phase2_freeze_manifest.json`

Contains:
- Experiment identifiers
- All artifact paths with full SHA-256 hashes
- Dataset identifiers and counts
- Sequence specification version 1.1
- Full training configurations
- Final metrics (E1 canonical, E1 common, E2b common)
- Bootstrap configuration and results
- Test protocol and results
- Known warnings (W1, W2)
- Limitations
- Freeze timestamp
- Immutability declarations for all E1/E2 artifacts

---

## 17. Final Phase 2 Status

```
PHASE 2 STATUS = 🔒 FROZEN

E1  LightGBM baseline          ✅ Immutable
E1.5 Phase 1.5 hardening      ✅ Immutable
E2a Unscaled GRU (control)    ✅ Immutable
E2b Scaled GRU (primary)      ✅ Immutable
STEP 9 Test evaluation         ✅ Complete — single test evaluation
STEP 10 Final audit + freeze   ✅ COMPLETE — THIS DOCUMENT

All 80 tests:                  ✅ PASS
All artifact hashes:           ✅ VERIFIED
All metrics:                   ✅ CONSISTENT
Known warnings:                ✅ DOCUMENTED (2)
```

---

*Phase 2 Final Audit — STEP 10 — 2026-09-18*  
*No training. No model changes. No artifact modifications.*  
*Freeze manifest: `reports/phase2/phase2_freeze_manifest.json`*
