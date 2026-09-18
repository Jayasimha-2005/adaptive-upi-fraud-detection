# COMPLETE RESEARCH STATUS REVIEW
## Adaptive Financial Fraud Detection — From Project Start to Current State

**Generated:** 2026-09-18  
**Scope:** Full audit, analysis, and honest assessment of the complete research program  
**Immutability:** This document is read-only with respect to all existing experiments and artifacts

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Chronological Project Timeline](#2-chronological-project-timeline)
3. [Research Question Analysis](#3-research-question-analysis)
4. [Dataset Complete Audit](#4-dataset-complete-audit)
5. [Are the Datasets Sufficient?](#5-are-the-datasets-sufficient)
6. [Phase 1 — Complete Explanation](#6-phase-1--complete-explanation)
7. [Phase 1 Results](#7-phase-1-results)
8. [Phase 1 Research Value](#8-phase-1-research-value)
9. [Phase 2 — Complete Explanation](#9-phase-2--complete-explanation)
10. [E2a Explained](#10-e2a-explained)
11. [E2b Explained](#11-e2b-explained)
12. [STEP 9 — Final Evaluation](#12-step-9--final-evaluation)
13. [Phase 2 Results](#13-phase-2-results)
14. [Phase 1 vs Phase 2 Comparison](#14-phase-1-vs-phase-2-comparison)
15. [What Phase 2 Established vs. Left Open](#15-what-phase-2-established-vs-left-open)
16. [Methodological Issues Audit](#16-methodological-issues-audit)
17. [Phase 2 Freeze Validation](#17-phase-2-freeze-validation)
18. [Research-Level Assessment](#18-research-level-assessment)
19. [Gap Analysis — What is Still Missing](#19-gap-analysis--what-is-still-missing)
20. [Future Experiment Map](#20-future-experiment-map)
21. [E3 Scientific Justification](#21-e3-scientific-justification)
22. [Final Honest Verdict](#22-final-honest-verdict)

---

## 1. Project Overview

### Title (from README and reports)
**Adaptive Financial Fraud Detection with Temporal Modeling, Concept Drift Adaptation, Explainable AI, and Real-Time Stream Processing**

### Problem Statement
Financial fraud detection systems face a fundamental challenge: fraud patterns evolve over time (concept drift), making static models gradually less effective. The research investigates whether temporal behavioral modeling, combined with strong tabular baselines and adaptive mechanisms, can produce more robust fraud detection systems than current practice.

### Research Motivation
- Financial fraud causes significant economic harm globally
- Production fraud detection systems typically use static tabular models that degrade under distributional shift
- Temporal sequence models (RNNs, Transformers) have shown promise in related domains but lack systematic evaluation against strong tabular baselines on established fraud benchmarks with proper leakage controls
- The research targets this methodological gap: rigorous comparison of temporal and tabular approaches under controlled experimental conditions, with explicit leakage prevention

### Research Gap Being Addressed
Most published fraud detection studies either:
(a) compare architectures without rigorous temporal leakage prevention, or
(b) evaluate temporal models without first establishing a strong tabular baseline

This project targets both: establishing a hardened tabular baseline (E1), then fairly comparing temporal approaches under locked conditions.

---

## 2. Chronological Project Timeline

### Stage 1 — Project Conception and Dataset Investigation
**What:** Identified research direction, selected primary dataset (IEEE-CIS), analyzed all available datasets (IEEE-CIS, BAF, PaySim, ULB, 10K toy).  
**Why:** Needed to establish experimental foundation and confirm dataset suitability.  
**Research question answered:** Which datasets are available and appropriate for which experimental phases?  
**Evidence obtained:** Dataset forensics (rows, fraud rates, entity identifiers, temporal coverage) documented in `dataset_analysis/`.  
**Remains unresolved:** Cross-dataset generalization not yet empirically tested.

### Stage 2 — Phase 1: E1 LightGBM Tabular Baseline
**What:** Built the primary tabular fraud detection model on IEEE-CIS using 406 engineered features, chronological train/val/test split, LightGBM with early stopping, and validation-only threshold selection.  
**Why:** Needed a strong, methodologically rigorous tabular baseline against which all temporal models would be compared.  
**Research question answered:** What is the best achievable performance from per-transaction tabular features on IEEE-CIS, under strict leakage prevention?  
**Evidence obtained:** PR-AUC = 0.531731, ROC-AUC = 0.898993, F1 = 0.534035 on 78,542 test transactions.  
**Remains unresolved:** Whether temporal information can complement tabular features.

### Stage 3 — Phase 1.5: Robustness Hardening
**What:** Applied paired bootstrap (2,000 resamples, seed=42) to E1 test metrics, computed operational metrics (recall@FPR, precision@Top-K), calibration (Brier score, ECE), encoding comparison audit.  
**Why:** A single point estimate is insufficient for rigorous research. Bootstrap CIs quantify estimation uncertainty. Operational metrics reflect real-world utility.  
**Research question answered:** How stable are E1's performance estimates? Is E1 well-calibrated?  
**Evidence obtained:** PR-AUC 95% CI [0.5126, 0.5504]; Brier = 0.0323; ECE = 0.0524; Precision@Top-100 = 0.98.  
**Remains unresolved:** Concept drift performance, streaming latency, production robustness.

### Stage 4 — Phase 2 Sequence Specification
**What:** Designed and locked the sequence construction methodology (v1.1): card1 entity grouping, L=5 window (4 history + 1 target), stride=1, strict chronological ordering, duplicate TransactionDT handling (Option A: exclude), gap feature, 406 input features.  
**Why:** Temporal modeling requires precise methodology to avoid temporal leakage. All design decisions locked before data was touched.  
**Research question answered:** How should IEEE-CIS transaction sequences be constructed to give a GRU a fair test while preventing all forms of temporal leakage?  
**Evidence obtained:** 33/33 leakage tests passed, 585 duplicate-DT windows excluded, 550,559 eligible windows from 6,511 entities.  
**Remains unresolved:** Whether L=4 is optimal; whether card1 is the best entity definition.

### Stage 5 — E2a: Unscaled GRU (Control)
**What:** Trained the locked GRU architecture (406→64→1, 90,689 parameters) on unscaled E1 features as sequences. Early stopping after 6 epochs.  
**Why:** Needed to determine whether the GRU could learn from raw (unscaled) E1 features before introducing any preprocessing changes.  
**Research question answered:** Does the locked GRU architecture learn useful representations from unscaled E1 features?  
**Evidence obtained:** Best validation PR-AUC = 0.0345 (epoch 1) — approximately equal to the training fraud prevalence (~3.39%), indicating near-random performance.  
**Remains unresolved:** Whether the poor performance was due to scale sensitivity or other factors.

### Stage 6 — E2b: Scaled GRU (Primary Experiment)
**What:** Applied train-only StandardScaler to all 406 features, retrained the same locked architecture. Best checkpoint at epoch 2, early stopping at epoch 7.  
**Why:** Neural networks are sensitive to feature scale; the near-random E2a result suggested scaling was a prerequisite for learning.  
**Research question answered:** Does standardizing input features enable the locked GRU to learn meaningful representations?  
**Evidence obtained:** Validation PR-AUC improved from 0.0345 to 0.1377 — a substantial change associated with the preprocessing difference.  
**Remains unresolved:** Whether other architecture or hyperparameter changes could further improve performance.

### Stage 7 — STEP 9: Final Test Evaluation
**What:** First (and only) evaluation of E2b on the test set. Computed full metrics on 76,520 common TransactionIDs, ran 2,000 paired bootstrap resamples (seed=42).  
**Why:** The test set must be evaluated only once; validation cannot substitute for unbiased final evaluation.  
**Research question answered:** Under the locked E2b configuration, how does standalone temporal GRU modeling compare to the frozen E1 tabular baseline on the common test population?  
**Evidence obtained:** E2b PR-AUC = 0.1683; E1 PR-AUC = 0.5267 (common); Δ = −0.3584; 95% CI [−0.3756, −0.3395].  
**Remains unresolved:** Whether temporal features can provide complementary signal when combined with tabular features (E3).

### Stage 8 — STEP 10: Final Phase 2 Audit + Freeze
**What:** Verified all artifacts (SHA-256 hashes), confirmed metric consistency, documented known warnings, and formally froze Phase 2.  
**Why:** Experimental integrity requires documented provenance before beginning the next phase.  
**Evidence obtained:** All 4 SHA-256 hashes verified; all 20 common-population metrics consistent; 80/80 tests pass.  
**Remains unresolved:** Everything in Phase 3 and beyond.

### Stage 9 — Dataset Suitability Audit (Parallel Track)
**What:** Forensic inspection of all 5 datasets (IEEE-CIS, BAF, PaySim, ULB, 10K toy). Role assignment and experiment impact analysis.  
**Why:** Addition of ULB required verification that it differs from existing datasets and confirmation that it does not require changes to locked experiments.  
**Evidence obtained:** ULB confirmed as independent (284,807 rows, 48h coverage, no entity ID); BAF identified as primary concept drift asset (6×1M rows, 8 ordinal months, purpose-designed drift variants); 10K toy excluded from primary research.

---

## 3. Research Question Analysis

### Current Main Research Problem
How can fraud detection systems be made more robust to concept drift and temporal pattern change, combining temporal behavioral modeling, explainability, and adaptive learning?

### Specific Sub-questions (in experimental order)

| # | Research Question | Experiment | Status |
|---|-----------------|-----------|--------|
| RQ1 | What does a rigorous tabular baseline achieve on IEEE-CIS under strict leakage prevention? | E1 | ✅ Answered |
| RQ2 | How stable and operationally useful are E1's performance estimates? | E1.5 | ✅ Answered |
| RQ3 | Does historical transaction behavior provide additional predictive value beyond E1? | E2a/E2b | ✅ Answered (for tested config) |
| RQ4 | Does temporal information provide complementary signal when combined with tabular features? | E3 (planned) | ⏳ Open |
| RQ5 | How do fraud detection models perform across different fraud domains? | Cross-dataset | ⏳ Open |
| RQ6 | How does performance degrade under concept drift? | BAF variants | ⏳ Open |
| RQ7 | Can adaptive retraining mitigate concept drift? | Adaptive (planned) | ⏳ Open |
| RQ8 | Can the system operate in real-time streaming conditions? | Kafka/Spark | ⏳ Open |

### Hypothesis Status

**H1 (Tested, partially supported):** Temporal sequence features contain additional fraud signal beyond per-transaction features.  
*Status:* Under the E2b configuration, standalone temporal modeling did not outperform E1. The hybrid question (E3) remains open.

**H2 (Untested):** Temporal models are more robust to concept drift than static tabular models.  
*Status:* Untested — requires BAF drift experiments.

**H3 (Untested):** Adaptive retraining can recover performance under distributional shift.  
*Status:* Untested — requires drift experiments.

---

## 4. Dataset Complete Audit

### Dataset 1 — IEEE-CIS (Primary Benchmark)

| Property | Value |
|---------|-------|
| Full name | IEEE-CIS Fraud Detection (Vesta Corporation / Kaggle competition) |
| Source | Kaggle competition dataset, originally from Vesta Corporation |
| Provenance | Real-world e-commerce transaction records, anonymized/obfuscated |
| Type | Real-world-derived, anonymized, historical |
| Rows | 590,540 total (train + val + test after chronological split) |
| Columns raw | ~434 (transaction + identity joined) |
| Columns retained | **406** (after E1 preprocessing) |
| Target | `isFraud` (0/1) |
| Fraud count | 20,663 across all splits (~3.53%) |
| Legit count | 569,877 |
| Fraud rate | ~3.53% |
| Temporal coverage | ~6 months (TransactionDT in seconds) |
| Temporal resolution | Second-level |
| Entity identifier | `card1` (anonymized card-level ID) |
| Missing values | Extensive — 60+ features have NaN |
| Duplicate rows | None in processed splits |
| V-features | V1–V339 — Vesta-proprietary engineered features (NOT PCA) |
| Categorical columns | ProductCD, card4, card6, P_emaildomain, R_emaildomain, DeviceType, id_12–id_38 |
| Temporal leakage risk | TransactionDT, TransactionID (excluded from features) |
| Tabular ML suitability | ✅ Excellent (large, rich, well-studied) |
| Temporal ML suitability | ✅ Good (entity ID, second-level timestamps, 6-month window) |
| Concept drift suitability | ⚠️ Limited (single 6-month window, no explicit drift labels) |
| Adaptive retraining | ⚠️ Possible but temporal window short |
| Streaming suitability | ✅ Good (TransactionDT enables ordered replay) |
| External validation role | Primary benchmark, not for external validation |

### Dataset 2 — ULB Credit Card Fraud

| Property | Value |
|---------|-------|
| Full name | Credit Card Fraud Detection — ULB Machine Learning Group |
| Source | Kaggle / ULB, European credit card transactions September 2013 |
| Type | Real-world-derived, anonymized (PCA-transformed) |
| Rows | 284,807 (unique: 283,726) |
| Columns | 31 (Time, V1–V28, Amount, Class) |
| Target | `Class` (0=legit, 1=fraud) |
| Fraud count | **492** |
| Legit count | 284,315 |
| Fraud rate | **0.1727%** (578:1 ratio — extreme imbalance) |
| Temporal coverage | **48 hours exactly** |
| Temporal resolution | Second-level (Time column, seconds elapsed) |
| Entity identifier | **NONE** |
| Missing values | 0 |
| Duplicate rows | 1,081 exact full-row duplicates |
| V-features | V1–V28 — **PCA principal components** (NOT equivalent to IEEE-CIS V features) |
| Categorical columns | None (all PCA-transformed) |
| Temporal leakage risk | Low (no entity ID means no grouping; Time is already sorted) |
| Tabular ML suitability | ✅ Excellent — well-known benchmark, extreme imbalance test |
| Temporal ML suitability | ❌ Not applicable — no entity ID for grouping |
| Concept drift suitability | ❌ Insufficient — 48 hours only |
| Adaptive retraining | ❌ Insufficient temporal window |
| Streaming suitability | ⚠️ Marginal — Time enables ordered replay |
| External validation role | ✅ **Primary role** — extreme imbalance, different domain |

### Dataset 3 — BAF (Bank Account Fraud, NeurIPS 2022)

| Property | Value |
|---------|-------|
| Full name | Bank Account Fraud Dataset Suite — Featurespace / NeurIPS 2022 |
| Source | NeurIPS 2022 competition, Featurespace |
| Type | **Realistic synthetic** (statistically designed to match real patterns) |
| Rows | **6,000,000** (6 files × 1,000,000) |
| Columns | 32 per file |
| Target | `fraud_bool` (0/1) |
| Fraud per file | 11,029–11,030 |
| Fraud rate | ~1.10% per file |
| Temporal coverage | 8 ordinal months (`month` column, 0–7) |
| Temporal resolution | Month-level |
| Entity identifier | **NONE** |
| Fraud rate trend | Rising from ~0.87% (month 0) to ~1.47% (month 7) — embedded drift signal |
| Variant design | Base + 5 drift variants (each simulates a different type of distributional shift) |
| Missing values | 0 (sample check) |
| V-features | None (32 directly interpretable features) |
| Notable columns | velocity_6h, velocity_24h, velocity_4w (pre-aggregated temporal signals) |
| Tabular ML suitability | ✅ Excellent |
| Temporal ML suitability | ❌ No entity ID for traditional sequence grouping |
| Concept drift suitability | ✅ **Primary role** — purpose-designed for concept drift benchmarking |
| Adaptive retraining | ✅ **Primary role** — 8-month ordinal windows support retraining experiments |
| Streaming suitability | ✅ Good — month ordering supports streaming simulation |

### Dataset 4 — PaySim (Simulated Mobile Money)

| Property | Value |
|---------|-------|
| Full name | PaySim Synthetic Financial Dataset |
| Source | Agent-based simulation of MPESA-like mobile money (academic) |
| Type | **Simulated** (rule-based agent simulation) |
| Rows | **6,362,620** |
| Columns | 11 |
| Target | `isFraud` (0/1) |
| Fraud count | 8,213 (0.1291%) |
| Legit count | 6,354,407 |
| Temporal coverage | 743 simulated hours (~31 simulated days) |
| Temporal resolution | Hour-level (`step`) |
| Entity identifier | `nameOrig` (sender account) |
| Missing values | 0 |
| Fraud present in | CASH_OUT and TRANSFER only (by simulation design) |
| V-features | None |
| Tabular ML suitability | ✅ Good |
| Temporal ML suitability | ⚠️ Possible — `nameOrig` entity + `step` time, but simulation artifacts present |
| Concept drift suitability | ⚠️ Limited (31 simulated days, simulation not reality) |
| Adaptive retraining | ⚠️ Limited — short window |
| Streaming suitability | ✅ **Strong role** — `step` column enables realistic streaming replay |

### Dataset 5 — "Credit Card Fraud Detection" (10K toy)

| Property | Value |
|---------|-------|
| Full name | Unknown — likely a Kaggle tutorial/demo dataset |
| Source | Unknown |
| Type | **Synthetic toy** — not a published benchmark |
| Rows | 10,000 |
| Columns | 10 |
| Target | `is_fraud` |
| Fraud count | 151 (1.51%) |
| Entity identifier | None |
| Temporal column | `transaction_hour` (0–23 only, no date) |
| V-features | None |
| Published benchmark? | No |
| **Assessment** | **NOT suitable for primary research.** Too small, no entity tracking, no real timestamps, unknown provenance. Should not be used in any primary experiment. |

---

## 5. Are the Datasets Sufficient?

### Dataset Sufficiency Verdict

The current dataset portfolio is **substantially sufficient** for the stated research program, with one genuine limitation (ULB temporal coverage) and one clear exclusion (10K toy).

**IEEE-CIS** provides a rich, real-world-derived benchmark with entity identifiers, second-level timestamps, 6 months of history, and extensive features. It is the appropriate primary benchmark for E1, E2, E3, and the initial explainability and streaming experiments.

**BAF** (6 variants × 1M rows) is the strongest concept drift asset in the portfolio. Its NeurIPS 2022 provenance, purpose-designed drift variants, and 8 ordinal months of data make it the natural dataset for Phase 5 (concept drift) and Phase 6 (adaptive retraining). No additional dataset is needed for this role.

**PaySim** provides the mobile money / streaming demonstration dimension. Its `nameOrig` entity and `step` temporal column enable a distinct experimental context. It is not a substitute for IEEE-CIS in the primary experiments, but contributes meaningfully to the streaming and domain demonstration phases.

**ULB** covers the external validation dimension. Its combination of real-world provenance, extreme class imbalance (578:1), PCA anonymization, and independent domain (European credit cards, 2013) makes it a legitimate cross-dataset validation target. The 48-hour limitation prevents its use for concept drift, but does not diminish its value as an external benchmark.

**What is NOT needed:**
- A second real credit card dataset: ULB already fills this role
- Another banking dataset: BAF is large and purpose-designed
- A synthetic UPI dataset: PaySim provides the mobile money dimension; adding a UPI-specific synthetic dataset would increase complexity without answering a new research question
- Any merger of the above datasets: scientifically invalid due to incompatible feature semantics

**Genuine gap (not a dataset gap):** The project lacks a dataset with labeled fraud occurring under documented real-world concept drift with ground-truth drift timestamps. BAF's synthetic drift variants partially address this, but the drift mechanism is simulated rather than observed. This is a limitation of the field, not a project-specific oversight.

---

## 6. Phase 1 — Complete Explanation

### What is Phase 1?

Phase 1 answers the question: *What is the best fraud detection performance achievable using only the information available at the moment a single transaction occurs (tabular features), without using any prior transaction history?*

This is the **foundational baseline** — all subsequent experiments are evaluated relative to it.

### Step-by-Step Walkthrough

**Step 1 — Raw IEEE-CIS Data**  
Two CSV files from Kaggle: `train_transaction.csv` (~434K rows, 394 columns) and `train_identity.csv` (~144K rows, 41 identity columns). Also test versions.  
*Why two files?* Vesta Corporation provides transaction features separately from device/identity features because not every transaction has identity information.

**Step 2 — Join**  
`train_transaction.csv` LEFT JOIN `train_identity.csv` on `TransactionID`.  
*Why LEFT JOIN?* Preserves all transactions (transaction table is the authoritative record). Transactions without identity data have NaN in identity columns — this is meaningful information (no identity link may itself be a fraud signal).

**Step 3 — Chronological Split (Critical Leakage Control)**  
Transactions are sorted by `TransactionDT` and split by time boundary — not random shuffling.  
- Train: earliest ~73.6% of transactions
- Validation: next ~13.2%
- Test: final ~13.3%

*Why chronological?* In production, a model is always trained on past data and evaluated on future data. Random splitting would allow the model to "see the future" during training — a severe form of temporal leakage that would inflate metrics and not reflect real deployment.

**Step 4 — Missing Value Handling**  
Extensive missingness (some V-features: >70% NaN). Strategy:  
- Numerical: filled with median (computed on training set only, applied to val/test)
- Categorical: filled with constant "MISSING" category
- Imputer fitted on training data only — prevents future information contaminating imputation statistics.

*What could go wrong?* If imputer were fitted on full data (including val/test), the imputed values would reflect future distribution — subtle leakage.  
*How controlled?* Preprocessing pipeline fitted only on training split.

**Step 5 — Categorical Encoding**  
Categorical columns (ProductCD, card types, email domains, device info, etc.) encoded using Target Encoding fitted on training data only. An encoding comparison audit (mean-target, ordinal, frequency) confirmed target encoding provided the best validation PR-AUC.

**Step 6 — Feature Engineering**  
406 features total:
- Original transaction features (TransactionAmt, ProductCD, card fields, addr, P/R email domains)
- Identity features (DeviceType, DeviceInfo, id_01–id_38)
- V-features (V1–V339, Vesta-proprietary)
- D-features (D1–D15, Vesta time delta features)
- C-features (C1–C14, count-type features)
- M-features (M1–M9, match flag features)
- Engineered temporal features (hour_sin, hour_cos, day_index, etc.)
- Excluded: TransactionID, TransactionDT, isFraud (target)

**Step 7 — LightGBM Training**  
LightGBM with gradient boosted decision trees. Key parameters:
- `is_unbalance=True` (handles class imbalance internally)
- `num_leaves=63`, `max_depth=-1`
- `learning_rate=0.05`
- Early stopping at 50 rounds (monitoring validation PR-AUC)
- `feature_fraction=0.8`, `bagging_fraction=0.8` (regularization)
- Seed=42

*Why LightGBM?* Gradient boosted trees are the state-of-the-art tabular ML method, requiring no feature scaling, handling missingness gracefully, and providing fast training.

**Step 8 — Threshold Selection (Validation Only)**  
Decision threshold selected to maximize F1 on the validation set (F1-max). Threshold = 0.616521.  
*Why not 0.5?* LightGBM outputs calibrated probabilities for an imbalanced problem. A 3.5% fraud rate means most transactions score low, so a higher threshold better separates classes. Using test labels for threshold selection would constitute test-set leakage.

**Step 9 — Final Test Evaluation**  
Applied frozen model and frozen threshold to test transactions exactly once.

**Step 10 — Phase 1.5 Robustness Hardening**  
- Paired bootstrap (2,000 resamples): quantifies metric uncertainty
- Operational metrics: recall@FPR (operationally realistic), Precision@Top-K (alert queue)
- Calibration: Brier score, ECE (probability reliability)
- Encoding comparison: confirms target encoding choice

---

## 7. Phase 1 Results

*All values from `experiments/E1_lightgbm/metrics.json` (SHA-256: `6f3ab9c86f9b9f47`)*

### Standard Test Metrics (78,542 transactions)

| Metric | Value | Interpretation |
|--------|-------|---------------|
| **PR-AUC** | **0.531731** | Area under Precision-Recall curve. A random classifier on 3.5% fraud would score ~0.035 (the fraud prevalence). E1 is approximately 15× above this baseline. |
| ROC-AUC | 0.898993 | Probability that a randomly chosen fraud scores higher than a randomly chosen legitimate transaction. 90% is strong for imbalanced data. |
| Precision | 0.58181 | Of transactions flagged as fraud, 58.2% were actually fraud. |
| Recall | 0.493511 | Of all actual frauds, 49.4% were caught. |
| F1 | 0.534035 | Harmonic mean of precision and recall. |
| MCC | 0.520301 | Matthews Correlation Coefficient — balanced metric robust to class imbalance. |
| Balanced Accuracy | 0.740262 | Average of sensitivity (recall) and specificity. |
| FPR | 0.012987 | Of all legitimate transactions, 1.3% were incorrectly flagged. |
| FNR | 0.506489 | Of all frauds, 50.6% were missed. |
| Threshold | 0.616521 | Selected on validation F1-max. |

**Confusion matrix (78,542 test):**

|  | Predicted Legit | Predicted Fraud |
|--|----------------|----------------|
| Actual Legit (75,768) | TN=74,784 | FP=984 |
| Actual Fraud (2,774) | FN=1,405 | TP=1,369 |

### Bootstrap Confidence Intervals (Phase 1.5)

| Metric | Point Estimate | 95% CI | Std |
|--------|--------------|--------|-----|
| PR-AUC | 0.531731 | [0.5126, 0.5504] | 0.0096 |
| ROC-AUC | 0.898993 | [0.8924, 0.9055] | 0.0034 |
| F1 | 0.534035 | [0.5179, 0.5496] | 0.0082 |
| Precision | 0.58181 | [0.5624, 0.6006] | 0.0097 |
| Recall | 0.493511 | [0.4751, 0.5120] | 0.0093 |
| MCC | 0.520301 | [0.5039, 0.5364] | 0.0083 |

The narrow bootstrap CIs indicate stable estimates — E1's performance is not an artefact of sampling variability.

### Operational Metrics

| Metric | Value | Meaning |
|--------|-------|---------|
| Recall @ 0.1% FPR | 0.2112 | At 1-in-1000 false alert rate, catches 21% of fraud |
| Recall @ 0.5% FPR | 0.3836 | At 5-in-1000 false alert rate, catches 38% of fraud |
| Recall @ 1.0% FPR | 0.4618 | At 1% false alert rate, catches 46% of fraud |
| Recall @ 2.0% FPR | 0.5306 | At 2% false alert rate, catches 53% of fraud |
| Precision @ Top-100 | 0.98 | Of the 100 highest-risk transactions, 98 are actual fraud |
| Precision @ Top-500 | 0.908 | Of the 500 highest-risk transactions, 90.8% are fraud |
| Precision @ Top-1,000 | 0.841 | Of the 1,000 highest-risk transactions, 84.1% are fraud |
| Precision @ Top-5,000 | 0.345 | Of the 5,000 highest-risk transactions, 34.5% are fraud |

### Calibration

| Metric | Value | Naive Baseline |
|--------|-------|---------------|
| Brier Score | 0.0324 | 0.0341 (predicts constant fraud rate) |
| ECE (10-bin) | 0.0524 | — |

Brier score is slightly below the naive baseline, indicating the model's probability estimates provide useful information. ECE of ~5% indicates moderate calibration quality.

---

## 8. Phase 1 Research Value

**Why was a tabular baseline necessary before temporal modeling?**

Without E1, it would be impossible to answer the primary research question: "Does temporal history add value?" You can only measure added value if you have a baseline to compare against. A weak baseline would make a temporal model look artificially impressive. A strong baseline that already captures most available signal is more informative — it tests whether temporal information adds anything *beyond what is already expressible in per-transaction features*.

**What would happen without E1?**  
Any temporal model trained without a strong tabular baseline would produce uninterpretable results. An improvement could reflect better features, better model capacity, or temporal information — there would be no way to separate these.

**Why is E1 the reference point for all future experiments?**  
E1 establishes the ceiling of achievable performance from single-transaction features. Every future experiment (E2, E3, cross-dataset) is evaluated against this ceiling. If a temporal model exceeds E1, it demonstrates genuine additional predictive value from history. If it does not, it demonstrates the tabular representation already captures most available signal.

**Which E1 result matters most for future experiments?**  
PR-AUC = 0.531731 (full test) and 0.5267 (common 76,520) are the primary reference metrics, because PR-AUC is the most appropriate metric for severe class imbalance. The bootstrap CI [0.5126, 0.5504] defines the uncertainty band within which a new experiment must exceed to claim meaningful improvement.

---

## 9. Phase 2 — Complete Explanation

### The Core Idea

Phase 2 tests whether *knowing what a cardholder did in the past* helps predict whether their current transaction is fraudulent. Phase 1 only looks at the current transaction in isolation.

### Conceptual Example

Imagine Card A has the following transaction history:

```
Card A:
  T1: Monday 9am — $45 grocery (Chicago)       → Legit
  T2: Monday 2pm — $120 online purchase        → Legit
  T3: Tuesday 8am — $35 coffee shop            → Legit
  T4: Tuesday 11am — $89 gas station           → Legit
  T5: Tuesday 11:02am — $5,200 electronics (Miami) → ??? Fraud?
```

**What Phase 1 (E1) sees:** Only T5's features — amount, merchant, device, email, card type, etc.  
**What Phase 2 (E2) sees:** The sequence [T1, T2, T3, T4] plus T5's features.

The hope is that the GRU can learn patterns like: "a sudden large purchase in a different city, minutes after a normal transaction, from a card with a pattern of small local purchases, is suspicious" — even if T5 alone is ambiguous.

### Key Design Decisions

**Why card1?**  
`card1` is the anonymized card-level entity identifier in IEEE-CIS. It allows transactions to be grouped by cardholder. `card1` is used for grouping only and is excluded from the GRU's input features — the model learns behavioral patterns, not card identities.

**Why minimum entity size = 5?**  
With a window of 5 transactions (4 history + 1 target), an entity needs at least 5 transactions to produce even one valid training window.

**Why L=4 history steps?**  
A balance between providing meaningful behavioral context and computational cost. Longer windows would require more memory and reduce the number of eligible entities.

**Why stride=1?**  
Sliding window: every eligible transaction in an entity's history is used as a prediction target, with the 4 preceding transactions as history. Maximizes data utilization.

**Temporal ordering — why strict ascending TransactionDT?**  
Using history in chronological order is mandatory for causal validity. If future transactions appeared in the history window, the model would be trained on information unavailable at prediction time — temporal leakage.

**Why exclude 585 duplicate-DT windows?**  
When the target transaction (T5) shares an identical TransactionDT with the last history transaction (T4), chronological ordering cannot establish which came first. Rather than inventing an ordering (which would be arbitrary), these 585 ambiguous windows are excluded. This is a principled conservative choice.

**The gap feature:**  
`log1p(DT(T5) − DT(T4))` — the log-transformed time elapsed between the last history transaction and the target. This encodes whether the target arrived quickly (possible burst fraud) or slowly (normal behavior) relative to prior activity.

**Target-based split:**  
Which split (train/val/test) a sequence belongs to is determined by the target transaction's split. History transactions may come from earlier splits — this is the correct causal design. A sequence's "label" is the target's label; a target's future transactions must not appear in its history.

---

## 10. E2a Explained

E2a is the **control experiment** — the locked GRU architecture applied to raw (unscaled) E1 features.

**Architecture:**
- Input: [batch, 4, 406] — 4 history steps, 406 features per step
- GRU: 406 → 64 (hidden), 1 layer, dropout=0.2
- Output: Linear(64 → 1) producing a single logit
- Trainable parameters: 90,689

**Training:** Adam, lr=0.001, batch=128, pos_weight=27.47, max 30 epochs, patience=5

**Result:** Best epoch=1, best validation PR-AUC = **0.0345**

**Why does 0.0345 matter?**  
The training set fraud prevalence is 14,397/398,312 = 3.61%. A model predicting "fraud probability = constant 3.5%" for every transaction would achieve PR-AUC ≈ fraud prevalence ≈ 0.035. E2a's validation PR-AUC of 0.0345 is approximately equal to this naive baseline — indicating near-random performance.

**What E2a tells us:**  
The GRU architecture, with the locked training configuration, cannot learn meaningful representations from unscaled E1 features. It achieves essentially the same score as a classifier that ignores all input. This is documented evidence that the architecture is sensitive to input scale — exactly the known property of gradient-based neural network training.

**Why keep E2a?**  
E2a is a scientifically important negative control. Without it, we could not demonstrate that E2b's improvement (0.0345 → 0.1377) is specifically associated with the scaling change. E2a establishes the unscaled baseline, and E2b tests the scaled variant — making the comparison interpretable.

---

## 11. E2b Explained

E2b introduces **StandardScaler** as the only change from E2a.

**Why does scaling matter for neural networks?**  
Gradient descent converges fastest when input features have comparable scale. If some features have values in [−300, +20,000] (e.g., large numerical aggregates in IEEE-CIS) and others in [0, 1] (binary indicators), the gradient signal is dominated by large-scale features. The network struggles to learn from small-scale features, and the optimization landscape becomes poorly conditioned. StandardScaler subtracts the mean and divides by the standard deviation of each feature, resulting in zero-mean, unit-variance features — creating a well-conditioned optimization landscape.

**Why train-only scaler fitting?**  
If the scaler were fitted on validation or test data, the scaling statistics would reflect future information. The standardization of training data would be influenced by future distribution — a form of preprocessing leakage. The scaler is fitted only on 398,312 × 4 = 1,593,248 training observations. Validation and test transformations use these frozen statistics.

**Training Results:**

| Epoch | Train Loss | Val Loss | Val PR-AUC |
|-------|-----------|---------|-----------|
| 1 | — | — | — |
| 2 | 1.1071 | 1.1311 | **0.1377** ← checkpoint |
| 3 | — | — | — |
| ... | | | |
| 7 | — | — | — (early stop) |

Best checkpoint = Epoch 2 (val PR-AUC = 0.1377, SHA-256: `31b17bb46d2c9257`)

**Observations:**  
- Validation loss rising after epoch 2 indicates overfitting
- Early stopping (patience=5) triggered at epoch 7
- Val PR-AUC of 0.1377 is substantially above the fraud prevalence baseline (~3.4%) — the model learned useful signal
- E2b threshold = 0.810221 (val F1-max) — notably high, consistent with a model that assigns low fraud probabilities overall

---

## 12. STEP 9 — Final Evaluation

### The Protocol

STEP 9 is the first and only evaluation of E2b on the test set. This is a methodological commitment: evaluating on test multiple times (test-driven hyperparameter selection) would produce optimistic results by effectively treating the test set as another validation set.

### Common Population

E1 was evaluated on 78,542 test transactions. E2b can only be evaluated on sequences with at least 4 valid causal history transactions within the same `card1` entity. This yields 76,520 eligible sequences — 97.43% coverage.

The 2,022 E1-only transactions are not "failures" — they simply lack sufficient card history to form a 4-step sequence. This is a documented limitation of the sequence construction methodology.

**Label alignment:** Both models are evaluated on exactly the same 76,520 TransactionIDs. The label set is verified identical. This is the paired comparison population.

### Threshold Selection

E1 threshold (0.616521) is frozen from Phase 1. E2b threshold (0.810221) was selected to maximize F1 on the validation set — using only validation labels, not test labels.

### Paired Bootstrap (Primary Statistical Analysis)

2,000 paired bootstrap resamples of the 76,520 common TransactionIDs, seed=42.  
Convention: Δ = PR-AUC(E2b) − PR-AUC(E1)  
Observed Δ = −0.3584, 95% CI [−0.3756, −0.3395]

**Interpretation:** The CI lies entirely below zero. Across all 2,000 resamples of the same test population, E1 consistently outperforms E2b on PR-AUC. The magnitude of the difference (−0.36) is large relative to the CI width (~0.04), indicating a robust finding under the tested conditions.

---

## 13. Phase 2 Results

### Summary Table — Common 76,520 Population

| Metric | E1 | E2b | Direction |
|--------|-----|------|---------|
| **PR-AUC** | **0.5267** | 0.1683 | E1 higher |
| ROC-AUC | 0.8981 | 0.7422 | E1 higher |
| Precision | 0.5790 | 0.1913 | E1 higher |
| Recall | 0.4900 | 0.3337 | E1 higher |
| F1 | 0.5308 | 0.2432 | E1 higher |
| MCC | 0.5170 | 0.2166 | E1 higher |
| Balanced Accuracy | 0.7385 | 0.6411 | E1 higher |
| Brier Score | 0.0324 | 0.2377 | E1 lower (better) |
| ECE | 0.0523 | 0.4247 | E1 lower (better) |
| FPR | 0.0130 | 0.0516 | E1 lower (better) |
| FNR | 0.5100 | 0.6663 | E1 lower (better) |

**What the Brier and ECE mean:**  
E2b's Brier score of 0.2377 is substantially higher than E1's 0.0324. E2b's ECE of 0.4247 indicates poor calibration — its predicted probabilities are not reliable estimates of actual fraud probabilities. This is consistent with the high threshold (0.810221) required to operate E2b at any reasonable precision.

**What the operational metrics mean:**

| Metric | E1 | E2b | Practical meaning |
|--------|-----|------|------------------|
| Recall @ 1% FPR | 0.4578 | 0.1441 | At a fixed alert budget (1% of legit flagged), E1 catches 46% of fraud vs. E2b's 14% |
| Precision @ Top-100 | 0.98 | 0.48 | Of the 100 highest-risk alerts, E1 correctly identifies 98 frauds; E2b only 48 |

---

## 14. Phase 1 vs Phase 2 Comparison

This comparison must be framed carefully.

**E1 and E2 answer different research questions:**
- E1 answers: What can per-transaction tabular features achieve?
- E2 answers: What can standalone historical sequence modeling achieve?

These are not equivalent questions. The empirical finding is that, under the tested configurations, E1 produced substantially stronger performance than E2b on the common test population.

**What this does NOT mean:**
- It does not establish that temporal information has no predictive value. E1's 406 features already contain substantial implicit temporal information (frequency features, aggregation features, time-of-day encodings). E2 may be attempting to find temporal signal that E1 already captures in a different form.
- It does not establish that the GRU architecture is inadequate. A single-layer GRU with hidden size 64, trained for a maximum of 7 epochs on 4-step sequences, is an intentionally simple configuration.
- It does not establish that E3 (hybrid: tabular + temporal) cannot exceed E1. E3 is specifically designed to test whether temporal information provides complementary (additive) signal beyond what E1's features already express.
- It does not establish that a different sequence length, entity definition, or architecture would produce the same result.

**What this DOES establish:**  
Under the locked E2b configuration (4-step windows, 406 standardized features, single-layer GRU, Adam lr=0.001), standalone temporal sequence modeling did not outperform the frozen tabular baseline on the common test population, with a 95% paired bootstrap CI that excludes zero.

---

## 15. What Phase 2 Established vs. Left Open

### Established Findings

1. The locked GRU architecture (FraudGRU, 90,689 params) produces deterministic outputs under seed=42 and passes 15/15 architecture tests.
2. The Phase 2 sequence construction methodology (card1 grouping, L=5 window, stride=1, Option A duplicate handling) produces 550,559 valid windows from 6,511 entities, with 33/33 leakage tests passed.
3. E2a (unscaled): The locked GRU with unscaled features achieves validation PR-AUC ≈ 0.0345, approximately equal to fraud prevalence.
4. E2b (scaled): StandardScaler preprocessing is associated with a validation PR-AUC improvement from 0.0345 to 0.1377.
5. E2b test evaluation: On the 76,520 common test population, E2b PR-AUC = 0.1683 vs. E1 PR-AUC = 0.5267. Δ PR-AUC = −0.3584, 95% CI [−0.3756, −0.3395] (excludes zero).

### Open Hypotheses

1. Whether temporal information provides complementary signal when combined with tabular features (E3).
2. Whether longer history windows (L>4) improve GRU performance.
3. Whether attention mechanisms or Transformer architectures produce different results on the same sequences.
4. Whether temporal modeling provides greater relative benefit on different datasets (BAF, PaySim, ULB).
5. Whether GRU-based features contribute to robustness under concept drift.
6. Whether the E1 feature set is well-suited as GRU input, or whether sequence-native features would be more effective.

---

## 16. Methodological Issues Audit

| Issue | Classification | Affects Results? | Action Required? |
|-------|---------------|-----------------|-----------------|
| W1: RuntimeWarning in f1_max_threshold() | MINOR | No | No — experiment frozen, documented |
| W2: SMOTE/oversample fields stored as null vs false | DOCUMENTATION ONLY | No | No — experiment frozen, documented |
| W3: ROC-AUC bootstrap CI rounded in narrative vs. manifest | DOCUMENTATION ONLY | No | ✅ Fixed — corrected in PHASE2_FINAL_AUDIT.md and E1_vs_E2b_paired_comparison.md |
| E2b val loss increases after epoch 2 (overfitting) | NO ISSUE | No — expected, controlled by early stopping | No |
| 585 duplicate-DT windows excluded | NO ISSUE | Affects coverage by ~0.11%, documented | No |
| 2,022 E1-only transactions excluded from paired comparison | DOCUMENTED LIMITATION | Documented in all reports | No |
| E2b threshold = 0.810221 (unusually high) | NO ISSUE | Consistent with low fraud scores; documented | No |
| E1 features used for E2 input (cross-use) | NO ISSUE | Intentional design — tests whether E1 features contain temporal signal | No |
| Train-only scaler fitting verified | NO ISSUE | Critical control — verified by 22/22 PT tests | No |
| Test evaluated exactly once | NO ISSUE | Critical control — verified by integrity audit | No |
| Chronological split (no random shuffle) | NO ISSUE | Correct methodology | No |
| Threshold selected on validation only | NO ISSUE | Critical control — verified | No |
| Bootstrap paired on common population | NO ISSUE | Correct methodology | No |
| Seed=42 used throughout | NO ISSUE | Reproducibility — documented | No |
| card1 excluded from GRU input | NO ISSUE | Intentional — prevents entity memorization | No |

**Critical issues:** None found.  
**Major issues:** None found.  
**Moderate issues:** None found.  
**Minor issues:** 2 (W1, W2) — both documented and frozen.  
**Documentation-only:** 1 (W3) — corrected.

---

## 17. Phase 2 Freeze Validation

**Freeze manifest:** `reports/phase2/phase2_freeze_manifest.json`  
**Final audit:** `reports/phase2/PHASE2_FINAL_AUDIT.md`

| Component | Verified |
|----------|---------|
| E1 predictions.parquet SHA-256 | ✅ `6a73279d81491437` MATCH |
| E2b gru_best.pt SHA-256 | ✅ `31b17bb46d2c9257` MATCH |
| E2b standard_scaler.pkl SHA-256 | ✅ `fe294184277da80f` MATCH |
| E1 metrics.json SHA-256 | ✅ `6f3ab9c86f9b9f47` MATCH |
| All common-population metrics (20 values) | ✅ Consistent |
| Paired bootstrap results | ✅ Consistent |
| E2b training configuration | ✅ All parameters verified |
| Test suite | ✅ 80/80 |
| Common IDs file | ✅ 76,520 entries |
| Known warnings documented | ✅ W1, W2 |

**Phase 2 should remain frozen.** All verifiable properties check out. No evidence of artifact modification.

---

## 18. Research-Level Assessment

### Can the Work Be Presented as Research-Grade?

**Yes — with important qualifications.**

The following dimensions are research-grade:

**1. Experimental Rigor:** The E1 baseline, E2a control, and E2b experiment follow a principled design — locked before execution, with explicit leakage prevention at every stage. Bootstrap CIs and paired statistical testing are correctly applied.

**2. Leakage Prevention:** Chronological splits, train-only preprocessing, validation-only threshold selection, single test evaluation, and 33/33 leakage tests represent serious leakage prevention. This is notably stronger than many published fraud detection papers.

**3. Statistical Validation:** Paired bootstrap on 2,000 resamples with seed documentation is appropriate for this sample size and imbalance level. The CI [−0.3756, −0.3395] is precise and well-motivated.

**4. Reproducibility:** Seed documentation (42 throughout), artifact hashes, frozen manifests, and detailed metadata provide a strong reproducibility trail.

**5. Dataset Methodology:** Chronological split, entity-aware sequence construction, and separate preprocessing pipelines per dataset demonstrate appropriate methodology.

### Dimensions Not Yet Research-Grade (for publication):

**6. Novelty:** The current experimental program is methodologically rigorous but does not yet claim a novel research contribution. E1 + E2 (tabular baseline + temporal comparison) is a well-established experimental paradigm in fraud detection. The contribution needs to be explicitly stated — what does this project establish that published literature does not?

**7. Generalization:** With only IEEE-CIS results, claims about the research cannot generalize beyond this dataset. The planned cross-dataset evaluation (BAF-E1, ULB-E1) is essential for any publication claim.

**8. Concept Drift:** The project's research title includes concept drift, but no concept drift experiment has been conducted. This is the most significant gap between stated research objectives and completed work.

**9. Literature Positioning:** No systematic literature review or quantitative comparison against published state-of-the-art results has been documented.

**10. Ablation:** Only two GRU variants (E2a, E2b) have been tested. A research paper typically requires ablations — different window lengths, different architectures, different feature subsets — to support the experimental findings.

**Summary:** The work demonstrates rigorous experimental practice suitable for a conference paper on methodology and baseline comparison. It is not yet at the level of a complete PhD contribution, which would require novelty, generalization, and the remaining experimental phases.

---

## 19. Gap Analysis — What is Still Missing

### MUST HAVE (for stated research objectives)

| Item | Why Required |
|------|-------------|
| E3 Hybrid (IEEE-CIS) | Directly addresses the main open RQ: complementary temporal signal |
| Cross-dataset tabular (ULB-E1, BAF-E1) | Generalization — required for any publication claim |
| BAF Concept Drift (Variants I–V) | Core stated research dimension — entirely missing |
| Adaptive Retraining (BAF) | Core stated research dimension — entirely missing |
| Explainability (SHAP on E1/E3) | Stated in research title — entirely missing |

### SHOULD HAVE

| Item | Why Valuable |
|------|-------------|
| PaySim-E1 baseline | Mobile money domain — completes dataset portfolio |
| Temporal ablations (L=2, 8, 16) | Justifies L=4 choice with evidence |
| Architecture ablations (LSTM, Transformer attention) | Strengthens E2 findings |
| Literature comparison table | Positions work against published results |
| Kafka/Spark streaming demo | Fulfills research title claim |

### OPTIONAL (not scientifically essential)

| Item | Comment |
|------|---------|
| FastAPI serving | Engineering demonstration, not research contribution |
| Docker containerization | Deployment concern, not research concern |
| End-to-end integration demo | Demonstrates system but does not add scientific value |
| Formal latency benchmarks | Relevant only for deployment paper |

---

## 20. Future Experiment Map

```
CURRENT STATE (Phase 2 frozen)
        │
        ▼
E3 Hybrid — IEEE-CIS
  RQ: Does temporal information provide COMPLEMENTARY signal with tabular features?
  Design: Concatenate GRU temporal embedding + E1 tabular features → MLP classifier
  Baseline: E1 (frozen), E2b (frozen)
  Evaluation: Same 76,520 common population, same paired bootstrap
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
Cross-Dataset Tabular                BAF-E1 Baseline
  ULB-E1 (LightGBM on ULB)          (tabular on BAF Base)
  BAF-E1 (LightGBM on BAF)          Establishes drift anchor
  PaySim-E1 (optional)                     │
                                           ▼
                                   BAF Concept Drift
                                   Variants I→V sequential
                                   Measure drift impact
                                         │
                                         ▼
                                   Adaptive Retraining
                                   BAF windowed retraining
                                         │
                          ┌──────────────┤
                          ▼              ▼
                   Explainability    Streaming
                   SHAP on E1/E3    Kafka + Spark
                   Feature importance PaySim replay
                          │              │
                          └──────┬───────┘
                                 ▼
                         FastAPI + Deployment
                                 │
                                 ▼
                         End-to-end Integration
```

---

## 21. E3 Scientific Justification

E3 is scientifically justified because E2's finding is informative, not terminal.

E2b established that a standalone GRU does not outperform E1 on PR-AUC. This finding is consistent with two non-mutually-exclusive explanations:

**Explanation A:** E1's 406 features already capture most fraud-predictive temporal information through aggregation features (velocity counts, frequency encodings, time deltas). The GRU operating on these features in sequence finds no additional signal.

**Explanation B:** The GRU architecture is insufficient — too shallow, too short a window, no attention — to extract temporal signal even if it exists.

E3 tests a third approach: use E1's tabular prediction as a strong feature, and add a GRU-derived temporal embedding as a complementary channel. If E3 improves over E1, it demonstrates that temporal information provides signal beyond what E1's tabular features express, even if E2 (standalone GRU) did not.

**Scientifically valid E3 outcomes:**
1. E3 substantially exceeds E1: temporal information is complementary; hybrid architecture is justified
2. E3 marginally exceeds E1: temporal information provides small complementary signal; hybrid may be worth deployment cost
3. E3 matches E1: temporal embedding adds no complementary value; E1's features already capture available temporal signal
4. E3 falls below E1: hybrid degrades baseline; temporal information actively harms the tabular decision

All four outcomes are informative research results.

---

## 22. Final Honest Verdict

| Question | Answer |
|---------|--------|
| 1. Are we going in the correct research direction? | Yes — rigorous tabular baseline → temporal comparison → hybrid → drift → adaptation is a principled research program |
| 2. Are current datasets sufficient? | Yes, for the stated research program — IEEE-CIS (primary), BAF (drift), PaySim (streaming), ULB (external validation) cover all required dimensions |
| 3. Should we add more datasets? | No — adding more datasets without completing experiments on existing ones would dilute focus without adding research value |
| 4. Are the assigned dataset roles appropriate? | Yes — each role assignment is supported by measured dataset characteristics, not assumptions |
| 5. Should any datasets be merged? | No — IEEE-CIS V features and ULB V features are semantically incompatible; all datasets have different feature spaces |
| 6. Was Phase 1 implemented correctly? | Yes — chronological split, train-only preprocessing, validation-only threshold, single test evaluation, bootstrap CI |
| 7. Was Phase 2 implemented correctly? | Yes — locked sequence spec, leakage tests, train-only scaler, single test evaluation, paired bootstrap; two documented warnings (W1, W2) do not affect results |
| 8. Is E1 the stronger empirical baseline on the common population? | Yes — empirically, on the 76,520 common test population, under the tested configurations |
| 9. Does E2 invalidate temporal modeling? | No — it establishes that the specific E2b configuration did not outperform E1; temporal modeling in hybrid configurations, longer windows, and other architectures remain open |
| 10. Is E3 scientifically justified? | Yes — it tests the complementary hypothesis that E2 could not test |
| 11. Are there critical mistakes? | None found |
| 12. Are there major mistakes? | None found |
| 13. Are there minor/documentation issues? | W1 (RuntimeWarning), W2 (null vs false), W3 (ROC CI rounding — corrected) |
| 14. Is Phase 2 correctly frozen? | Yes — all hashes verified, metrics consistent, 80/80 tests, warnings documented |
| 15. Most important next experiment? | E3 Hybrid (IEEE-CIS) |
| 16. Most important unresolved question? | Does temporal information provide complementary signal when combined with tabular features? |
| 17. Current strongest evidence? | E1 PR-AUC = 0.531731 with bootstrap CI [0.5126, 0.5504]; Precision@Top-100 = 0.98 |
| 18. Biggest current limitation? | No concept drift experiments completed despite being a stated research objective |
| 19. What would make the work substantially stronger? | E3 + cross-dataset + BAF drift experiments + literature comparison |
| 20. What should NOT be prioritized? | Docker, FastAPI, end-to-end integration — these are engineering tasks, not research contributions |

---

*Complete Research Status Review — 2026-09-18*  
*Audit only. No training. No model changes. No artifact modifications.*  
*All metrics from frozen artifacts. All values measured, not assumed.*
