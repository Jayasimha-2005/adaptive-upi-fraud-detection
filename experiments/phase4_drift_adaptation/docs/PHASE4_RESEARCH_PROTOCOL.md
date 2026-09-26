# PHASE 4 RESEARCH PROTOCOL
## Concept Drift Detection + Adaptive Retraining
### Adaptive Financial Fraud Detection — BAF Temporal Experiment

**Version:** 1.1 (FINAL — approved for implementation)  
**Date:** 2026-09-19  
**Status:** APPROVED FOR IMPLEMENTATION — v1.1 FINAL  
**Preceding phases:** E1 (frozen), E2a/E2b (frozen), E3-A/B/C (verified)  
**Dataset:** BAF Base.csv — 1,000,000 rows, 32 columns, months 0–7

---

## 1. RESEARCH QUESTION

> **"Does drift-triggered adaptive retraining maintain fraud-detection performance better than a static model under temporal distribution changes in the BAF dataset?"**

This question must remain the central focus of Phase 4. It is not about:
- Finding the best-performing fraud model
- Optimizing LightGBM hyperparameters on BAF
- Comparing BAF to IEEE-CIS
- Implementing streaming infrastructure

---

## 2. HYPOTHESES

### Primary Null Hypothesis (H0)
Drift-triggered adaptive retraining does not produce a measurable improvement in PR-AUC over the static model on future temporal evaluation periods, compared to the static baseline.

### Alternative Hypothesis (H1)
Drift-triggered adaptive retraining produces a measurable improvement in PR-AUC over the static model on future temporal evaluation periods.

> **Do not assume H1 is true.** A null or negative result is scientifically valid.

---

## 3. DATASET

| Parameter | Value | Source |
|-----------|-------|--------|
| Dataset | BAF (Bank Account Fraud) Base variant | Verified from `Base.csv` |
| Citation | Jesus et al., NeurIPS 2022 | BAF paper |
| Repository | https://github.com/feedzai/bank-account-fraud | Official |
| Rows | 1,000,000 | Verified |
| Columns | 32 | Verified |
| Temporal col | `month` (integer 0–7) | Verified |
| Target col | `fraud_bool` (0/1) | Verified |
| Overall fraud rate | 1.103% | Verified |
| Missing values | None | Verified |
| Duplicate rows | None | Verified |
| File hash (16) | `7bf10a37ce07e72e` | Verified |

---

## 4. FEATURES

### 4.1 Excluded Columns
| Column | Reason |
|--------|--------|
| `fraud_bool` | Target — must never enter feature set |
| `month` | Temporal index — temporal leakage risk |
| `device_fraud_count` | Constant (all zero) — provides no signal |

### 4.2 Feature Set
**Total usable features: 29**

Numerical (24): `income`, `name_email_similarity`, `prev_address_months_count`, `current_address_months_count`, `customer_age`, `days_since_request`, `intended_balcon_amount`, `zip_count_4w`, `velocity_6h`, `velocity_24h`, `velocity_4w`, `bank_branch_count_8w`, `date_of_birth_distinct_emails_4w`, `credit_risk_score`, `email_is_free`, `phone_home_valid`, `phone_mobile_valid`, `bank_months_count`, `has_other_cards`, `proposed_credit_limit`, `foreign_request`, `session_length_in_minutes`, `keep_alive_session`, `device_distinct_emails_8w`

Categorical (5): `payment_type`, `employment_status`, `housing_status`, `source`, `device_os`

### 4.3 Near-Constant Features (Retained)
The following have ≤5 unique values but are not dropped: `email_is_free`, `phone_home_valid`, `phone_mobile_valid`, `has_other_cards`, `foreign_request`, `keep_alive_session`, `device_distinct_emails_8w`. They are binary indicators with potential signal. LightGBM handles low-variance features gracefully.

---

## 5. TEMPORAL SPLIT — LOCKED

Based on the verified period-level data from the BAF audit:

| Period | Rows | Fraud | Fraud Rate | Role |
|--------|------|-------|-----------|------|
| Month 0 | 132,440 | 1,500 | 1.133% | **Training** |
| Month 1 | 127,620 | 1,198 | 0.939% | **Training** |
| Month 2 | 136,979 | 1,198 | 0.875% | **Training** |
| Month 3 | 150,936 | 1,392 | 0.922% | **Training** |
| Month 4 | 127,691 | 1,452 | 1.137% | **Validation (threshold selection only)** |
| Month 5 | 119,323 | 1,411 | 1.183% | **Monitoring window 1 / Drift detection** |
| Month 6 | 108,168 | 1,450 | 1.341% | **Evaluation 1 + Monitoring window 2** |
| Month 7 | 96,843 | 1,428 | 1.475% | **Final evaluation (PROTECTED)** |

### 5.1 Sequential Adaptation Loop

The experiment implements a **genuinely sequential adaptation loop**, not a one-shot trigger:

```
             MONTHS 0–3
             Initial training
                  │
                  ▼
             MONTH 4
             Validation — threshold frozen
                  │
                  ▼
             MONTH 5
             Drift monitoring (vs months 0–3 reference)
                  │
           ┌──────┴──────┐
        no drift       drift
           │               │
           │          retrain on months 0–5
           └──────┬─────────────┘
                  ▼
             MONTH 6
             Evaluate static vs adaptive
             +
             Month 6 labels become available
             Drift monitoring (vs months 0–3 reference)
                  │
           ┌──────┴──────┐
        no drift       drift
           │               │
           │          retrain on months 0–6
           └──────┬─────────────┘
                  ▼
             MONTH 7
          🔒 FINAL EVALUATION
             (protected — evaluated exactly once)
```

### 5.2 Period-Role Rationale
- **Months 0–3 (initial training):** ~548K rows, ~5,288 fraud — stable fraud rate (0.875%–1.133%), representing the baseline distribution for both models and the PSI reference.
- **Month 4 (validation):** Used ONLY for F1-max threshold selection. Not used for drift detection or evaluation.
- **Month 5 (monitoring 1):** First drift window. PSI computed vs months 0–3 reference. Temporal distribution shift observed (KS=0.142, fraud rate rising to 1.183%).
- **Month 6 (evaluation 1 + monitoring 2):** Static and adaptive models are both evaluated. After evaluation, month 6 labels are assumed available; a second drift check is performed for the month-7 prediction. Drift further elevated (KS=0.152, fraud rate=1.341%).
- **Month 7 (final evaluation — PROTECTED):** Highest observed distribution shift (KS=0.182, fraud rate=1.475%). Evaluated exactly once after all adaptation decisions for month 6 are frozen.

### 5.3 No Random Splits
Temporal ordering is strictly preserved. No shuffle splits. No stratified random splits for the primary experiment.

---

## 6. STATIC BASELINE MODEL (Model A)

| Parameter | Value | Justification |
|-----------|-------|--------------|
| Algorithm | LightGBM | Strongest baseline established in E1 on IEEE-CIS |
| Training data | Months 0–3 only | Initial training period |
| Preprocessing | Fit on months 0–3 only | No future leakage |
| Validation | Month 4 (threshold only) | Threshold selection only, no model selection |
| Future data | NEVER retrained | Defines "static" |
| Predictions | Sequential on months 5, 6, 7 | Using fixed month-0-to-3 weights |

### 6.1 Static Model LightGBM Configuration — LOCKED
```
n_estimators     = 500
learning_rate    = 0.05
max_depth        = 6
num_leaves       = 63
min_child_samples = 50
subsample        = 0.8
colsample_bytree = 0.8
class_weight     = "balanced"
random_state     = 42
```
> **The specified LightGBM hyperparameters are fixed before any Month-5 monitoring or evaluation. No hyperparameter tuning, model-selection procedure, or configuration change is permitted during Phase 4 based on training or monitoring results. `scale_pos_weight` is NOT used — `class_weight="balanced"` is the single, deterministic imbalance treatment for all model versions.**

### 6.2 Preprocessing
- **Numerical:** No scaling required (LightGBM is scale-invariant).
- **Categorical:** **LightGBM native categorical handling.** The five categorical columns (`payment_type`, `employment_status`, `housing_status`, `source`, `device_os`) are passed as LightGBM `categorical_feature`. No ordinal encoder is used.
  - v1: category representation built from months 0–3 training data
  - v2: representation refitted on months 0–5 at retraining event
  - v3: representation refitted on months 0–6 at retraining event
  - No future-period category information is used when constructing any version's representation
- **Missing values:** None in BAF Base. If any encountered, use training-period mode imputation for categoricals and median for numericals.
- **Constant column:** `device_fraud_count` dropped before fitting.

---

## 7. ADAPTIVE MODEL (Model B)

The adaptive model is identical to the static model **except** it is allowed to retrain when the predefined drift detection criterion is met.

### 7.1 Sequential Adaptation Logic

The adaptive model participates in two sequential monitoring windows:

**Window 1 (Month 5):**
- Drift detector runs on month 5 features vs months 0–3 reference
- If triggered: retrain on months 0–5; preprocessing refit
- Evaluate month 6 with the updated model

**Window 2 (Month 6, after evaluation):**
- Month 6 fraud labels are assumed available after month 6 evaluation
- Drift detector runs on month 6 features vs months 0–3 reference
- If triggered: retrain on months 0–6; preprocessing refit
- Evaluate month 7 with the updated model

**Control rule:** At most one retraining event per evaluation period (one per monitoring window). There is no global cap across the experiment — two retraining events are possible if both windows trigger.

### 7.2 Retraining Policy: Expanding Window
When drift is detected after monitoring month M, retrain on **all data available up to and including month M**.

| Trigger at | Retraining data | Model version |
|-----------|----------------|---------------|
| Month 5 | Months 0–5 | v2 |
| Month 6 | Months 0–6 (if v2 exists, expands further) | v3 |
| No trigger | Original months 0–3 model retained | v1 |

- **Preprocessing (encoder, scaler) is refit** on the expanded training set at each retraining event
- LightGBM hyperparameters remain identical (no re-search)
- Each model version is logged with the trigger month, drift statistic, and training data range

### 7.3 Label Availability for Retraining
Drift detection is **label-free** (PSI on feature distributions only). Retraining, however, uses fraud labels. The explicit deployment assumption is:

> **Fraud labels for a completed monitoring period are assumed to become available before the subsequent retraining event.** The lag between period close and label availability is not modeled within the 8-period experiment. This is an acknowledged limitation.

### 7.4 Alternative Control Model (Optional)
**Model C — Periodic Retraining:** Retrain unconditionally after every monitoring period, regardless of drift.
- Provides context: does periodic retraining help independent of drift detection?
- Must be fully defined before evaluation if included
- Primary comparison remains Static (A) vs Drift-Triggered Adaptive (B)

---

## 8. DRIFT DETECTION

### 8.1 Label Availability Assumption
**UNSUPERVISED trigger:** Drift detection uses **feature distributions only**. Fraud labels are NOT required to trigger retraining. This is realistic — in production, fraud labels arrive with delay.

Fraud labels are used for:
- Computing evaluation metrics retrospectively after each period
- Threshold selection on validation month (month 4)
- **Retraining** (labels from the completed monitoring period are assumed available before the subsequent retraining event — see Section 7.3)

### 8.2 Drift Detector: Population Stability Index (PSI)
PSI is selected as the primary drift statistic because:
1. Industry-standard for tabular feature monitoring in financial services
2. Interpretable and well-documented
3. Does not require distributional assumptions
4. Aggregates naturally across features
5. Produces a numerical stability measure suitable for thresholding

**PSI formula (per feature):**
```
PSI = Σ (Actual% − Expected%) × ln(Actual% / Expected%)
```

### 8.3 PSI Implementation — Fully Specified

PSI computation must be completely specified before implementation to ensure reproducibility.

**Numerical features (24 features):**

| Parameter | Value |
|-----------|-------|
| Binning method | Quantile bins |
| Number of bins | 10 |
| Bin edges | Fitted on months 0–3 ONLY (reference period) |
| Edge reuse | Same edges applied to all future monitoring windows without re-fitting |
| Out-of-range values | Clipped to the outermost bin boundaries of the reference |
| Zero-bin smoothing | ε = 1e-6 (added to both reference and current proportions before log) |
| PSI formula | `Σ (cur_i − ref_i) × ln(cur_i / ref_i)` where proportions include ε |

**Categorical features (5 features: `payment_type`, `employment_status`, `housing_status`, `source`, `device_os`):**

| Parameter | Value |
|-----------|-------|
| Reference categories | Category frequency distribution from months 0–3 |
| Unseen categories | Assigned to an explicit `__UNSEEN__` bucket |
| Frequency computation | Proportion of each category in reference and monitoring window |
| Zero-frequency smoothing | ε = 1e-6 applied as per numerical case |
| PSI formula | Same formula as numerical, with categories as bins |

**Coverage:** PSI is computed for all **29 model-input features** (24 numerical + 5 categorical), after dropping `device_fraud_count`, `fraud_bool`, and `month`.

### 8.4 Reference Distribution
**FIXED INITIAL REFERENCE:** The reference distribution is always months 0–3 (the initial training period). It does NOT change after retraining. This makes the detector interpretable as:
> "How different is the current distribution from the environment in which the original model was trained?"

### 8.5 Feature-Level Aggregation — Exact Rule

**Denominator:** 29 model-input features (24 numerical + 5 categorical).

**Per-feature threshold:** PSI ≥ 0.10

**Aggregate trigger:** Retraining is triggered if **at least 6 of the 29 features** have PSI ≥ 0.10.

> Derivation: 20% × 29 = 5.8 → rounded up to 6 for an exact integer rule.

This is more robust than any single-feature trigger and avoids sensitivity to one outlier feature.

### 8.6 PSI Threshold — Operational Warning, Not Universal Law
| PSI value | Interpretation (credit-risk monitoring convention) |
|-----------|---------------------------------------------------|
| < 0.10 | No significant population shift |
| 0.10 – 0.25 | Moderate shift — worth monitoring |
| > 0.25 | Significant shift — action typically warranted |

> **PSI = 0.10 is adopted as a pre-specified operational warning threshold based on commonly used credit-risk model monitoring conventions. It is not treated as a universal statistical significance threshold, and its sensitivity to binning choice is acknowledged as a limitation.**

### 8.7 Minimum Observation Requirement
- Minimum monitoring window: **50,000 rows**
- Month 5: 119,323 rows ✅
- Month 6: 108,168 rows ✅
- Minimum fraud count: not required (label-free detection)

### 8.8 Sequential Monitoring Policy
- **No global maximum on retraining events** — at most one per monitoring window
- Two monitoring windows = maximum two possible retraining events
- There is no multi-window consecutive requirement — a single window exceeding the threshold triggers retraining

---

## 9. DRIFT DETECTION PARAMETERS — LOCKED

| Parameter | Locked Value |
|-----------|--------------|
| Per-feature PSI threshold | **PSI ≥ 0.10** |
| Aggregate trigger | **≥ 6 of 29 model-input features** |
| Denominator | **29** (all model-input features after removing device_fraud_count) |
| Reference distribution | Months 0–3 pooled |
| Monitoring windows | Month 5, then Month 6 (sequential) |
| Minimum observations | 50,000 rows per window |
| Consecutive windows required | 1 (single-window trigger) |
| Max triggers per monitoring window | 1 |
| Global max triggers | Not capped — at most 2 (one per window) |
| Numerical binning | 10 quantile bins, edges from months 0–3 |
| Categorical handling | Frequency-based, `__UNSEEN__` bucket |
| Smoothing epsilon | 1e-6 |
| PSI threshold character | Pre-specified operational warning; not universal statistical law |

> These values are fixed before implementation. They may not be changed after seeing month 6 or 7 results.

---

## 10. THRESHOLD SELECTION POLICY

**Method:** F1-max sweep over 200 thresholds (0.01–0.99) on validation month (month 4), applied to **Model v1 predictions**.

**The threshold is selected exactly once and frozen permanently:**

```
Month 4
    ↓
v1 predictions on Month 4
    ↓
F1-max threshold selected
    ↓
🔒 THRESHOLD FROZEN
    ↓
v2 uses same threshold
    ↓
v3 uses same threshold
```

**Why the threshold cannot be re-derived for v2/v3:**
When the adaptive model retrains on months 0–5, Month 4 becomes part of its training data. Evaluating v2 on Month 4 to select a threshold would constitute validation contamination — the model has already been fitted on those rows. Therefore:

> **The classification threshold derived from Model v1's Month-4 predictions is the single, permanent threshold for all model versions (v1, v2, v3). No retrained model version may re-optimize the threshold using Month 4 or any other period that has entered its training set.**

**PR-AUC and ROC-AUC remain threshold-independent** and are always reported regardless of which model version is active.

> No test labels (months 5, 6, 7) are used for threshold selection.

---

## 11. EVALUATION

### 11.1 Primary Metric
**PR-AUC** — justified by class imbalance (fraud rate ~1.1%) and consistency with Phases 1–3.

### 11.2 Secondary Metrics
ROC-AUC, Precision, Recall, F1, MCC, Balanced Accuracy, FPR, FNR, Brier Score, Recall at FPR=0.1%, 0.5%, 1%, 2%, Precision@Top-K (100, 500, 1000)

### 11.3 Evaluation Design
For each evaluation period (months 6 and 7), report:

| Metric | Static (A) | Adaptive (B) |
|--------|-----------|-------------|
| PR-AUC | | |
| ROC-AUC | | |
| F1 | | |
| Recall | | |
| Precision | | |
| Brier | | |
| Drift detected? | N/A | Yes/No |
| Retrained? | N/A | Yes/No |
| Model version | v1 (never changes) | v1, v2, or v3 |

### 11.4 Temporal Analysis Scope
Temporal analysis distinguishes between what is available at each stage:

| Analysis type | Months covered | Notes |
|--------------|---------------|-------|
| Distribution / drift (PSI, KS, fraud prevalence) | **0–7** | Full 8-period story |
| Feature shift plots | **0–7** | Reference always months 0–3 |
| Model performance (PR-AUC, F1, etc.) | **6–7 only** | Out-of-sample evaluation only |
| Adaptation event timeline | **5–6** | Monitoring windows |

> **Do not report or plot model performance for months 0–5.** The static model has no out-of-sample predictions for the training/validation/monitoring periods. Implying model performance over all 8 months would conflate in-sample fitting with out-of-sample evaluation.

---

## 12. STATISTICAL TESTING

| Parameter | Value |
|-----------|-------|
| Method | Paired bootstrap |
| Resamples | 2,000 |
| Seed | 42 |
| Statistic | Δ PR-AUC = Adaptive − Static |
| Confidence interval | 95% |
| Population | Identical evaluation period rows (months 6+7) |

> Resample observations consistently between static and adaptive predictions (same indices).

---

## 13. TEST SET PROTECTION

**Month 7 is the protected final evaluation period.**

Once the drift detector decision has been made and any retraining has occurred, month 7 is evaluated exactly once. No protocol changes are permitted based on month 7 results.

---

## 14. PREPROCESSING — TEMPORAL LEAKAGE CONTROLS

| Step | Policy |
|------|--------|
| Numerical scaling | If used: fit on months 0–3 only |
| Categorical encoding | Fit on months 0–3 only |
| Imputation | Fit on months 0–3 only (not needed for Base — no missingness) |
| Feature selection | Based on months 0–3 only |
| Target | `fraud_bool` excluded from features |
| Month column | `month` excluded from features |
| Constant col | `device_fraud_count` dropped |
| After retraining | Encoder/scaler refit on expanded training set (months 0–5) |

---

## 15. REPRODUCIBILITY

| Parameter | Value |
|-----------|-------|
| Random seed | 42 (all operations) |
| File hash | `7bf10a37ce07e72e` (Base.csv) |
| Protocol version | 1.0 |
| Python version | To be recorded at implementation time |
| LightGBM version | To be recorded at implementation time |
| Manifest path | `experiments/phase4_drift_adaptation/artifacts/PHASE4_PROTOCOL_MANIFEST.json` |

---

## 16. LIMITATIONS

1. **BAF is synthetic** — findings may not transfer to real bank account opening systems
2. **Only 8 temporal periods** — limited temporal resolution (no intra-month granularity)
3. **Single dataset** — adaptation conclusions are scoped to BAF Base
4. **Only PSI tested** — other detectors (KS, JS divergence) are not evaluated in primary protocol
5. **Only LightGBM** — no deep learning comparison for adaptation
6. **Single seed** — no cross-seed variance quantification
7. **Fixed architecture** — adaptive and static differ only in training data, not hyperparameters
8. **No entity tracking** — applications cannot be linked across months
9. **No production deployment** — adaptation delay and label latency are simulated
10. **Fraud rate drift** — BAF's fraud increase may partially reflect synthetic data generation choices rather than real behavioral shifts

---

## 17. STOPPING/FREEZE RULES

Once any of the following are computed, the protocol is frozen:
- Month 5 drift statistic (determines retraining)
- Month 4 threshold
- Any month 6 or 7 prediction

After freeze, no changes to:
- PSI threshold
- Feature set
- Hyperparameters
- Model architecture
- Retraining policy
- Statistical methodology
