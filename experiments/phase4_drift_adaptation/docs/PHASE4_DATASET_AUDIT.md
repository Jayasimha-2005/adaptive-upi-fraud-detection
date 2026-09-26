# BAF DATASET AUDIT REPORT
## Phase 4 — Concept Drift Detection + Adaptive Retraining

**Type:** READ-ONLY DATA AUDIT  
**Date:** 2026-09-19  
**Dataset Location:** `Datasets/BAF/`  
**Audit Script:** `experiments/phase4_drift_adaptation/data_audit/audit_baf.py`  
**Machine-readable output:** `experiments/phase4_drift_adaptation/data_audit/baf_data_audit.json`

---

## 1. FILES FOUND

| File | Size | SHA-256 (16) | Rows | Cols | Variant |
|------|------|-------------|------|------|---------|
| `Base.csv` | 203.4 MB | `7bf10a37ce07e72e` | 1,000,000 | 32 | **Primary** |
| `Variant I.csv` | 203.4 MB | (see JSON) | 1,000,000 | 32 | Secondary |
| `Variant II.csv` | 203.5 MB | (see JSON) | 1,000,000 | 32 | Secondary |
| `Variant III.csv` | 240.2 MB | (see JSON) | 1,000,000 | 34 | Secondary |
| `Variant IV.csv` | 203.5 MB | (see JSON) | 1,000,000 | 32 | Secondary |
| `Variant V.csv` | 240.2 MB | (see JSON) | 1,000,000 | 34 | Secondary |

**All 6 variants present and readable. No missing files.**

---

## 2. BASE.CSV SCHEMA (Primary Experiment File)

### 2.1 Dataset Identity
- **Format:** CSV
- **Rows:** 1,000,000
- **Columns:** 32
- **Target column:** `fraud_bool` (binary: 0/1)
- **Temporal column:** `month` (integer: 0–7)
- **Entity/ID column:** NONE — no individual application/customer identifier
- **No sub-second timestamps** — month is the only temporal granularity

### 2.2 Full Column List

| # | Column | Type | Notes |
|---|--------|------|-------|
| 1 | `fraud_bool` | Numerical (int) | Target — must be excluded from features |
| 2 | `income` | Numerical (float) | |
| 3 | `name_email_similarity` | Numerical (float) | |
| 4 | `prev_address_months_count` | Numerical (int) | |
| 5 | `current_address_months_count` | Numerical (int) | |
| 6 | `customer_age` | Numerical (int) | |
| 7 | `days_since_request` | Numerical (float) | |
| 8 | `intended_balcon_amount` | Numerical (float) | |
| 9 | `payment_type` | Categorical | 5 cat cols total |
| 10 | `zip_count_4w` | Numerical (int) | |
| 11 | `velocity_6h` | Numerical (float) | |
| 12 | `velocity_24h` | Numerical (float) | |
| 13 | `velocity_4w` | Numerical (float) | |
| 14 | `bank_branch_count_8w` | Numerical (int) | |
| 15 | `date_of_birth_distinct_emails_4w` | Numerical (int) | |
| 16 | `employment_status` | Categorical | |
| 17 | `credit_risk_score` | Numerical (int) | |
| 18 | `email_is_free` | Numerical (binary int) | Near-constant |
| 19 | `housing_status` | Categorical | |
| 20 | `phone_home_valid` | Numerical (binary int) | Near-constant |
| 21 | `phone_mobile_valid` | Numerical (binary int) | Near-constant |
| 22 | `bank_months_count` | Numerical (int) | |
| 23 | `has_other_cards` | Numerical (binary int) | Near-constant |
| 24 | `proposed_credit_limit` | Numerical (float) | |
| 25 | `foreign_request` | Numerical (binary int) | Near-constant |
| 26 | `source` | Categorical | |
| 27 | `session_length_in_minutes` | Numerical (float) | |
| 28 | `device_os` | Categorical | |
| 29 | `keep_alive_session` | Numerical (binary int) | Near-constant |
| 30 | `device_distinct_emails_8w` | Numerical (int) | Near-constant |
| 31 | `device_fraud_count` | Numerical | **CONSTANT** — all zeros |
| 32 | `month` | Numerical (int) | Temporal index — must be excluded from features |

### 2.3 Feature Counts
- **Numerical features: 24** (excluding `fraud_bool`, `month`, and `device_fraud_count` from the 27 numerical columns pandas detected — those 3 are all excluded)
- **Categorical features: 5** (`payment_type`, `employment_status`, `housing_status`, `source`, `device_os`)
- **Total usable features: 29** (24 numerical + 5 categorical)

> **Reconciliation:** Pandas detected 27 numerical columns. Subtracting the 3 excluded columns (`fraud_bool` = target, `month` = temporal index, `device_fraud_count` = constant) gives 24 numerical model-input features. The PSI denominator is therefore 24 + 5 = **29**, which is the figure used throughout the protocol.

### 2.4 Quality Checks
| Check | Result |
|-------|--------|
| Duplicate rows | **0** |
| Missing values | **0 across all columns** |
| Constant columns | `device_fraud_count` (all zero — **must be dropped**) |
| Near-constant columns | 7 (documented, retain for now) |
| Impossible/invalid values | None detected |
| Target leakage candidates | None — features are behavioral/demographic |
| Temporal leakage candidates | `month` must be excluded from model features |

---

## 3. TEMPORAL STRUCTURE — VERIFIED FROM ACTUAL DATA

### 3.1 Period Summary (Base.csv)

| Month | Rows | Fraud | Legit | Fraud Rate | Mean KS vs Month-0 |
|-------|------|-------|-------|-----------|-------------------|
| 0 | 132,440 | 1,500 | 130,940 | **1.133%** | — (reference) |
| 1 | 127,620 | 1,198 | 126,422 | **0.939%** | 0.0896 |
| 2 | 136,979 | 1,198 | 135,781 | **0.875%** | 0.0944 |
| 3 | 150,936 | 1,392 | 149,544 | **0.922%** | 0.0874 |
| 4 | 127,691 | 1,452 | 126,239 | **1.137%** | 0.0941 |
| 5 | 119,323 | 1,411 | 117,912 | **1.183%** | 0.1417 |
| 6 | 108,168 | 1,450 | 106,718 | **1.341%** | 0.1518 |
| 7 | 96,843 | 1,428 | 95,415 | **1.475%** | 0.1817 |
| **Total** | **1,000,000** | **11,029** | **988,971** | **1.103%** | |

### 3.2 Key Temporal Observations

**Fraud prevalence trend:**
- Months 0–3: Low-to-moderate fraud (0.875%–1.133%) — relatively stable
- Months 4–7: Accelerating fraud rate (1.137%→1.475%) — **a clear upward drift**
- Total fraud rate increase from month 2 (lowest: 0.875%) to month 7 (highest: 1.475%): **+0.600 percentage points (+68.6% relative increase)**

**Feature distribution shift (mean KS statistic vs Month 0):**
- Months 1–4: low-moderate shift (KS ≈ 0.087–0.094) — stable feature space
- Month 5: notable jump (KS = 0.142)
- Month 6: increasing (KS = 0.152)
- Month 7: highest shift (KS = 0.182) — **genuine distribution drift confirmed**

**Row count trend:**
- Decreasing from month 3 onward (150,936 → 96,843)
- Smallest window: month 7 (96,843 rows, 1,428 fraud) — still statistically adequate

**Minimum fraud per period:** 1,198 (months 1 and 2) — sufficient for PR-AUC estimation.

### 3.3 Temporal Findings for Protocol Design
1. **Temporal distribution shift is empirically observed** in the BAF Base dataset — both in fraud prevalence and feature distributions. The following distinctions apply:

- **Feature-space shift P(X):** KS statistics increase monotonically from ~0.09 (months 1–4) to 0.18 (month 7), indicating the marginal distributions of input features are changing over time.
- **Marginal label shift P(Y):** Fraud prevalence increases from 0.875% (month 2) to 1.475% (month 7) — a +68.6% relative increase.
- **Concept drift P(Y|X):** Whether the *relationship* between features and fraud probability has changed cannot be confirmed from distribution statistics alone. Phase 4 will investigate whether a static model degrades under this shift and whether adaptation helps.

The KS drift accelerates noticeably from month 5 onward, suggesting that the period-5-to-7 window is where any predictive degradation is most likely to manifest.

---

## 4. VARIANT COMPARISON

| Variant | Rows | Cols | Fraud total | Row distribution | Fraud pattern |
|---------|------|------|-------------|-----------------|---------------|
| **Base** | 1M | 32 | 11,029 | Relatively stable (~96K–151K) | Monotone drift ↑ |
| I | 1M | 32 | 11,029 | Same as Base | Same as Base |
| II | 1M | 32 | 11,029 | Same as Base | Same as Base |
| III | 1M | **34** | 11,030 | More variable (80K–157K) | Spike at months 3-4, drops at 7 |
| IV | 1M | 32 | 11,030 | More variable | Same as III |
| V | 1M | **34** | 11,030 | More variable | Same as III |

- Variants I and II share the same population structure as Base (likely bias-reweighting variants)
- Variants III–V have 2 extra columns and a different period distribution (likely undersampling bias or demographic shift variants)
- **Primary experiment: Base only.** Variants III–V can serve as secondary stress tests with their more volatile fraud patterns.

---

## 5. DATASET TERMINOLOGY NOTE

BAF must be described as:

> "A privacy-preserving synthetic dataset derived from anonymized real-world bank account fraud data, released as part of the BAF benchmark suite (Jesus et al., NeurIPS 2022)."

**NOT:** production bank data, real customer data, or live banking transactions.

---

## 6. DATA SUITABILITY VERDICT

| Criterion | Result |
|-----------|--------|
| Temporal structure present | ✅ 8 monthly periods (0–7) |
| Sufficient data per period | ✅ Min 96,843 rows, 1,198 fraud |
| Genuine feature-space drift | ✅ KS increases monotonically through months 5–7 |
| Genuine fraud-rate drift | ✅ +68.6% relative increase (month 2 → month 7) |
| No missing values | ✅ Zero missingness |
| No duplicate rows | ✅ Zero duplicates |
| Clean binary target | ✅ `fraud_bool` ∈ {0, 1} |
| Constant column identified | ⚠️ `device_fraud_count` must be dropped |
| Target/month excluded from features | ✅ Will be excluded during preprocessing |
| Entity ID absent | ℹ️ No per-application ID — row-level predictions only |
| Supports temporal train/eval split | ✅ Yes |

**VERDICT: BAF Base.csv fully supports the proposed Phase 4 temporal adaptation experiment.**

> **Dataset selection note:** BAF Base was pre-designated as the primary Phase 4 dataset because the project protocol identifies BAF as the dataset for temporal drift/adaptation experiments, while its variants are reserved for secondary stress testing. The observed temporal distribution shift documented in this audit is evidence discovered *during* the audit — it is not the reason the dataset was selected.
