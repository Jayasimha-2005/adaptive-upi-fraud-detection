# Dataset Suitability Audit
**Generated:** 2026-09-18 | **Scope:** Read-only forensic inspection | **No training, no model changes**

---

## 1. Repository Dataset Inventory (Actual Filesystem)

```
Datasets/
├── BAF/                                  ← 6 files × 1M rows
│   ├── Base.csv             (203 MB)
│   ├── Variant I.csv        (203 MB)
│   ├── Variant II.csv       (203 MB)
│   ├── Variant III.csv      (240 MB)
│   ├── Variant IV.csv       (203 MB)
│   └── Variant V.csv        (240 MB)
├── Credit card Fraud detection/
│   └── credit_card_fraud_10k.csv  (360 KB)   ← synthetic, 10K rows
├── IEEE CIS-20260829T103704Z-1-001/
│   └── IEEE CIS/
│       ├── sample_submission.csv
│       ├── test_identity.csv
│       ├── test_transaction.csv
│       ├── train_identity.csv
│       └── train_transaction.csv
├── Paysim/
│   └── paysim dataset.csv           (471 MB)
├── processed/
│   └── ieee_cis/                    ← processed E1 splits (parquet)
├── raw/                             ← EMPTY
└── ulb_creditcard/
    └── creditcard.csv               (144 MB)   ← ULB, 284K rows
```

**Gitignore:** `Datasets/` is correctly excluded from Git tracking. ✅  
**ULB confirmed at:** `Datasets/ulb_creditcard/creditcard.csv` ✅

---

## 2. Dataset Forensics

### Dataset 1 — IEEE-CIS Transaction Fraud (Primary Benchmark)

| Property | Value |
|---------|-------|
| Source | Kaggle IEEE-CIS Fraud Detection competition |
| Files | train_transaction.csv, train_identity.csv, test_transaction.csv, test_identity.csv |
| Type | **Real-world** (anonymized/obfuscated) |
| Total rows | 590,540 (434,176 train + 77,822 val + 78,542 test) |
| Total columns | ~434 raw (406 retained after E1 preprocessing) |
| Target column | `isFraud` (0 / 1) |
| Fraud count | ~20,663 (3.5% across splits) |
| Missing values | Extensive (~60+ features with NaN) |
| Duplicate rows | None in processed splits |
| Entity identifier | `card1` (used for E2 sequence grouping) |
| Temporal column | `TransactionDT` (seconds from reference) |
| Temporal range | ~6 months |
| Temporal resolution | Second-level |
| Chronological ordering | ✅ Strict (Option A, used in E2) |
| Train/val/test splits | ✅ Already locked (target-based) |
| V features | V1–V339 (Vesta proprietary engineered, NOT PCA) |
| Leakage columns | `isFraud`, `TransactionID`, `TransactionDT` (excluded from features) |
| Experiments using it | **E1, E1-Phase1.5, E2a, E2b (all complete)** |

---

### Dataset 2 — ULB Credit Card Fraud

| Property | Value |
|---------|-------|
| Source | ULB Machine Learning Group / Kaggle |
| File | `creditcard.csv` |
| Type | **Real-world** (European cardholders, Sep 2013) |
| Total rows | **284,807** |
| Unique rows | 283,726 (1,081 exact duplicate rows) |
| Total columns | **31** |
| Column names | Time, V1–V28, Amount, Class |
| Target column | `Class` (0=legit, 1=fraud) |
| Fraud count | **492** |
| Legit count | **284,315** |
| Fraud rate | **0.1727%** |
| Missing values | **0** |
| Duplicate rows | **1,081** (1,822 legit + 32 fraud in duplicates — full-row duplicates across all 31 columns) |
| Duplicate (Time+Amount only) | 4,863 |
| Entity identifier | **NONE** (no card/account ID) |
| Temporal column | `Time` (seconds elapsed from first transaction) |
| Time range | 0 – 172,792 sec = **exactly 48.0 hours** |
| Time is monotone? | ✅ Yes (non-decreasing) |
| Time is strictly increasing? | ❌ No (many ties — 160,215 duplicate Time values) |
| V features | V1–V28 (**PCA-transformed**, original features withheld for privacy) |
| Amount | min=0, mean=88.35, max=25,691.16, std=250.12 |
| Leakage candidates | None obvious (PCA anonymized) |

> [!IMPORTANT]
> **ULB V1–V28 are PCA-transformed and are COMPLETELY DIFFERENT from IEEE-CIS V1–V339.** IEEE-CIS V features are Vesta-proprietary engineered features. ULB V features are principal components of withheld real features. They share only naming convention — not semantics, scale, or interpretation.

---

### Dataset 3 — BAF (Bank Account Fraud, NeurIPS 2022)

| Property | Value |
|---------|-------|
| Source | NeurIPS 2022 / Featurespace |
| Files | Base.csv + Variant I–V.csv |
| Type | **Synthetic** (realistic simulation of bank account fraud) |
| Total rows | **6,000,000** (6 × 1,000,000) |
| Total columns | **32** per file |
| Target column | `fraud_bool` (0/1) |
| Fraud per file | ~11,029–11,030 |
| Fraud rate per file | ~**1.10%** |
| Missing values | None observed in sample |
| Entity identifier | **NONE** (no card/account ID linking applications) |
| Temporal column | `month` (0–7, ordinal integer — 8 months of data) |
| Fraud rate by month | ~0.87%–1.47% (rising trend Base→later months) |
| Variant design | **Each variant introduces a specific concept drift** (covariate shift, prior shift, etc.) |
| V features | **NONE** (32 directly-interpretable features: income, credit score, velocity, etc.) |
| Columns notable | velocity_6h, velocity_24h, velocity_4w (pre-aggregated temporal signals) |

> [!NOTE]
> **BAF variants are NOT duplicates.** Base.csv and Variant I–V.csv have the same schema and same fraud counts but different feature distributions. They were designed for concept drift benchmarking — each variant simulates a different type of distribution shift. This is the most valuable asset for the concept drift phase.

---

### Dataset 4 — PaySim (Mobile Money Fraud Simulation)

| Property | Value |
|---------|-------|
| Source | Synthetic, based on MPESA-like mobile money transactions |
| File | `paysim dataset.csv` |
| Type | **Simulated** (agent-based simulation) |
| Total rows | **6,362,620** |
| Total columns | **11** |
| Target column | `isFraud` (0/1) |
| Fraud count | **8,213** |
| Legit count | **6,354,407** |
| Fraud rate | **0.1291%** |
| Entity identifier | `nameOrig` (sender account) — repeated entities exist |
| Temporal column | `step` (simulated hours, 1–743 ≈ **31 simulated days**) |
| Transaction types | PAYMENT, CASH_OUT, CASH_IN, TRANSFER, DEBIT |
| Fraud in types | Only CASH_OUT and TRANSFER contain fraud (by simulation design) |
| Missing values | 0 |
| V features | **NONE** |

---

### Dataset 5 — "Credit card Fraud detection" (10K synthetic)

| Property | Value |
|---------|-------|
| Source | Unknown — likely a Kaggle demo/toy dataset |
| File | `credit_card_fraud_10k.csv` |
| Type | **Synthetic, toy** |
| Total rows | **10,000** |
| Columns | transaction_id, amount, transaction_hour, merchant_category, foreign_transaction, location_mismatch, device_trust_score, velocity_last_24h, cardholder_age, is_fraud |
| Target column | `is_fraud` |
| Fraud count | 151 (1.51%) |
| Missing values | 0 |
| Duplicate rows | 0 |
| Entity identifier | **NONE** — transaction_id is sequential 1..10000 |
| Temporal column | `transaction_hour` only (0–23, hour of day — no date) |
| V features | **NONE** |
| Real-world data | **No** — 5 merchant categories, manufactured feature values |
| Comparison to ULB | **Completely different dataset**, not a duplicate or subset |

> [!WARNING]
> This dataset is a **toy/demo dataset** (10K rows, 10 manufactured features, no entity tracking, no real timestamps). It is **not a published benchmark** and is not suitable for rigorous fraud detection research. It should not be used as a primary experiment dataset.

---

## 3. Existing Experiments (Read-Only Audit)

| Experiment | Status | Dataset | Key Artifacts |
|-----------|--------|---------|--------------|
| **E1 LightGBM** | ✅ COMPLETE | IEEE-CIS | `experiments/E1_lightgbm/` — model.txt, preprocessing.joblib, predictions.parquet, metrics.json |
| **E1 Phase 1.5** | ✅ COMPLETE | IEEE-CIS | Hardening, bootstrap CI, operational metrics — embedded in metrics.json |
| **E2a GRU (unscaled)** | ✅ COMPLETE | IEEE-CIS | `phase2_gru/artifacts/E2a_unscaled/` |
| **E2b GRU (scaled)** | ✅ COMPLETE | IEEE-CIS | `phase2_gru/artifacts/E2b_scaled/` — test_metrics.json, paired_bootstrap.json |
| **STEP 9 Evaluation** | ✅ COMPLETE | IEEE-CIS | E1 vs E2b paired bootstrap complete. ΔPR-AUC = −0.3584, 95% CI [−0.3756, −0.3395] |

**All existing experiments are READ-ONLY. Zero modifications made.**

---

## 4. Question: Does Adding ULB Require Changing E1?

**Answer: NO.**

E1 is a locked IEEE-CIS LightGBM experiment. ULB is a completely different dataset with different features (PCA V1–V28), different temporal coverage (48 hours), no entity identifiers, and a different fraud mechanism. Adding ULB creates a new, independent experiment pipeline — it does not modify, invalidate, or require rerunning E1.

---

## 5. Question: Does Adding ULB Require Changing E2a or E2b?

**Answer: NO.**

E2a and E2b operate on IEEE-CIS sequences. ULB has **no entity identifier** — it is impossible to group transactions by cardholder and construct 4-step history windows as defined in the locked Phase 2 sequence specification.

If a temporal GRU experiment were desired for ULB, it would need to:
- Use `Time` as a global ordering variable (no entity grouping)
- Define a different sequence construction methodology
- Be documented as a new, separate experiment (e.g., E-ULB-1)

This does not change E2a or E2b.

---

## 6. Gitignore Status

`Datasets/` is correctly excluded from Git tracking. The `ulb_creditcard/creditcard.csv` file is therefore automatically excluded. ✅ No changes needed.

---

*AUDIT ONLY — 2026-09-18 | No training. No model changes. No preprocessing changes.*
