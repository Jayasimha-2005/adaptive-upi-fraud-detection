# Phase 2 — Repository Inventory
## Step 1 Output: Inspection of Existing Artifacts Before Any Phase 2 Code Is Written

**Generated:** 2026-09-17  
**Inspection only — zero files modified**

---

## 1. Repository Root Structure

```
adaptive-upi-fraud-detection/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── src/                          ← E1 source modules [IMMUTABLE]
│   ├── data/                     validate.py, load.py, split.py
│   ├── features/                 ieee_cis_features.py (IEEECISPreprocessor)
│   ├── models/                   lightgbm_baseline.py
│   ├── evaluation/               metrics.py, bootstrap.py, calibration.py
│   └── audit/                    feature_provenance.py, encoding_comparison.py
│
├── experiments/                  ← E1 experiment artifacts [IMMUTABLE]
│   ├── run_phase1.py
│   ├── run_phase15_extend.py
│   ├── E1_lightgbm/              ← FROZEN BASELINE ARTIFACTS
│   └── E1_encoding_comparison/
│
├── reports/                      ← Phase 1 + 1.5 reports [IMMUTABLE]
│   ├── sprint-1.md
│   ├── phase1/
│   └── phase2/
│       └── SEQUENCE_GENERATION_SPEC.md  ← Locked Phase 2 spec
│
├── datasets/processed/ieee_cis/  ← Processed parquets [READ-ONLY]
│   ├── train.parquet             (54.9 MB, 434,176 rows, 409 cols)
│   ├── validation.parquet        (10.5 MB, 77,822 rows, 409 cols)
│   └── test.parquet              (11.5 MB, 78,542 rows, 409 cols)
│
├── dataset_analysis/             ← Forensic analysis outputs [IMMUTABLE]
├── analysis_scripts/             ← Dataset analysis scripts [IMMUTABLE]
├── configs/                      ← E1 configs [IMMUTABLE]
├── tests/                        ← E1 integrity tests [IMMUTABLE]
│
└── phase2_gru/                   ← NEW — ALL PHASE 2 WORK HERE
```

---

## 2. E1 Artifact Inventory (experiments/E1_lightgbm/)

| Artifact | Size | Status | Purpose for Phase 2 |
|----------|------|--------|---------------------|
| `model.txt` | 6.64 MB | FROZEN | Reference only — do not load or modify |
| `preprocessing.joblib` | 0.06 MB | READ-ONLY | May be inspected; copy to phase2_gru/artifacts/preprocessing/ if needed |
| `predictions.parquet` | 3.19 MB | READ-ONLY | Used for E1 vs E2 paired comparison |
| `metrics.json` | tiny | READ-ONLY | Frozen E1 metrics for comparison table |
| `feature_names.json` | tiny | READ-ONLY | **Critical** — defines exact 406 E1 features |
| `split_meta.json` | tiny | READ-ONLY | Confirms split boundaries |
| `run_metadata.json` | tiny | READ-ONLY | Frozen E1 experiment parameters |
| All `.png` plots | — | FROZEN | Do not regenerate or overwrite |

---

## 3. Confirmed Split Boundaries (from split_meta.json)

```
TRAIN:
  TransactionDT range : 86,400 – 10,972,793
  Boundary (inclusive): TransactionDT <= 10,972,800
  Rows  : 434,176
  Fraud : 15,252  (3.513%)
  Days  : 1.0 – 127.0

VALIDATION:
  TransactionDT range : 10,972,801 – 13,391,998
  Boundary (inclusive): 10,972,800 < TransactionDT <= 13,392,000
  Rows  : 77,822
  Fraud : 2,637   (3.389%)
  Days  : 127.0 – 155.0

TEST:
  TransactionDT range : 13,392,056 – 15,811,131
  Boundary (inclusive): TransactionDT > 13,392,000
  Rows  : 78,542
  Fraud : 2,774   (3.532%)
  Days  : 155.0 – 183.0

Zero TransactionID overlap between any two splits: CONFIRMED
```

---

## 4. Processed Parquet Schema (409 columns)

The processed parquets contain **409 columns** = 406 model features + 3 special columns:

| Column | Type | Role in Phase 2 |
|--------|------|-----------------|
| `TransactionID` | int64 | Entity tracking, split assignment, join key |
| `TransactionDT` | float64 | **Temporal ordering** — must sort by this |
| `isFraud` | int64 | **Target label only** — NEVER used as feature |
| 406 model features | mixed | GRU input vector |

---

## 5. E1 Feature Matrix — Confirmed (406 features)

Confirmed from `experiments/E1_lightgbm/feature_names.json`:

| Group | Count | Examples | Phase 2 Notes |
|-------|-------|---------|---------------|
| V-columns | 292 | V1–V339 (non-contiguous) | UNKNOWN_OPAQUE provenance |
| id-columns | 38 | id_01–id_38 | Device/identity signals |
| Missingness flags | 27 | D2_missing, addr1_missing… | Binary flags for missing values |
| C-columns | 14 | C1–C14 | Backward-looking count aggregates |
| D-columns | 10 | D1–D5, D10, D11, D15 | Time-delta features |
| M-columns | 9 | M1–M9 | Match/mismatch flags |
| Engineered temporal | 3 | hour_sin, hour_cos, day_index | Cyclical time encoding |
| Transaction base | 13 | TransactionAmt, ProductCD, card1–6, addr1–2, dist1, P/R_emaildomain | Core transaction fields |

**Safety confirmations:**
- `isFraud` → NOT in feature list ✅
- `TransactionID` → NOT in feature list ✅
- `TransactionDT` → NOT in feature list ✅
- `card1` → IS in feature list (see **Critical Ambiguity** below)

---

## 6. Frozen E1 Baseline Metrics

```
Source: experiments/E1_lightgbm/metrics.json + run_metadata.json
Random seed: 42
Decision threshold: 0.616521  (selected on validation, frozen before test)

TEST SET METRICS (immutable):
  PR-AUC            = 0.531731
  ROC-AUC           = 0.898993
  Precision         = 0.581810
  Recall            = 0.493511
  F1                = 0.534035
  MCC               = 0.520301
  Balanced Accuracy = 0.740262
  FPR               = 0.012987
  FNR               = 0.506489

  Confusion matrix:
    TN = 74,784 | FP = 984
    FN = 1,405  | TP = 1,369

BOOTSTRAP CI (2000 valid samples):
  PR-AUC: mean=0.531499  [0.512586 – 0.550448]

OPERATIONAL METRICS:
  Recall @ 1% FPR    = 0.4618
  Precision @ Top-1K = 0.8410

VALIDATION PR-AUC (for reference): 0.584935
```

---

## 7. Predictions Parquet (for Paired Comparison)

```
Path: experiments/E1_lightgbm/predictions.parquet
Shape: 156,364 rows × 6 columns

Columns:
  - TransactionID      : join key
  - TransactionDT      : temporal reference
  - isFraud            : true label
  - fraud_probability  : E1 raw probability score (used for paired bootstrap)
  - prediction         : binary decision at threshold=0.616521
  - split              : 'validation' or 'test'

Counts:
  validation : 77,822 rows
  test       : 78,542 rows
```

---

## 8. ⚠️ CRITICAL AMBIGUITY — card1 as GRU Input Feature

**Finding:** `card1` IS present in the E1 model's 406 feature list.

**The problem for Phase 2:**

`card1` serves two different roles simultaneously:

| Role | Usage |
|------|-------|
| **Entity key** | Used to group transactions into sequences (e.g., "all transactions from card #12345") |
| **Feature input** | Currently used as one of the 406 LightGBM features (after LabelEncoding) |

**Why this matters for GRU:**

- If `card1` is included as a GRU input feature, the model can memorize card identity rather than learning behavioral patterns. A GRU that sees card #12345 repeatedly may simply learn "card #12345 = high fraud" rather than learning that the *sequence of behaviors* is anomalous.
- This is qualitatively different from LightGBM, where card1 is just one static feature across 406.

**Three defensible treatments:**

| Option | Description | Risk |
|--------|-------------|------|
| **E2a** (Recommended): card1 excluded from GRU inputs | Used only as entity key; 405 features → GRU | Reduces feature count by 1; cleaner behavioral learning |
| **E2b**: card1 included as GRU input | Matches E1's feature set exactly | Entity memorization risk; may inflate metrics |
| **E2ab**: Run both, report both | Full sensitivity analysis | Higher computational cost |

> **This ambiguity CANNOT be silently resolved. Review required before proceeding to STEP 6.**

---

## 9. Temporal Gap Feature — Resolution Needed

The locked `SEQUENCE_GENERATION_SPEC.md` mentions a possible gap feature. From inspection of the spec:

```
reports/phase2/SEQUENCE_GENERATION_SPEC.md — key section:
  Window size: L=5 (4 history + 1 prediction)
  Entity key: card1
  Causal history rule: prior-split history available
```

The spec does not conclusively specify whether a `time_gap` feature (seconds since previous transaction) is included.

**Impact on dimensionality:**
- Without gap feature: GRU input dim = 405 or 406 (depending on card1 decision)
- With gap feature: GRU input dim = 406 or 407

> **This must be resolved before sequence builder is implemented.**

---

## 10. Dependencies Required for Phase 2

### Read-only access (no modification):
```
datasets/processed/ieee_cis/train.parquet       → sequence construction
datasets/processed/ieee_cis/validation.parquet  → sequence construction
datasets/processed/ieee_cis/test.parquet        → sequence construction
experiments/E1_lightgbm/feature_names.json      → feature list
experiments/E1_lightgbm/split_meta.json         → split boundaries
experiments/E1_lightgbm/predictions.parquet     → paired E2 vs E1 comparison
experiments/E1_lightgbm/metrics.json            → frozen E1 baseline numbers
experiments/E1_lightgbm/preprocessing.joblib    → reference for feature encoding
```

### New packages required (not in current requirements.txt):
```
torch >= 2.0.0       # GRU implementation
pytest >= 7.0.0      # test suite
```

### No modification to:
```
src/                 # all E1 source modules
experiments/E1_*     # all E1 artifacts
reports/             # all Phase 1/1.5 reports
configs/             # E1 config
tests/test_phase1.py # E1 integrity tests
.gitignore           # existing root gitignore
requirements.txt     # E1 requirements
```

---

## 11. Open Questions Before STEP 6 Can Proceed

> The following must be answered **before any code is written for Phase 2**.

| # | Question | Impact |
|---|---------|--------|
| **Q1** | Should `card1` be included or excluded from GRU input features? | Changes input dimensionality (405 vs 406) |
| **Q2** | Should a temporal gap feature (`time_since_prev_transaction`) be added? | Changes input dimensionality (+1) |
| **Q3** | Which parquet path should be canonical? (`datasets/` or `Datasets/`) | Both exist and are identical — lowercase `datasets/` recommended |

---

## 12. Summary: What Is Safe to Proceed With

After inspection, the following are unambiguously confirmed and Phase 2 can proceed for these:

- ✅ Split boundaries (train_dt_max=10,972,800 / val_dt_max=13,392,000)
- ✅ Entity key = `card1`
- ✅ Sequence length L=5 (4 history + 1 target)
- ✅ Target = `isFraud` of the last transaction in the window
- ✅ `isFraud` NOT used as input
- ✅ Feature count = 406 (from feature_names.json)
- ✅ Processed parquets available at `datasets/processed/ieee_cis/`
- ✅ Canonical processed parquets have 409 cols (406 features + TransactionID + TransactionDT + isFraud)
- ✅ E1 predictions available for paired bootstrap comparison

**BLOCKED pending your answers to Q1, Q2 above.**

---

*Generated by: STEP 1 repository inspection | Phase 2 STEP 1 ONLY | No files modified*  
*Next step: Await review of Q1 and Q2 before proceeding to STEP 6 (sequence builder)*
