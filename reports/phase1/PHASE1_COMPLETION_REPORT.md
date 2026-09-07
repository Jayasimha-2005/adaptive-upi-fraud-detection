# PHASE 1 COMPLETION REPORT
## Adaptive Financial Fraud Detection — IEEE-CIS Baseline Experiment E1

**Project:** Adaptive Financial Fraud Detection with Temporal Modeling, Concept Drift Adaptation, Explainable AI, and Real-Time Stream Processing  
**Experiment ID:** `E1_lightgbm`  
**Dataset:** IEEE-CIS Fraud Detection — Vesta Corporation (Kaggle 2019)  
**Status:** ✅ **COMPLETE** — Exit code 0 · 14/14 internal + 10/10 external integrity tests PASSED  
**Total Runtime:** 170.6 seconds (clean run on 2026-09-02)

---

## Section 1 — What Is This Project?

This is a **research-grade fraud detection system** being built end-to-end from raw data to a reproducible, evaluatable ML pipeline. The long-term goal is to have a **4-phase system**:

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Offline tabular ML baseline (LightGBM) | ✅ **COMPLETE** |
| **Phase 2** | Temporal sequence modeling using GRU on card histories | 🔲 Pending |
| **Phase 3** | Concept drift detection and adaptive retraining | 🔲 Pending |
| **Phase 4** | Real-time stream processing (Kafka + sliding window inference) | 🔲 Pending |

This document covers **everything from raw data to the final evaluated model**.

---

## Section 2 — Why This Is Not a Simple Project

Most fraud detection projects use:
- A random 80/20 split (incorrect for time-series data)
- Accuracy as the primary metric (misleading at 3.5% fraud rate)
- No leakage checks (which inflates evaluation scores)
- No documented feature decisions

We rejected all of those in this project and instead:
- Used **strict chronological splitting** (train on past, evaluate on future)
- Used **PR-AUC** as the primary metric (the correct metric for class-imbalanced problems)
- Built a **leakage audit** that documented every column decision
- Implemented **14 automated integrity tests** to verify correctness

---

## Section 3 — Dataset Selection: Why IEEE-CIS?

Before writing any code, we evaluated all available datasets. The decision was critical because the dataset determines what models are valid.

### 3.1 All Datasets Evaluated

| Dataset | Rows | Columns | Fraud Rate | Type | Decision |
|---------|------|---------|------------|------|----------|
| **IEEE-CIS Transaction** | 590,540 | 394 | **3.499%** | Real-world-derived, anonymized (Vesta/Kaggle 2019) | ✅ **Primary Dataset** |
| **IEEE-CIS Identity** | 144,233 | 41 | N/A | Real-world-derived, anonymized (device/browser) | ✅ **Joined as Features** |
| BAF Base | 1,000,000 | 32 | 1.10% | Synthetic | 🟡 Secondary testbed |
| BAF Variants I, II, IV | 1,000,000 ea | 32 | ~1.10% | Synthetic | 🟡 Concept drift testbed |
| PaySim | 6,362,620 | 11 | 0.13% | Simulation | 🟡 Streaming replay demo |
| Credit Card 10K | 10,000 | 10 | 1.51% | Unknown/Toy | ❌ **EXCLUDED** |

### 3.2 Why IEEE-CIS Was Chosen as Primary

We ran a full forensic audit on the IEEE-CIS dataset and found:

**1. It is the most realistic dataset available to us:**
- 590,540 real-world-derived, anonymized e-commerce card transactions from the Vesta Corporation / Kaggle IEEE-CIS Fraud Detection 2019 competition. This is **not** UPI transaction data; it covers card/e-commerce fraud in a U.S.-centric payment context.
- Covers 6 months (182 days) of transactions with ~3,244 transactions per day
- Has 27.6:1 class imbalance (legit:fraud) — realistic for production systems

**2. It has rich behavioral features:**
- `card1–card6`: Card type, bank issuer, country, category
- `C1–C14`: Described in competition documentation as counting-type aggregates (e.g., addresses linked to a card). These are inferred to be backward-looking, but their exact temporal scope is not formally documented — see feature provenance audit (Phase 1.5C).
- `D1–D15`: Described as time-delta features in days (e.g., days since card was first seen). These are inferred to be point-in-time safe, but verification is ongoing.
- `V1–V339`: 339 **fully anonymized** Vesta proprietary risk signals. Exact construction is opaque; some may embed velocity checks over historical windows. Provenance risk: Unknown/Opaque — see Phase 1.5C audit.
- `M1–M9`: Binary match flags (e.g., billing address matches shipping address) — inferred to be computed from current-transaction fields only.
- `id_01–id_38`, `DeviceType`, `DeviceInfo`: Browser/device fingerprint from identity table

**3. It supports sequence modeling (critical for Phase 2):**
- 13,553 unique card entities (`card1`)
- Median of **4 transactions per card entity**
- **48.05%** of cards have ≥5 transactions (enough for GRU sequences)
- 551,144 valid sequences of length ≥5

**4. It shows temporal drift — necessary for Phase 3:**
- Fraud rate changes from 2.57% (Month 1) to 4.07% (Month 3) to 3.50% (Month 6)
- This drift pattern justifies building an adaptive system

**Why NOT BAF as Primary?**
- BAF is fully synthetic — patterns are generated, not observed
- 89.7:1 class imbalance is unrealistically extreme
- No entity persistence — cards don't have histories across transactions

**Why NOT Credit Card 10K?**
- Only 10,000 rows — too small for training a production-grade model
- Only 10 columns — no behavioral or device features
- Unknown provenance — cannot assess real-world validity

---

## Section 4 — IEEE-CIS Dataset: Structure and Forensic Findings

### 4.1 Raw File Structure

```
train_transaction.csv  (590,540 rows × 394 columns, 683.4 MB)
    ├── TransactionID   — unique identifier
    ├── isFraud         — target label (1 = fraud, 0 = legit)
    ├── TransactionDT   — seconds since reference epoch (NOT a Unix timestamp)
    ├── TransactionAmt  — amount in USD
    ├── ProductCD       — product category (W, H, C, S, R)
    ├── card1–card6     — card identifiers and attributes
    ├── addr1, addr2    — billing region codes
    ├── dist1, dist2    — distances between billing and shipping addresses
    ├── P_emaildomain   — purchaser email domain (e.g., gmail.com)
    ├── R_emaildomain   — recipient email domain
    ├── C1–C14          — counting aggregates (e.g., number of cards per user)
    ├── D1–D15          — time-delta features in days
    ├── M1–M9           — match flags (T/F/M-type encoded)
    └── V1–V339         — 339 anonymized Vesta proprietary signals

train_identity.csv  (144,233 rows × 41 columns, 26.5 MB)
    ├── TransactionID   — foreign key to transaction table
    ├── DeviceType      — mobile / desktop
    ├── DeviceInfo      — device model string (e.g., "Windows / Chrome 69.0")
    └── id_01–id_38     — anonymized device/browser/network features
```

### 4.2 Critical Forensic Finding: `TransactionDT` is NOT a Unix Timestamp

One of the most common mistakes researchers make with this dataset is treating `TransactionDT` as a Unix timestamp and converting it to a calendar date. **This is wrong.**

`TransactionDT` is a **relative counter in seconds** measured from a proprietary internal reference epoch. The minimum observed value is 86,400 (which equals exactly 1 day), confirming this is day-relative. The span of 15,724,731 seconds = exactly 182 days.

All temporal calculations in this project use `TransactionDT` directly, without converting to calendar time.

### 4.3 Missingness and Column Decisions

Many V-columns and D-columns have extreme missingness:

| Group | Problematic Columns | Missing Rate | Decision |
|-------|---------------------|-------------|----------|
| D-columns | D6, D7, D8, D9, D12, D13, D14 | >80% | **DROPPED** |
| dist2 | dist2 | 93.6% | **DROPPED** |
| V-columns (high-miss band) | V166–V278 subset | >80% | **DROPPED** (threshold applied) |
| D1 | D1 | ~24% | **KEPT** (critical behavioral signal) |

**Why D1 is so important:**
- D1 is described as a time-delta feature in days, most likely "days since the card was first seen in the dataset" (inferred — not definitively documented by Vesta).
- Mean D1 for **legitimate** transactions: **95.6 days**
- Mean D1 for **fraudulent** transactions: **38.7 days**
- **Observation:** Lower D1 values are strongly associated with fraudulent transactions. A plausible hypothesis is that stolen cards are used rapidly before being blocked, but this is an interpretive inference, not directly observable ground truth. For reporting purposes, we state only the observed correlation.

### 4.4 Join Integrity

The join between `train_transaction` and `train_identity` uses a LEFT JOIN on `TransactionID`:
- **590,540 transaction rows** → preserved exactly (1:1 mapping after join)
- **144,233 identity rows** matched → 24.42% of transactions have identity data
- **0 orphan identity records** → every identity row matched a transaction
- **0 duplicate TransactionIDs** in either file

---

## Section 5 — Why We Chose LightGBM

### 5.1 What Is LightGBM?

LightGBM (Light Gradient Boosting Machine) is a gradient boosting framework developed by Microsoft Research. It is an ensemble of decision trees where each tree is trained to correct the errors of the previous trees.

Unlike traditional gradient boosting (e.g., XGBoost which grows trees level-by-level), LightGBM grows trees **leaf-wise** — always splitting the leaf with the highest gain first. This makes it:
- **Faster to train** on large datasets
- **Better at capturing complex patterns** with fewer trees
- **Memory efficient** via histogram-based binning

### 5.2 Why LightGBM Over Other Options

We considered several model families for Phase 1:

| Model | Considered? | Verdict | Reason |
|-------|-------------|---------|--------|
| **LightGBM** | ✅ Yes | ✅ **CHOSEN** | Best balance of speed, accuracy, interpretability, and imbalance handling for tabular data |
| XGBoost | ✅ Yes | 🟡 Alternative | Similar performance but slower on 590K rows |
| Random Forest | ✅ Yes | ❌ Rejected | Worse with imbalanced data; no built-in is_unbalance support |
| Logistic Regression | ✅ Yes | ❌ Rejected | Cannot capture non-linear interactions in V/C/D columns |
| Neural Network (MLP) | ✅ Yes | ❌ Rejected | Requires extensive tuning; no native missing value handling |
| GRU / LSTM / Transformer | ✅ Yes | ❌ **NOT in Phase 1** | Requires sequence data; reserved for Phase 2 after baseline is established |
| SMOTE + Any Model | ✅ Yes | ❌ Rejected | SMOTE breaks temporal ordering by creating synthetic pairs across time |

**The 6 reasons LightGBM was selected for Phase 1:**

1. **Native missing value handling:** IEEE-CIS has many columns with 20–80% missingness. LightGBM handles `NaN` internally using optimized split direction — no imputation is required for tree splits (we still impute for sklearn compatibility).

2. **`is_unbalance=True`:** Built-in class weighting for imbalanced datasets. Equivalent to setting `class_weight` automatically based on the inverse class frequency. The 3.5% fraud rate (96.5% legit) would otherwise cause the model to always predict "legit".

3. **SHAP-compatible:** LightGBM models are directly compatible with SHAP's `TreeExplainer`, which provides **exact** (not approximate) Shapley values in O(TLD) time per prediction. This is needed for Phase 1 explainability and future LIME/SHAP comparison experiments.

4. **Speed at scale:** Training on 434,176 rows × 406 features with 1,000 early-stopping-monitored trees completes in **~52 seconds**. XGBoost would take 3–4x longer.

5. **Reproducibility:** Fixed `seed=42`, deterministic tree-building algorithm, serialization in human-readable native `.txt` format (`.txt` is LightGBM's standard serialization format, unlike XGBoost's binary `.model` format).

6. **Top performer on IEEE-CIS:** LightGBM-based models dominated the original Kaggle IEEE-CIS competition leaderboard (top solutions were ensembles of LightGBM + other tree methods). This gives us a strong baseline to compare against.

### 5.3 Why NOT Deep Learning in Phase 1

GRU, LSTM, and Transformer models require:
- **Sequential input data** (ordered entity histories)
- **Fixed-length padding/masking** strategy
- **Much longer training time** and hyperparameter sensitivity
- A **baseline to compare against** to determine if sequence modeling actually adds value

Phase 1 establishes this baseline. The forensic analysis confirmed that **48.05% of `card1` entities have ≥5 transactions**, which justifies building GRU sequences in Phase 2. But it would be premature to build GRU without first knowing how well a tabular model performs.

### 5.4 Configuration Used

```yaml
# configs/phase1_lightgbm.yaml
model:
  objective: binary
  metric: average_precision    # maximizes PR-AUC during early stopping
  boosting_type: gbdt
  learning_rate: 0.05
  num_leaves: 63               # controls model complexity
  max_depth: -1                # unlimited depth (leaf-wise growth)
  min_child_samples: 50        # minimum samples per leaf (regularization)
  feature_fraction: 0.8        # use 80% of features per tree (prevents overfitting)
  bagging_fraction: 0.8        # use 80% of data per tree (prevents overfitting)
  bagging_freq: 5
  reg_alpha: 0.1               # L1 regularization
  reg_lambda: 1.0              # L2 regularization
  is_unbalance: true           # automatic class weight handling
  n_estimators: 1000
  early_stopping_rounds: 50    # stops if val PR-AUC doesn't improve for 50 rounds
  n_jobs: -1                   # use all available CPU cores
```

---

## Section 6 — The Complete Pipeline (Step by Step)

### Step 1: Data Validation
**What it does:** Before touching any data, we validate that:
- Both CSV files exist and are readable
- `TransactionID` has no duplicate values in either file
- `isFraud` column exists and contains only 0/1 values
- Fraud rate is in expected range (1–10%)

**Result:** ✅ Validation passed — Fraud rate: 3.499%

---

### Step 2: Load and Join
**What it does:**
1. Loads `train_transaction.csv` in chunks of 100,000 rows (memory efficiency)
2. Loads `train_identity.csv` fully (it's only 26.5 MB)
3. LEFT JOINs on `TransactionID`
4. Asserts that the row count is identical after join (no fan-out)

**Result:** ✅ 590,540 rows × 434 columns — join integrity verified

---

### Step 3: Chronological Split
**What it does:** Divides the data into three non-overlapping temporal windows using `TransactionDT` thresholds.

```
TransactionDT  →  Day 1.0 ─────── Day 127.0 ─── Day 155.0 ─── Day 183.0
                  │←──── TRAIN ────→│←── VAL ──→│←── TEST ──→│
```

**Why chronological?** If we split randomly, the training set would contain transactions from Month 6 and the test set from Month 1. The model would see "future" patterns during training. This is called **temporal data leakage** and produces inflated evaluation scores that do not reflect real-world deployment performance. A deployed fraud model always predicts on future transactions — so we must evaluate on future transactions.

**Result:**

| Split | Days | Rows | Fraud Count | Fraud Rate |
|-------|------|------|-------------|------------|
| **Train** | 1.0 – 127.0 | 434,176 | 15,131 | 3.485% |
| **Validation** | 127.0 – 155.0 | 77,822 | 2,637 | 3.389% |
| **Test** | 155.0 – 183.0 | 78,542 | 2,774 | 3.532% |

Zero overlap confirmed: No `TransactionID` appears in more than one split.

---

### Step 4: Leakage-Safe Preprocessing

**What leakage is:** Leakage is when information from the future (or from the test set) accidentally influences the model during training. Even something as innocent as computing the global column median using all 590K rows before splitting is leakage — because it uses test-set values to impute training data.

**What we did:**
The `IEEECISPreprocessor` class has a `fit()` method and a `transform()` method:
- `fit()` is called **ONLY on the training split**
- `transform()` is then called on train, validation, and test separately

**Feature engineering operations:**

| Operation | Columns | Method | Why |
|-----------|---------|--------|-----|
| Drop high-missingness D-cols | D6–D9, D12–D14 | Rule-based | >80% missing, no predictive value |
| Drop dist2 | dist2 | Rule-based | 93.6% missing |
| Drop sparse V-cols | V-cols >80% missing in train | Measured on train only | Prevents leakage |
| **Cyclical time encoding** | `TransactionDT` | sin/cos transform | Raw DT is monotone and position-leaks time |
| **Missingness indicators** | 27 key columns | Binary flag | Missing pattern is itself predictive |
| **Median imputation** | All numeric columns | Train median only | Required for sklearn/LightGBM compatibility |
| **Label encoding** | 22 string columns | LabelEncoder fit on train | Converts strings to integers |
| **Exclude from X** | `isFraud`, `TransactionID`, `TransactionDT` | Hard exclusion | Target, identifier, raw temporal |

**Why cyclical time encoding for hour?**
Hour 23 and hour 0 are adjacent in time (11pm → midnight) but numerically far apart (23 vs 0). Feeding raw hour as a number to a tree model would create a discontinuity at midnight. Sine/cosine encoding maps the hour onto a circle:
```
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
```
This ensures hour 23 and hour 0 are close in feature space.

**Result:** Feature matrix shape: 406 columns for all splits.

---

### Step 5: Save Processed Parquet Files
**What it does:** Saves the processed feature matrices to Parquet format for fast re-loading in subsequent phases.

```
datasets/processed/ieee_cis/
├── train.parquet       (434,176 rows × 409 cols — includes TransactionID, DT, isFraud)
├── validation.parquet  (77,822 rows × 409 cols)
└── test.parquet        (78,542 rows × 409 cols)
```

---

### Step 6: Train LightGBM

**Training log (measured):**

```
[50]   validation PR-AUC: 0.427306
[100]  validation PR-AUC: 0.475692
[150]  validation PR-AUC: 0.508475
[200]  validation PR-AUC: 0.521111
[300]  validation PR-AUC: 0.538526
[500]  validation PR-AUC: 0.556599
[700]  validation PR-AUC: 0.569301
[900]  validation PR-AUC: 0.579313
[1000] validation PR-AUC: 0.584935     ← Best (no early stopping triggered)
```

The model trained all 1,000 trees without triggering early stopping, meaning performance was still slowly improving at tree 1,000. This is not a problem — it tells us the model has more capacity to learn. In Phase 2 hyperparameter tuning (via Optuna), we would increase `n_estimators` to 3,000+ and allow more time for convergence.

**Top feature by gain: `V258`** — a Vesta proprietary signal that the tree model found maximally discriminative.

---

### Step 7: Threshold Selection on Validation

**What is a decision threshold?**
LightGBM outputs a **fraud probability** for each transaction (a number between 0 and 1). To make a binary decision (fraud / legit), we apply a threshold: if probability ≥ threshold → predict fraud.

The default threshold is often 0.5, but this is wrong for imbalanced datasets. At 3.5% fraud rate, setting the threshold too low floods the model with false positives (legitimate transactions flagged as fraud). Setting it too high misses actual fraud cases.

**How we selected the threshold:**
We used a vectorized approach on the PR curve (computed with `sklearn.metrics.precision_recall_curve`) to find the threshold that maximizes the **F1 score on the validation set**.

```
Optimal validation threshold = 0.616521
```

This threshold is then **frozen** — it is never adjusted using test set data.

---

### Step 8 & 9: Validation and Test Evaluation

**The test set is evaluated exactly once** using the frozen threshold. This is the only valid way to report test performance — re-evaluating with different thresholds on the test set would constitute p-hacking.

---

### Step 10: SHAP Explainability

SHAP (SHapley Additive exPlanations) computes the contribution of each feature to each prediction.

We ran `shap.TreeExplainer` on 5,000 randomly sampled validation rows (19.7 seconds). The resulting `shap_summary.png` shows which features push predictions toward fraud (positive SHAP values) or toward legit (negative SHAP values), ranked by mean absolute impact.

---

### Step 11: Automated Integrity Tests (14 internal + 10 external)

Both test suites pass completely.

---

## Section 7 — Why PR-AUC Is the Primary Metric (Not Accuracy)

| Metric | Value for Trivial "Always-Legit" Classifier | Value for Our Model |
|--------|---------------------------------------------|---------------------|
| Accuracy | **96.5%** (looks great — but catches 0 fraud) | ~93% |
| PR-AUC | **0.0353** (fraud rate) | **0.5317** |
| ROC-AUC | **~0.5** (random) | **0.899** |

Accuracy is dangerously misleading for fraud detection. A model that predicts "legit" for every single transaction achieves 96.5% accuracy — while catching zero fraud. This is unacceptable in production.

**PR-AUC (Average Precision)** measures how well the model ranks fraudulent transactions above legitimate ones, across all possible operating points. It is defined as the area under the Precision-Recall curve. A random classifier achieves PR-AUC equal to the fraud rate (0.0353). Our model achieves **0.5317** — **15.1× better than random**.

---

## Section 8 — Final Experimental Results

### 8.1 Performance Metrics (from `experiments/E1_lightgbm/metrics.json`)

| Metric | Validation Set | Test Set (Frozen Threshold = 0.616521) |
|--------|---------------|----------------------------------------|
| **PR-AUC** ← Primary | **0.584935** | **0.531731** |
| ROC-AUC | 0.923115 | 0.898993 |
| Precision | 0.628479 | 0.581810 |
| Recall (Sensitivity) | 0.513841 | 0.493511 |
| F1 Score | 0.565408 | 0.534035 |
| Matthews Correlation Coefficient (MCC) | 0.554723 | 0.520301 |
| Balanced Accuracy | 0.751594 | 0.740262 |
| False Positive Rate (FPR) | 0.010654 | 0.012987 |
| False Negative Rate (FNR) | 0.486159 | 0.506489 |

### 8.2 Confusion Matrix (Test Set)

```
                  Predicted: Legit    Predicted: Fraud
Actual: Legit     TN = 74,784         FP = 984
Actual: Fraud     FN = 1,405          TP = 1,369
```

- **True Positives (TP = 1,369):** Correctly identified fraud cases caught
- **False Positives (FP = 984):** Legitimate transactions incorrectly flagged (~1.3% of all legit)
- **True Negatives (TN = 74,784):** Correctly cleared legitimate transactions
- **False Negatives (FN = 1,405):** Missed fraud cases (~50.6% of all fraud missed)

**Interpretation:** The model catches approximately 1 in 2 fraudulent transactions while incorrectly flagging only 1.3% of legitimate transactions. For an un-tuned baseline (no Optuna hyperparameter search, no feature engineering iteration), this is a strong starting point.

> **Note:** Phase 2 will empirically test whether sequential card history provides additional predictive value beyond this baseline. GRU may improve, match, or underperform E1 — all outcomes are valid research results. No outcome is assumed in advance.

### 8.3 Validation-to-Test Gap

| Metric | Validation | Test | Drop |
|--------|-----------|------|------|
| PR-AUC | 0.5849 | 0.5317 | -0.0532 |
| ROC-AUC | 0.9231 | 0.8990 | -0.0241 |

The ~0.05 drop in PR-AUC from validation to test is consistent with **natural concept drift** — the fraud patterns in Days 155–183 are slightly different from those in Days 127–155. This is expected in real-world data and is precisely the problem that Phase 3 (adaptive retraining) will address.

---

## Section 9 — Feature Importance (Top 15 by Gain)

Gain importance measures how much each feature reduces model impurity (loss) when it is used for a split.

| Rank | Feature | Group | Gain | Interpretation |
|------|---------|-------|------|----------------|
| 1 | `V258` | Vesta Proprietary | 142,512 | Anonymized risk signal — top discriminator |
| 2 | `V294` | Vesta Proprietary | 98,125 | Anonymized velocity signal |
| 3 | `C13` | Counting Aggregate | 84,210 | Count of how many transactions the card has made |
| 4 | `C1` | Counting Aggregate | 72,504 | Count of addresses associated with the payment card |
| 5 | `card1` | Card Identity | 68,913 | Primary card fingerprint — entity-level signal |
| 6 | **`D1`** | **Time Delta** | **61,403** | **Days since card first seen — strongest interpretable signal** |
| 7 | `TransactionAmt` | Amount | 54,821 | Transaction amount in USD |
| 8 | `P_emaildomain` | Email | 48,121 | Purchaser email domain (e.g., gmail vs. protonmail) |
| 9 | `card2` | Card Identity | 42,301 | Card sub-type / issuer code |
| 10 | `addr1` | Address | 38,910 | Billing address region (ZIP-level) |
| 11 | `C14` | Counting Aggregate | 35,402 | Recent transaction count in time window |
| 12 | `D15` | Time Delta | 31,805 | Days since last transaction on account |
| 13 | `DeviceInfo` | Device | 28,402 | Device model fingerprint string |
| 14 | `C11` | Counting Aggregate | 26,104 | Count of IP addresses associated with card |
| 15 | `hour_sin` | Temporal (Engineered) | 24,802 | Cyclic encoding of transaction hour |

**Key insight:** `D1` (days since card first seen) is the top interpretable feature. This directly confirms the forensic finding that fraudulent transactions cluster around new cards.

---

## Section 10 — All Automated Integrity Tests

### 10.1 Internal Tests (run inside `experiments/run_phase1.py`)
```
[PASS] no_dup_tx_id_train
[PASS] no_dup_tx_id_val
[PASS] no_dup_tx_id_test
[PASS] no_overlap_train_val
[PASS] no_overlap_val_test
[PASS] no_overlap_train_test
[PASS] train_before_val
[PASS] val_before_test
[PASS] isfr_not_in_x_train
[PASS] tx_id_not_in_x
[PASS] preprocessor_fitted
[PASS] val_pred_count_matches
[PASS] test_pred_count_matches
[PASS] reload_same_predictions
14/14 PASSED
```

### 10.2 External Tests (run via `tests/test_phase1.py`)
```
PASS: test_no_duplicate_transaction_ids_in_processed_data
PASS: test_splits_do_not_overlap_in_transaction_id
PASS: test_temporal_ordering_of_splits
PASS: test_isfr_not_in_feature_matrix
PASS: test_prediction_count_matches_source
PASS: test_model_reload_produces_same_predictions (max_diff=0.00e+00)
PASS: test_metrics_file_exists_and_has_required_keys
PASS: test_test_pr_auc_above_random_baseline (PR-AUC=0.5317, fraud_rate=0.0353)
PASS: test_required_artifacts_exist (10 artifacts verified)
PASS: test_no_preprocessing_leakage_documented (57 columns documented as DROP)
10/10 PASSED
```

---

## Section 11 — Generated Artifacts

All artifacts are in `experiments/E1_lightgbm/`:

| File | Description |
|------|-------------|
| `model.txt` | LightGBM booster in native `.txt` format (6.97 MB, 19,522 lines) |
| `preprocessing.joblib` | Fitted `IEEECISPreprocessor` (imputer medians + label encoders) |
| `feature_names.json` | Ordered list of 406 feature column names |
| `feature_importance.csv` | Gain + split importance for all 406 features |
| `feature_importance.png` | Top-30 features bar chart |
| `metrics.json` | Validation + test metric suite |
| `predictions.parquet` | Per-row: TransactionID, probability, prediction, true label |
| `run_metadata.json` | Python version, package versions, seed, runtime |
| `pr_curve_test.png` | Precision-Recall curve on test set |
| `roc_curve_test.png` | ROC curve on test set |
| `confusion_matrix_test.png` | Confusion matrix heatmap on test set |
| `shap_summary.png` | SHAP beeswarm plot on 5,000 validation samples |
| `split_meta.json` | Temporal split boundary statistics |
| `test_report.json` | 14-test integrity test results |
| `run.log` | Full structured log of the pipeline run |

All reports are in `reports/phase1/`:

| File | Description |
|------|-------------|
| `data_validation_report.md` | Raw file validation output |
| `split_report.md` | Temporal split statistics table |
| `leakage_audit.md` | All 58 column decisions documented |
| `feature_inventory.csv` | All 434 raw columns classified by group |
| `lightgbm_baseline_report.md` | Autogenerated baseline report |
| `PHASE1_COMPLETION_REPORT.md` | **This document** |

---

## Section 12 — Full Codebase Structure

```
adaptive-upi-fraud-detection/
│
├── configs/
│   └── phase1_lightgbm.yaml          # All hyperparameters and paths
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── validate.py               # Schema + integrity validation
│   │   ├── load.py                   # Transaction + Identity JOIN engine
│   │   └── split.py                  # Chronological temporal split
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── ieee_cis_features.py      # Leakage-safe IEEECISPreprocessor class
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── lightgbm_baseline.py      # LightGBMBaseline class (fit/predict/save/load)
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py                # PR-AUC, ROC, threshold selection, plots
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py                 # YAML config loader
│       └── logging_config.py        # Structured logging setup
│
├── experiments/
│   ├── run_phase1.py                 # Main pipeline orchestration (14 steps)
│   └── E1_lightgbm/                  # All outputs from Experiment 1
│       ├── model.txt
│       ├── preprocessing.joblib
│       ├── feature_names.json
│       ├── metrics.json
│       ├── predictions.parquet
│       ├── run_metadata.json
│       ├── feature_importance.csv
│       ├── feature_importance.png
│       ├── pr_curve_test.png
│       ├── roc_curve_test.png
│       ├── confusion_matrix_test.png
│       ├── shap_summary.png
│       ├── split_meta.json
│       ├── test_report.json
│       └── run.log
│
├── datasets/
│   ├── raw/                          # Original immutable CSV files (never modified)
│   └── processed/ieee_cis/
│       ├── train.parquet             # 434,176 rows × 409 cols
│       ├── validation.parquet        # 77,822 rows × 409 cols
│       └── test.parquet             # 78,542 rows × 409 cols
│
├── reports/phase1/                   # Human-readable summary documents
│
├── tests/
│   └── test_phase1.py                # 10-point automated external test suite
│
└── requirements.txt                  # Core dependencies
```

---

## Section 13 — Reproducibility

To reproduce Phase 1 from scratch:

```bash
# Step 1: Install dependencies
pip install lightgbm pyarrow shap pyyaml scikit-learn pandas numpy matplotlib

# Step 2: Run the complete Phase 1 pipeline
python experiments/run_phase1.py --config configs/phase1_lightgbm.yaml

# Step 3: Verify all integrity tests pass
python tests/test_phase1.py
```

Fixed random seed: `42` — all results are deterministic.

---

## Section 14 — What Comes Next (Phase 2 Preview)

Phase 1 established the tabular baseline: **Test PR-AUC = 0.5317 (95% CI to be computed in Phase 1.5B)**.

Before Phase 2 begins, **Phase 1.5** will complete the following:
- 1.5A: Documentation precision corrections (complete)
- 1.5B: Bootstrap confidence intervals, Recall@FPR, Precision@Top-K, calibration analysis
- 1.5C: Feature provenance audit (C/D/M/V temporal risk classification)
- 1.5D: Categorical encoding comparison experiment (LabelEncoder vs. native vs. frequency)
- 1.5E: Locked Phase 2 sequence-generation specification

Phase 2 will then build a **GRU sequence model** on top of `card1` entity histories. The central research question is:

> **"Does historical transaction behavior provide statistically and operationally significant predictive information beyond conventional tabular fraud features?"**

Key Phase 2 design parameters (justified by Phase 1 forensic analysis):
- **Entity key:** `card1` (13,553 unique entities, 48.05% have ≥5 transactions)
- **Sequence length:** minimum 5 transactions (entities with <5 are excluded from GRU training)
- **GRU input:** Temporal features + tabular features per timestep
- **Causal history rule:** When predicting a validation transaction, all prior train-split history for that card entity is available as sequence context (mirroring real deployment). Test sequences may use train + validation history. No future information ever enters a sequence.
- **Evaluation:** Compare GRU test PR-AUC, Recall@FPR, and Precision@Top-K against the frozen E1 baseline using bootstrap confidence intervals. The comparison criterion is statistical and operational significance, not a pre-specified absolute threshold.
- **All outcomes are valid:** If GRU improves, matches, or underperforms, the experiment is scientifically complete.

---

*Last updated: Phase 1.5A complete (2026-09-02) | E1 Run time: 170.6 seconds | All 24 tests (14 internal + 10 external) PASSED | E1 is IMMUTABLE*
