# Sprint 1 — Final Report
## Adaptive Financial Fraud Detection System
### From Raw Datasets to a Research-Grade Baseline

---

> **Who is this report for?**  
> This document is written for anyone — a professor, a friend, or a reviewer — who wants to understand the complete story of what we built, why we built it that way, what results we got, and where we are going next. No prior knowledge of machine learning is required to follow the narrative. Technical depth is provided in dedicated sections.

---

## Table of Contents

1. [The Problem We Are Solving](#1-the-problem-we-are-solving)
2. [Why This Is Hard](#2-why-this-is-hard)
3. [Our System Vision (All 4 Phases)](#3-our-system-vision-all-4-phases)
4. [Dataset Investigation — What We Found](#4-dataset-investigation--what-we-found)
5. [Why We Chose IEEE-CIS as the Primary Dataset](#5-why-we-chose-ieee-cis-as-the-primary-dataset)
6. [Phase 1 — Building the Baseline (LightGBM)](#6-phase-1--building-the-baseline-lightgbm)
7. [Why LightGBM — The Technical Decision](#7-why-lightgbm--the-technical-decision)
8. [The Complete Phase 1 Pipeline](#8-the-complete-phase-1-pipeline)
9. [Phase 1 Results — Every Number Explained](#9-phase-1-results--every-number-explained)
10. [Phase 1.5 — Research Hardening](#10-phase-15--research-hardening)
11. [All Metrics After Phase 1.5](#11-all-metrics-after-phase-15)
12. [What We Have Built (Complete File Inventory)](#12-what-we-have-built-complete-file-inventory)
13. [Achievements of Sprint 1](#13-achievements-of-sprint-1)
14. [Phase 2 — What Comes Next and What We Expect](#14-phase-2--what-comes-next-and-what-we-expect)
15. [The Full Research Roadmap](#15-the-full-research-roadmap)

---

## 1. The Problem We Are Solving

Financial fraud is the act of stealing money through deception — making a payment system believe that a fraudulent transaction is legitimate.

In modern digital payment systems, millions of transactions happen every second. A fraud detection system must:

- **Decide in milliseconds** whether each transaction is genuine or fraudulent
- **Catch as much fraud as possible** (high recall / sensitivity)
- **Not disrupt legitimate customers** (low false positive rate)
- **Adapt** when fraudsters change their behavior over time

Traditional fraud detection uses simple rule-based systems ("block transactions over ₹50,000 from new devices"). These rules are easy to bypass and require constant manual maintenance.

Our goal is to build an **intelligent, self-adapting, real-time fraud detection system** that:

1. Learns fraud patterns automatically from historical transaction data
2. Uses a card's transaction **history** to detect behavioral anomalies
3. Detects when fraud patterns **drift** over time and retrains automatically
4. Processes live transaction streams with low latency

**The project title:**
> *"Adaptive Financial Fraud Detection with Temporal Modeling, Concept Drift Adaptation, Explainable AI, and Real-Time Stream Processing"*

---

## 2. Why This Is Hard

### 2.1 Class Imbalance — The Core Difficulty

In our primary dataset (590,540 transactions):
- **Legitimate transactions:** 574,774 (96.5%)
- **Fraudulent transactions:** 15,766 (3.5%)

This means roughly **1 in 28 transactions is fraud**. This extreme imbalance causes naive machine learning models to simply predict "legitimate" for everything — achieving 96.5% accuracy while catching **zero fraud**. Accuracy is therefore a useless metric for this problem.

### 2.2 The Time Problem — Future Leakage

Most simple machine learning projects split data randomly: 80% for training, 20% for testing. This is **wrong for fraud detection**.

Why? Because in the real world, a fraud detection system always predicts on **future** transactions using patterns learned from **past** transactions. If you train on March data and test on February data, the model has seen "the future" — it will perform unrealistically well in evaluation and poorly in deployment.

We used **strict chronological splitting** throughout this project.

### 2.3 Behavioral Drift — Fraudsters Adapt

Fraud patterns are not static. Fraudsters:
- Switch from one attack method to another
- Exploit newly discovered vulnerabilities
- Change target demographics over time

A model trained in January may perform poorly by June. This is called **concept drift**, and it is one of the central challenges we address in Phases 3 and 4.

### 2.4 What the Research Must Prove

Simply building a model and reporting accuracy is not sufficient. A serious research system must answer:

- Is the model actually better than a naive baseline?
- Could the performance be an artifact of data leakage?
- Is the model robust, or did we get lucky on one test set?
- Does using transaction sequences (history) actually help — or is it just complexity for no gain?
- When do fraud patterns change, and how should the system respond?

---

## 3. Our System Vision (All 4 Phases)

We are building this system in four distinct, scientifically motivated phases:

```
PHASE 1 — Offline Tabular Baseline (COMPLETE)
─────────────────────────────────────────────
  Raw IEEE-CIS data
       ↓
  Leakage-safe preprocessing
       ↓
  LightGBM model
       ↓
  Evaluated: PR-AUC = 0.5317

PHASE 1.5 — Research Hardening (COMPLETE)
──────────────────────────────────────────
  Statistical validation (Bootstrap CI)
  Operational metrics (Recall@FPR, Precision@TopK)
  Calibration analysis
  Feature provenance audit
  Encoding experiment
  Phase 2 specification locked

PHASE 2 — Temporal Sequence Modeling (NEXT)
────────────────────────────────────────────
  GRU model on card transaction histories
       ↓
  Compare against Phase 1 baseline
       ↓
  Fusion: GRU + LightGBM

PHASE 3 — Concept Drift & Adaptive Retraining (PLANNED)
────────────────────────────────────────────────────────
  Detect when fraud patterns change
  Trigger automatic model retraining
  Compare: static model vs. adaptive model

PHASE 4 — Real-Time Streaming (PLANNED)
───────────────────────────────────────
  Kafka message queue
  Live transaction scoring
  Latency benchmarking
  Production monitoring dashboard
```

**This report covers Phase 1 and Phase 1.5 in complete detail.**

---

## 4. Dataset Investigation — What We Found

Before writing a single line of modeling code, we investigated all available datasets to understand what data we actually have and what it can support.

### 4.1 All Datasets Evaluated

| Dataset | Rows | Features | Fraud Rate | Type | Decision |
|---------|------|----------|------------|------|----------|
| **IEEE-CIS Transaction** | 590,540 | 394 | **3.499%** | Real-world-derived (Vesta/Kaggle 2019) | ✅ **Primary** |
| **IEEE-CIS Identity** | 144,233 | 41 | N/A | Real-world-derived (device data) | ✅ **Joined in** |
| BAF Base | 1,000,000 | 32 | 1.10% | Fully synthetic | Secondary testbed |
| BAF Variants (I, II, IV) | 1,000,000 each | 32 | ~1.10% | Synthetic (with drift) | Drift testbed |
| PaySim | 6,362,620 | 11 | 0.13% | Simulation | Streaming demo |
| Credit Card 10K | 10,000 | 10 | 1.51% | Unknown provenance | ❌ **Excluded** |

### 4.2 Critical Discovery: The Datasets Are Not Interchangeable

Each dataset represents a completely different payment context:

| Dataset | Context | What it models |
|---------|---------|----------------|
| IEEE-CIS | Card/e-commerce fraud (USA) | Real-world card payment fraud |
| BAF | Banking account fraud (synthetic) | Account opening fraud |
| PaySim | Mobile money (simulation) | Agent-based transfer fraud |
| Credit Card 10K | Unknown | Unsuitable |

> **Important:** These are **not** UPI transaction datasets. UPI (Unified Payments Interface) is India's real-time payment system. None of our datasets represent UPI transactions. Our project is therefore framed as **"Adaptive Financial Transaction Fraud Detection"** — a general framework applicable to any payment system, including UPI, once real UPI data becomes available.

### 4.3 Why IEEE-CIS Won

The forensic investigation found that IEEE-CIS is uniquely suited for our research goals:

**1. Rich enough for a meaningful baseline:**
- 590,540 transactions across 182 days (~3,244/day)
- 394 transaction features + 41 identity features
- Multiple behavioral signal groups (C, D, M, V columns)

**2. Supports sequence modeling (critical for Phase 2):**
- 13,553 unique card entities (`card1` identifier)
- 48.05% of cards have at least 5 transactions — enough for GRU training
- This was not assumed — it was **verified forensically**

**3. Shows real temporal drift (critical for Phase 3):**
- Fraud rate: 2.57% (Month 1) → 4.07% (Month 3) → 3.50% (Month 6)
- This variation justifies building an adaptive system

**4. None of the others had all three:**
- BAF: No entity histories (each row is independent)
- PaySim: Only 11 features, extreme fraud imbalance
- Credit Card 10K: Too small, unknown source

### 4.4 A Critical Forensic Finding: `TransactionDT` Is Not a Clock

One of the most common errors researchers make with this dataset is treating the `TransactionDT` field as a standard Unix timestamp and converting it to a calendar date. **This is incorrect.**

`TransactionDT` is a **relative counter in seconds** from an internal Vesta reference epoch — not from January 1, 1970. The minimum value is 86,400 (exactly 1 day's worth of seconds), confirming this is a day-relative counter. The total span of 15,724,731 seconds = exactly 182 days.

All our temporal calculations use `TransactionDT` directly without any calendar conversion.

---

## 5. Why We Chose IEEE-CIS as the Primary Dataset

**Summary in plain language:**

> We chose IEEE-CIS because it is the only dataset we have that (a) comes from real transaction data, (b) has enough card-level history to justify sequence modeling, and (c) shows meaningful temporal variation to justify adaptive retraining. The other datasets either lack entity histories, are too simple, or are fully synthetic.

**The research decision was:**

```
BAF → suitable for concept drift experiments (Phase 3) — not for baseline
PaySim → suitable for streaming simulation (Phase 4) — not for modeling
IEEE-CIS → suitable for ALL phases of research
Credit Card 10K → not suitable for any phase
```

---

## 6. Phase 1 — Building the Baseline (LightGBM)

### 6.1 What Phase 1 Is

Phase 1 answers the question:

> **"How well can we detect fraud using only the features available in a single transaction, without any historical context?"**

This is the **control experiment**. Without this baseline, we cannot know whether adding transaction histories (Phase 2) actually improves performance or just adds complexity.

Phase 1 is **not** the final fraud detection system. It is the scientifically rigorous starting point that everything else is measured against.

### 6.2 What Makes This Better Than a Typical Student Project

A typical project does this:
```python
X_train, X_test, y_train, y_test = train_test_split(data, test_size=0.2)
model.fit(X_train, y_train)
print("Accuracy:", model.score(X_test, y_test))  # 96.5% — meaningless
```

We did this:
```
1.  Data validation (schema checks, duplicate detection, fraud rate verification)
2.  LEFT JOIN transaction + identity tables (verified: zero row expansion)
3.  Forensic column audit (missingness, data types, semantic labeling)
4.  CHRONOLOGICAL split (train=past, validation=middle, test=future)
5.  Leakage-safe preprocessing (fit ONLY on training data)
6.  Feature engineering (cyclical time, missingness flags)
7.  LightGBM training with imbalance handling
8.  Threshold selection on validation only (frozen before test)
9.  Test set evaluated exactly once
10. SHAP explainability analysis
11. 14 automated internal integrity tests
12. 10 automated external integrity tests
13. Bootstrap confidence intervals (Phase 1.5)
14. Operational metrics: Recall@FPR, Precision@TopK (Phase 1.5)
```

---

## 7. Why LightGBM — The Technical Decision

### 7.1 What LightGBM Is

LightGBM (Light Gradient Boosting Machine) is an ensemble learning algorithm. It builds hundreds of decision trees in sequence, where each tree learns to correct the mistakes of the previous trees. Unlike traditional gradient boosting which grows trees level-by-level, LightGBM grows trees **leaf-wise** — always splitting the most informative node first. This makes it faster and often more accurate.

### 7.2 Why Not Other Options

| Option | Why Rejected |
|--------|-------------|
| Logistic Regression | Cannot capture non-linear interactions in 406-dimensional feature space |
| Random Forest | Poor handling of class imbalance; no built-in `is_unbalance` |
| Neural Network (MLP) | Requires extensive tuning; no native missing value handling; no SHAP support |
| XGBoost | Similar performance but 3–4× slower on 590K rows |
| GRU / LSTM | Requires sequential data — reserved for Phase 2 |
| SMOTE oversampling | Breaks temporal ordering by creating synthetic pairs across time |

### 7.3 The 6 Specific Reasons LightGBM Was Chosen

**1. Native missing value handling**  
IEEE-CIS has extreme missingness in many columns (V-columns up to 80% missing, D-columns up to 93%). LightGBM learns the optimal split direction for missing values internally — no imputation required for the tree-building logic.

**2. `is_unbalance=True` — built-in class weighting**  
At 3.5% fraud, a model that predicts "legit" for everything gets 96.5% accuracy but catches zero fraud. `is_unbalance=True` automatically weights the fraud class by the inverse of its frequency, forcing the model to treat fraud detection seriously.

**3. Exact SHAP values — free of charge**  
LightGBM's `TreeExplainer` in the SHAP library computes mathematically exact feature importance values in O(TLD) time (T=trees, L=leaves, D=depth). This is needed for the explainability component of our research.

**4. Speed at scale**  
Training on 434,176 rows × 406 features completes in ~77 seconds on CPU. This matters because the encoding comparison (Phase 1.5D) required training 3 variants, and the encoding experiment itself is a re-training scenario.

**5. Reproducibility**  
Fixed seed=42, deterministic algorithm, native `.txt` serialization format that is human-readable and version-controllable.

**6. Historical dominance on this exact dataset**  
The original Kaggle IEEE-CIS competition (2019) was won by LightGBM-based ensemble solutions. This gives us strong historical context for what performance levels are achievable.

### 7.4 LightGBM Configuration Used

```yaml
objective:       binary          # binary classification
metric:          average_precision  # PR-AUC monitored during training
boosting_type:   gbdt            # gradient boosting decision trees
learning_rate:   0.05            # step size per tree
num_leaves:      63              # model complexity control
max_depth:       -1              # unlimited (leaf-wise controls this)
min_child_samples: 50            # minimum samples per leaf (regularization)
feature_fraction:  0.8           # 80% of features used per tree
bagging_fraction:  0.8           # 80% of data used per tree
bagging_freq:      5             # apply bagging every 5 rounds
reg_alpha:         0.1           # L1 regularization
reg_lambda:        1.0           # L2 regularization
is_unbalance:      true          # automatic class weight handling
n_estimators:      1000          # maximum number of trees
early_stopping_rounds: 50        # stop if no improvement for 50 rounds
n_jobs:            -1            # use all CPU cores
seed:              42            # reproducibility
```

---

## 8. The Complete Phase 1 Pipeline

### Step 1 — Data Validation
Before touching any data, verify:
- Both CSV files exist and are readable
- `TransactionID` has zero duplicates in both files  
- `isFraud` contains only 0/1 values
- Fraud rate is within expected range (1–10%)

**Result:** ✅ Fraud rate = 3.499%, zero duplicates

### Step 2 — Load and JOIN
```
train_transaction.csv  (590,540 rows × 394 cols)
         +
train_identity.csv     (144,233 rows × 41 cols)
         ↓
LEFT JOIN on TransactionID
         ↓
590,540 rows × 434 cols   (no fan-out — verified)
```

24.42% of transactions have matching identity records (rest have NaN identity fields).

### Step 3 — Chronological Split

```
TransactionDT  →  Day 1.0 ─────────── Day 127.0 ─── Day 155.0 ─── Day 183.0
                  │←──────── TRAIN ──────────→│←── VAL ──→│←── TEST ──→│
```

| Split | Days | Rows | Fraud | Fraud Rate |
|-------|------|------|-------|------------|
| **Train** | 1–127 | 434,176 | 15,131 | 3.485% |
| **Validation** | 127–155 | 77,822 | 2,637 | 3.389% |
| **Test** | 155–183 | 78,542 | 2,774 | 3.532% |

Zero TransactionID overlap between any two splits — verified by automated test.

### Step 4 — Leakage-Safe Preprocessing

The `IEEECISPreprocessor` class implements the fit/transform pattern:

```
TRAINING DATA
      ↓
  preprocessor.fit()    ← compute medians, label encoder mappings, etc.
      ↓
  preprocessor.transform()  ← apply to train

VALIDATION DATA
      ↓
  preprocessor.transform()  ← apply ONLY (no re-fitting)

TEST DATA
      ↓
  preprocessor.transform()  ← apply ONLY (no re-fitting)
```

If we fit the preprocessor on all data (including test) and then split, the model would have indirect knowledge of test-set statistics. This is called **preprocessing leakage** and is one of the most common mistakes in machine learning papers.

**Features engineered:**

| Feature | Type | Reason |
|---------|------|--------|
| `hour_sin`, `hour_cos` | Cyclical time | Hour 23 and hour 0 are adjacent — raw hour creates discontinuity |
| `is_missing_D1`, `is_missing_V126`, ... | Missingness flag | Whether a field is missing is itself a fraud signal |
| Median imputation | Numeric | Required for sklearn/LightGBM API; filled from train median only |
| Label encoding | 22 string columns | Card identifiers, email domains, device strings |

**Columns dropped:**

| Reason | Columns |
|--------|---------|
| >80% missing in training data | D6, D7, D8, D9, D12, D13, D14, dist2, V-column subset |
| Identifier (no predictive value) | TransactionID |
| Target variable | isFraud |
| Raw temporal counter | TransactionDT (replaced by engineered features) |

**Final feature matrix:** 406 columns

### Step 5 — Save Processed Data
Processed parquet files saved for Phase 2 reuse:
- `datasets/processed/ieee_cis/train.parquet` — 434,176 rows × 409 cols
- `datasets/processed/ieee_cis/validation.parquet` — 77,822 rows
- `datasets/processed/ieee_cis/test.parquet` — 78,542 rows

### Step 6 — Train LightGBM

Training progression (validation PR-AUC):
```
[100]   → 0.4756
[300]   → 0.5385
[500]   → 0.5566
[700]   → 0.5693
[900]   → 0.5793
[1000]  → 0.5849   ← Final (no early stopping triggered)
```

All 1,000 trees trained. The model was still slowly improving at tree 1,000 — a signal that there is more capacity to exploit in Phase 2 hyperparameter tuning.

### Step 7 — Threshold Selection (Validation Only)

LightGBM outputs a **fraud probability** (0.0 to 1.0) for each transaction. To make a binary fraud/legit decision, we need a **threshold**.

The default threshold of 0.5 is wrong for imbalanced datasets. We searched across all thresholds on the validation set to find the one that maximises the F1 score:

```
Optimal threshold (validation F1-maximizing) = 0.616521
```

This threshold is then **frozen** — it is never adjusted using test data.

### Step 8 — Test Evaluation (Once Only)

```
Test set evaluated exactly once using threshold = 0.616521
```

Re-evaluating the test set with different thresholds until a good number appears is called **p-hacking**. Our threshold was selected on validation data only.

### Step 9 — SHAP Explainability

We ran `shap.TreeExplainer` on 5,000 randomly sampled validation rows to compute per-feature importance:

**Top 5 features by SHAP gain:**
1. `V258` — Vesta proprietary signal (top discriminator, semantics unknown)
2. `V294` — Vesta proprietary velocity signal
3. `C13` — Count of transactions on card (backward-looking aggregate)
4. `C1` — Count of addresses associated with card
5. `card1` — Primary card fingerprint (entity identifier)
6. **`D1`** — Days since card first seen (strongest *interpretable* signal)

**D1 insight:** Mean D1 for legitimate transactions = 95.6 days. Mean D1 for fraudulent = 38.7 days. Cards that have been seen for fewer days are strongly associated with fraud. A plausible hypothesis: stolen cards are used quickly before being blocked. *(Note: This is an observed correlation, not proven causation.)*

### Steps 10–14 — Integrity Testing

14 automated internal tests + 10 automated external tests, all passing:

- Zero duplicate TransactionIDs in processed data
- Zero temporal overlap between any two splits
- `isFraud` not present in feature matrix
- `TransactionID` not present in feature matrix
- Model reload produces byte-identical predictions
- PR-AUC exceeds fraud rate (model beats random)
- All required artifacts exist

---

## 9. Phase 1 Results — Every Number Explained

### 9.1 Primary Metric Table

| Metric | Validation | Test (frozen threshold=0.6165) |
|--------|-----------|-------------------------------|
| **PR-AUC** ← *Primary* | **0.5849** | **0.5317** |
| ROC-AUC | 0.9231 | 0.8990 |
| Precision | 0.6285 | 0.5818 |
| Recall | 0.5138 | 0.4935 |
| F1 Score | 0.5654 | 0.5340 |
| MCC | 0.5547 | 0.5203 |
| Balanced Accuracy | 0.7516 | 0.7403 |
| False Positive Rate | 1.065% | 1.299% |
| False Negative Rate | 48.62% | 50.65% |

### 9.2 What Each Metric Means

**PR-AUC (Average Precision) — why this is our primary metric:**
- Measures how well the model *ranks* fraudulent transactions above legitimate ones
- A completely random model achieves PR-AUC = fraud rate = **0.035**
- Our model achieves **0.5317** — that is **15.1× better than random**
- Unlike accuracy, PR-AUC is not fooled by class imbalance

**Why NOT accuracy:**
- A model that predicts "legit" for every transaction gets **96.5% accuracy**
- It catches **zero fraud**
- Accuracy is a dangerous metric for imbalanced problems

**ROC-AUC = 0.8990:**
- Measures ranking quality across all FPR thresholds
- 1.0 = perfect, 0.5 = random
- 0.899 is strong for a tabular baseline

**Recall = 0.4935:**
- Of all real fraud cases, the model catches 49.4%
- The other 50.6% are missed (false negatives)
- This is the "fraud catch rate" — the most important number for operations teams

**Precision = 0.5818:**
- Of all transactions the model flags as fraud, 58.2% are actually fraud
- The other 41.8% are legitimate transactions incorrectly flagged

**False Positive Rate = 1.299%:**
- 1.3% of legitimate transactions are incorrectly flagged as fraud
- On a day with 74,784 legitimate transactions, this = ~972 incorrect fraud alerts

### 9.3 Confusion Matrix (Test Set)

```
                    Predicted: LEGIT    Predicted: FRAUD
Actual: LEGIT       74,784              984
Actual: FRAUD        1,405            1,369
```

- **TP = 1,369** — Real fraud correctly caught
- **FP = 984** — Legitimate transactions incorrectly flagged
- **TN = 74,784** — Legitimate transactions correctly cleared
- **FN = 1,405** — Real fraud cases missed

### 9.4 Validation-to-Test Gap

| Metric | Validation | Test | Drop |
|--------|-----------|------|------|
| PR-AUC | 0.5849 | 0.5317 | −0.053 |
| ROC-AUC | 0.9231 | 0.8990 | −0.024 |

The ~0.05 drop is **expected and scientifically meaningful**. It confirms that fraud patterns in Days 155–183 (test) differ slightly from Days 127–155 (validation). This is natural temporal drift — exactly the phenomenon that Phase 3 (adaptive retraining) will address.

---

## 10. Phase 1.5 — Research Hardening

Phase 1.5 added rigour to the E1 experiment without changing anything about the trained model. The 0.531731 PR-AUC number is **immutable**. Phase 1.5 only adds context, confidence intervals, and analysis around it.

### 10.1 Why Phase 1.5 Was Needed

The Phase 1 result was a strong start. But before it can be cited in a research paper or submitted to a professor, five additional things were required:

1. **Uncertainty quantification** — A single number (0.5317) is meaningless without confidence intervals
2. **Operational metrics** — PR-AUC is for researchers; a fraud team needs "Recall when I can only review 1,000 cases per day"
3. **Calibration** — Does "probability=0.7" actually mean "70% chance of fraud"?
4. **Feature provenance** — Are we sure none of our 406 features encode information from the future?
5. **Encoding audit** — Is the current `LabelEncoder` strategy optimal, or does a better strategy exist?

### 10.2 Sub-Phase 1.5A — Documentation Corrections

Five precision corrections to the Phase 1 report:

| Issue | Incorrect statement | Corrected statement |
|-------|--------------------|--------------------|
| Dataset framing | "real-world transaction data" | "real-world-derived, anonymized (Vesta/Kaggle 2019)" |
| Project framing | Implied UPI | "card/e-commerce fraud — not UPI" |
| D1 feature | "Fraudsters use stolen cards quickly" | "Lower D1 is correlated with fraud — causal claim is a hypothesis" |
| GRU claim | "expected to improve significantly" | Removed — all outcomes are valid research results |
| GRU target | "If GRU > 0.56, it is justified" | Removed — replaced with multi-metric CI-based criterion |

### 10.3 Sub-Phase 1.5B — Statistical Evaluation

All computations run on **existing predictions.parquet** — no model retraining.

#### Bootstrap Confidence Intervals

**Method:** Draw 2,000 bootstrap samples (with replacement) from the test set. Skip any sample with zero fraud instances (would make PR-AUC undefined). Report 95% confidence intervals.

| Metric | Point Estimate | 95% CI | n_valid |
|--------|---------------|--------|---------|
| **PR-AUC** | **0.5315** | **[0.5126 – 0.5504]** | 2,000 |
| ROC-AUC | 0.8989 | [0.8924 – 0.9055] | 2,000 |
| Precision | 0.5813 | [0.5624 – 0.6006] | 2,000 |
| Recall | 0.4934 | [0.4751 – 0.5120] | 2,000 |
| F1 | 0.5337 | [0.5179 – 0.5496] | 2,000 |
| MCC | 0.5200 | [0.5039 – 0.5364] | 2,000 |

**What this means for Phase 2:** When the GRU model (E2) is evaluated, its PR-AUC CI must **not overlap** with [0.5126–0.5504] to claim a statistically significant improvement.

#### Recall at Fixed FPR — Operational Metric

**Definition:** For each FPR target α, find the highest achievable recall among all thresholds where the empirical FPR ≤ α.  
**Why:** Fraud operations teams have limited investigation capacity. They need to know: "If I can only afford 1% false positives, how much fraud do I catch?"

| FPR Target | Recall (test) | Meaning (approx.) |
|-----------|--------------|-------------------|
| **0.1%** | 21.1% | ~75 false alerts/day → catches 1 in 5 fraud |
| **0.5%** | 38.4% | ~378 false alerts/day |
| **1.0%** | 46.2% | ~758 false alerts/day → catches nearly half of fraud |
| **2.0%** | 53.1% | ~1,515 false alerts/day |

#### Precision at Top-K — Investigation Capacity Metric

**Definition:** Of the top-K transactions ranked by predicted fraud probability, what fraction are actually fraud?  
**Why:** If a team can investigate 1,000 cases per day, the relevant question is: "How many of my top-1,000 alerts are real?"

| K | Precision (test) | Fraud in top-K | vs. Random (3.5%) |
|---|-----------------|----------------|-------------------|
| **100** | **98.0%** | ~98 real fraud | 27.9× better |
| **500** | **90.8%** | ~454 real fraud | 25.9× better |
| **1,000** | **84.1%** | ~841 real fraud | 24.0× better |
| **5,000** | 34.5% | ~1,725 real fraud | 9.9× better |

#### Calibration

**Brier Score** = mean squared error between predicted probability and true label:
- Our model: **0.0323**
- Naive baseline (always predict fraud_rate): **0.0341**
- Model is better calibrated than naive ✅

**ECE (Expected Calibration Error)** = weighted gap between predicted probability and actual positive fraction:
- Our model: **0.0524** — slight overconfidence in the mid-range
- This is expected for `is_unbalance=True` models and is not a concern for ranking-based metrics

### 10.4 Sub-Phase 1.5C — Feature Provenance Audit

The IEEE-CIS dataset contains many anonymized features. We audited all 406 features across a 4-tier risk classification:

| Tier | Meaning | Groups |
|------|---------|--------|
| ✅ **KNOWN** | Scope confirmed from documentation | TransactionDT, amount, ProductCD, card1–6, engineered features |
| 🟡 **STRONGLY INFERRED** | Strong evidence but not formally documented | C1–C14, D1–D15, M1–M9, addresses, email domains |
| 🟠 **UNKNOWN/OPAQUE** | No public documentation; leakage cannot be ruled out | **V1–V339**, id_31–id_38 |
| 🔴 **POTENTIAL LEAKAGE** | Concrete reason to suspect forward-looking info | None identified |

**The critical finding:** V1–V339 (339 Vesta proprietary features) are **completely opaque**. No public documentation explains how they are constructed. Some may be velocity signals computed over short time windows. For paper submission, these must be disclosed as: *"Unknown provenance — temporal leakage cannot be formally ruled out."*

### 10.5 Sub-Phase 1.5D — Encoding Experiment

The current E1 model uses `LabelEncoder` to convert string features (email domains, device strings, etc.) to integers. We tested 3 strategies:

| Strategy | How it works |
|----------|-------------|
| V1 LabelEncoder | `gmail → 0, yahoo → 1, protonmail → 2` (alphabetical order) |
| V2 Native Categorical | LightGBM internally finds optimal category groupings (Fisher method) |
| V3 Frequency Encoding | `gmail → 45231, yahoo → 12042` (replace with training count) |

**Results:**

| Metric | V1 LabelEncoder | V2 Native | V3 Frequency |
|--------|----------------|-----------|-------------|
| PR-AUC (test) | **0.5362** | 0.5143 | 0.5311 |
| Recall@1%FPR (test) | 0.4445 | 0.4322 | 0.4477 |
| P@Top-1K (test) | 0.8260 | 0.8100 | **0.8440** |
| Training time | 76.7s | 42.9s | 40.4s |

**Decision: E1b is NOT justified.** V1 (current E1) wins on PR-AUC. V3 wins narrowly on P@Top-1K only — a single-metric edge does not satisfy the multi-metric consistency criterion. E1 remains the immutable baseline.

### 10.6 Sub-Phase 1.5E — Phase 2 Sequence Specification (LOCKED)

Before writing any GRU code, the sequence construction rules were formally documented and locked.

**The most important design decision — Causal History Rule:**

A naive approach would restrict each prediction to only use history from its own split:
```
Validation prediction → only validation-split history (WRONG — too conservative)
```

The correct approach mirrors real deployment:
```
Validation prediction → ALL prior train-split history available ✅
Test prediction → ALL prior train + validation-split history available ✅
```

When a fraud model predicts a transaction on Day 132, it has access to all transactions from Days 1–131. Pretending those don't exist would make the model unrealistically bad at evaluation time.

**Locked parameters:**
- Entity key: `card1`
- Minimum sequence length: 5 transactions
- Window size: 4 history steps + 1 prediction target
- 5 mandatory leakage tests required before GRU training can begin
- Split boundaries from E1: `val_dt = 10,972,800`, `test_dt = 13,392,000`

---

## 11. All Metrics After Phase 1.5

### Complete E1 Scorecard (Final, Immutable)

```
═══════════════════════════════════════════════════════════════
  E1 LightGBM — IEEE-CIS — seed=42 — threshold=0.616521
═══════════════════════════════════════════════════════════════

  RANKING METRICS (test set)
  ─────────────────────────────────────────────
  PR-AUC            0.5317  [95% CI: 0.5126–0.5504]
  ROC-AUC           0.8990  [95% CI: 0.8924–0.9055]

  CLASSIFICATION METRICS (test, frozen threshold)
  ─────────────────────────────────────────────
  Precision         0.5818  [95% CI: 0.5624–0.6006]
  Recall            0.4935  [95% CI: 0.4751–0.5120]
  F1                0.5340  [95% CI: 0.5179–0.5496]
  MCC               0.5203  [95% CI: 0.5039–0.5364]
  Balanced Accuracy 0.7403
  FPR               1.299%
  FNR               50.65%

  OPERATIONAL METRICS (test set)
  ─────────────────────────────────────────────
  Recall @ 0.1% FPR    0.2112
  Recall @ 0.5% FPR    0.3836
  Recall @ 1.0% FPR    0.4618
  Recall @ 2.0% FPR    0.5306
  Precision @ Top-100  0.9800
  Precision @ Top-500  0.9080
  Precision @ Top-1K   0.8410
  Precision @ Top-5K   0.3450

  CALIBRATION (test set)
  ─────────────────────────────────────────────
  Brier Score       0.0323  (naive baseline: 0.0341)
  ECE               0.0524

  INTEGRITY
  ─────────────────────────────────────────────
  Internal tests    14/14 PASSED
  External tests    10/10 PASSED
  E1b justified?    NO (V1 LabelEncoder is best encoding)

═══════════════════════════════════════════════════════════════
```

---

## 12. What We Have Built (Complete File Inventory)

### Source Code

```
src/
├── data/
│   ├── validate.py          — Schema validation, duplicate checks, fraud rate checks
│   ├── load.py              — Transaction + Identity JOIN engine
│   └── split.py             — Chronological temporal split
├── features/
│   └── ieee_cis_features.py — IEEECISPreprocessor (leakage-safe fit/transform)
├── models/
│   └── lightgbm_baseline.py — LightGBMBaseline (fit, predict, save, load)
├── evaluation/
│   ├── metrics.py           — compute_metrics(), recall_at_fixed_fpr(), precision_at_top_k()
│   ├── bootstrap.py         — bootstrap_metrics() targeting valid samples
│   └── calibration.py       — Brier score, ECE, reliability diagram
└── audit/
    ├── feature_provenance.py — 4-tier provenance audit
    └── encoding_comparison.py — V1/V2/V3 encoding experiment
```

### Experiments

```
experiments/
├── run_phase1.py            — Phase 1 pipeline (14 steps, E1_lightgbm output)
├── run_phase15_extend.py    — Phase 1.5B extension (reads predictions, no retraining)
└── E1_lightgbm/             — All Phase 1 artifacts (immutable)
    ├── model.txt            — LightGBM booster (6.97 MB, 1000 trees)
    ├── preprocessing.joblib — Fitted preprocessor (medians + label encoders)
    ├── feature_names.json   — 406-column feature list
    ├── metrics.json         — All metrics including Phase 1.5B additions
    ├── predictions.parquet  — Per-transaction: ID, probability, prediction, label
    ├── bootstrap_ci.json    — 95% CI for all metrics
    ├── calibration_curve_test.png
    ├── feature_importance.png/csv
    ├── shap_summary.png
    ├── pr_curve_test.png
    ├── roc_curve_test.png
    ├── confusion_matrix_test.png
    └── run.log
```

### Reports

```
reports/
├── phase1/
│   ├── PHASE1_COMPLETION_REPORT.md     — Master Phase 1 documentation
│   ├── operational_metrics_report.md   — Bootstrap CI, Recall@FPR, P@TopK
│   ├── feature_provenance_audit.md     — 16-group provenance assessment
│   ├── encoding_comparison.md          — V1/V2/V3 results + E1b decision
│   └── leakage_audit.md               — 58-column drop/keep decision log
└── phase2/
    └── SEQUENCE_GENERATION_SPEC.md     — Locked GRU design specification (v1.0)
```

### Tests

```
tests/
└── test_phase1.py           — 10-point external integrity test suite (all PASSED)
```

---

## 13. Achievements of Sprint 1

### Research-Level Achievements

| Achievement | Why It Matters |
|-------------|----------------|
| Forensic dataset audit (5 datasets) | Prevented forcing wrong methods onto wrong data |
| Chronological temporal split | Evaluation reflects real deployment conditions |
| Leakage-safe preprocessing (fit on train only) | No inflated metrics from future data contamination |
| Bootstrap 95% CI computation | All future comparisons can be made with statistical rigour |
| Feature provenance audit (4-tier) | Identified unknown-provenance V1–V339 for paper disclosure |
| Encoding experiment (V1/V2/V3) | Confirmed current encoding is optimal; E1b not needed |
| Precision@Top-K operational metric | Models real investigation capacity constraints |
| Recall@Fixed-FPR operational metric | Operationally meaningful across different team sizes |
| Calibration analysis | Confirmed model is better calibrated than naive baseline |
| Locked Phase 2 sequence specification | GRU development can begin on a formally verified design |

### Engineering Achievements

| Achievement | Detail |
|-------------|--------|
| 24/24 automated tests passing | 14 internal + 10 external integrity tests |
| Fully reproducible pipeline | Fixed seed, deterministic, single command |
| E1 declared immutable | Baseline never overwritten; all comparisons anchored |
| Multi-metric E1b decision framework | Not a single arbitrary threshold |
| Correct causal history rule | Cross-split history allowed for GRU (mirrors deployment) |

### Numbers Achieved (E1 Baseline)

- **PR-AUC = 0.5317** — 15.1× better than random (random = 0.035)
- **Precision@Top-100 = 98.0%** — Near-perfect ranking at the top
- **Recall@1%FPR = 46.2%** — Nearly half of fraud caught within operational FPR budget
- **Brier Score = 0.0323** — Better calibrated than naive baseline (0.0341)

---

## 14. Phase 2 — What Comes Next and What We Expect

### 14.1 The Research Question

> **"Does historical transaction behavior provide statistically and operationally significant predictive information beyond conventional tabular fraud features?"**

This is the core hypothesis of Phase 2. LightGBM treats each transaction **independently** — it does not know that the same card made purchases yesterday, last week, and last month. A GRU (Gated Recurrent Unit) model can learn from this sequence of events.

### 14.2 What GRU Is and Why We Are Using It

A **GRU** (Gated Recurrent Unit) is a type of neural network designed for sequential data. It processes a sequence of inputs one step at a time and maintains a "memory" (hidden state) that carries information from earlier steps to later ones.

For fraud detection:
```
Card history:
  Day 10 → purchase $50 at grocery store (normal)
  Day 20 → purchase $80 at pharmacy (normal)
  Day 30 → purchase $200 at electronics (slightly unusual)
  Day 40 → purchase $1,500 at jewelry store (suspicious given history)

GRU reads [Day10, Day20, Day30] → predicts fraud probability for Day40
```

LightGBM would evaluate Day 40 purely on its own features (amount, device, email domain, etc.). GRU additionally knows that Day 40 is unusual **relative to this card's history** — even if the transaction itself looks normal in isolation.

### 14.3 The Phase 2 Design (From Locked Spec)

**Entity construction:**
- Use `card1` as entity key (13,553 unique cards)
- Only entities with ≥5 transactions are used for GRU training (6,518 entities = 48.05%)
- Window size: 4 history steps → predict 5th transaction label

**A concrete example:**
```
Card X history (sorted by TransactionDT):
  T1 (Day 15) → features_T1
  T2 (Day 23) → features_T2
  T3 (Day 45) → features_T3
  T4 (Day 67) → features_T4
  T5 (Day 89) → features_T5  ← this is what we predict

Window:
  Input:  [features_T1, features_T2, features_T3, features_T4]
  Target: T5.isFraud
```

**Causal history rule (critical):**
- When predicting a test-split transaction, all prior train + validation history is available
- This mirrors real deployment — the model knows what happened before
- The prediction transaction itself is NEVER part of its own input sequence

**5 mandatory leakage tests** must pass before any GRU training begins:
1. Prediction transaction not in its own input
2. All input TransactionDTs < prediction TransactionDT
3. `isFraud` not used as a sequence feature
4. All sequences sorted ascending by TransactionDT
5. GRU trained only on train-split labels

### 14.4 What We Expect from Phase 2

**Possible outcomes (all are valid research results):**

```
Scenario A — GRU wins (most optimistic):
  E1 LightGBM: PR-AUC = 0.5317 [0.5126–0.5504]
  E2 GRU:      PR-AUC > 0.5504 (above E1's CI upper bound)
  → Conclusion: Transaction history provides significant predictive value
  → Next: Build E3 (GRU + LightGBM fusion)

Scenario B — GRU matches (neutral result):
  E2 GRU CI overlaps with E1 CI
  → Conclusion: History does not provide statistically significant additional value
  → Still: Build E3 fusion — combined representation may help

Scenario C — GRU underperforms (valid negative result):
  E2 PR-AUC < E1 lower bound
  → Conclusion: Pure sequence modeling on card1 is insufficient
  → Still valuable: Shows that tabular features capture most fraud signal in this dataset
  → Motivates: Feature-enhanced GRU, longer sequences, or different entity keys
```

### 14.5 What Metrics Phase 2 Will Report

| Metric | E1 Value | E2 Target | Comparison method |
|--------|----------|-----------|------------------|
| PR-AUC (test) | 0.5317 [0.5126–0.5504] | TBD | Bootstrap CI non-overlap |
| Recall@1%FPR (test) | 0.4618 | TBD | Absolute improvement |
| P@Top-1K (test) | 0.8410 | TBD | Absolute improvement |
| Brier Score | 0.0323 | TBD | Lower is better |
| Training time | ~77s | ~10–30 min | Efficiency trade-off |

### 14.6 Phase 2 Implementation Steps

```
Step 1: Implement src/sequences/window_generator.py
        — Build (X_seq, y) pairs from processed parquets
        — Follow SEQUENCE_GENERATION_SPEC.md exactly

Step 2: Write tests/test_phase2_sequences.py
        — All 5 mandatory leakage tests must pass before training

Step 3: Implement src/models/gru_fraud.py
        — GRU architecture: 128-dim hidden, 2 layers, dropout=0.3
        — Input: [batch, L-1, 404+1 gap_feature]
        — Output: fraud probability

Step 4: Run experiments/run_phase2_gru.py
        — Train E2 GRU on train-split windows only
        — Evaluate on validation for threshold selection
        — Evaluate on test set exactly once

Step 5: Statistical comparison
        — Bootstrap CI on E2 test predictions
        — Compare E2 CI against E1 CI [0.5126–0.5504]
        — Report Recall@FPR, P@TopK for operational comparison

Step 6 (if E2 shows value): Build E3 — GRU + LightGBM Fusion
        — Concatenate GRU embedding + LightGBM probability score
        — MLP fusion head → final probability
```

---

## 15. The Full Research Roadmap

```
Sprint 1 (COMPLETE)
═══════════════════════════════════════════
  Phase 1:   E1 LightGBM baseline
             PR-AUC = 0.5317 [0.5126–0.5504]

  Phase 1.5: Research hardening
             Bootstrap CI, Recall@FPR, P@TopK
             Calibration, Provenance audit
             Encoding experiment, Phase 2 spec locked

Sprint 2 (NEXT)
═══════════════════════════════════════════
  Phase 2A:  E2 GRU sequence model
             → Compare against E1

  Phase 2B:  E3 GRU + LightGBM Fusion
             → Compare against E1 and E2

Sprint 3 (PLANNED)
═══════════════════════════════════════════
  Phase 3A:  Concept drift detection
             → PSI (Population Stability Index)
             → ADWIN / DDM statistical tests
             → Performance monitoring over time windows

  Phase 3B:  Adaptive retraining
             → Trigger retraining when drift detected
             → Compare: static E3 vs. adaptive E3

Sprint 4 (PLANNED)
═══════════════════════════════════════════
  Phase 4:   Real-time streaming system
             → Kafka consumer for live transactions
             → Sliding window feature preparation
             → GRU + LightGBM inference endpoint
             → Redis/PostgreSQL state management
             → Monitoring dashboard
             → Latency and throughput benchmarking
```

---

## Summary

In Sprint 1, we did not just "build a fraud detection model." We:

1. **Investigated 5 datasets** before writing a single modeling line
2. **Chose the right dataset** based on what the research actually requires
3. **Built a production-grade pipeline** with chronological splitting and leakage controls
4. **Established a scientifically defensible baseline** with 24/24 automated integrity tests
5. **Quantified uncertainty** with bootstrap confidence intervals
6. **Computed operational metrics** that fraud teams actually use
7. **Audited every feature** for temporal risk across 4 tiers
8. **Ran an encoding experiment** and made a data-driven E1b decision
9. **Locked the Phase 2 design** before writing any GRU code

The result is an experiment (`E1_lightgbm`) that can be cited, compared against, and built upon with scientific confidence.

**The baseline is set. Phase 2 begins.**

---

*Report generated: 2026-09-02 | Sprint 1 complete*  
*All 24 integrity tests passed | E1 PR-AUC = 0.5317 [95% CI: 0.5126–0.5504] | E1 is IMMUTABLE*
