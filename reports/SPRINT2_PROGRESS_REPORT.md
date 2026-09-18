# Sprint 2 — Research Progress Report
## Adaptive Financial Fraud Detection
### "What We Built After Phase 1"

**Sprint Period:** September 2026  
**Report Date:** 2026-09-18  
**Prepared by:** Research Team  
**Audience:** Anyone — technical or non-technical

---

## 🚀 Quick Recap: What was Sprint 1?

Before we dive into Sprint 2, here's a one-paragraph reminder of Sprint 1:

> **Sprint 1 (Phase 1)** built our first fraud detection model — a LightGBM tabular model called **E1** — using the IEEE-CIS dataset. We carefully prevented data leakage by splitting transactions in chronological order (past = training, future = testing), processed 406 features, and produced a strong baseline. E1 achieved a **PR-AUC of 0.5317** on the test set (compared to a random baseline of ~0.035 — about 15× better). We then hardened the results in **Phase 1.5** with statistical confidence intervals, operational metrics, and calibration checks. Phase 1 + 1.5 are **fully complete and frozen.**

---

## 📋 Sprint 2 — Everything We Did

Sprint 2 covers **Phase 2 (Temporal GRU Modeling)** and a **Dataset Suitability Audit**.

Here's the complete picture of what we implemented and why.

---

## Part 1 — Designing the Sequence Builder

### What is a sequence?

In Phase 1, we only looked at **one transaction at a time**.

In Phase 2, we asked: *"Can the model learn from the history of past transactions for the same cardholder?"*

Imagine a cardholder's transaction history:

```
Card A:
  T1 → Monday 9am   — $45 grocery
  T2 → Monday 2pm   — $120 online purchase
  T3 → Tuesday 8am  — $35 coffee
  T4 → Tuesday 11am — $89 gas station
  T5 → Tuesday 11:02am — $5,200 electronics (Miami) ← Is this fraud?
```

**Phase 2 feeds T1, T2, T3, T4 (the history) into a GRU neural network to help predict T5 (the target).**

Phase 1 only saw T5 alone.

### The locked rules we designed

We wrote a complete **Sequence Specification (v1.1)** before touching any data. Every rule was locked before implementation:

| Rule | Decision | Why |
|------|----------|-----|
| How to group transactions? | By `card1` (anonymized card ID) | Groups transactions by the same cardholder |
| Minimum transactions per cardholder? | 5 | Need at least 4 history + 1 target |
| History length? | 4 transactions | L=4 gives meaningful context without excluding too many cardholders |
| Ordering? | Strictly by time (TransactionDT) | Cannot use future transactions as history |
| What if two transactions have the same timestamp? | Exclude that window | Cannot guess the ordering — 585 windows excluded |
| Is the cardholder ID (card1) in the model input? | ❌ No | Prevents the model from memorizing card identities instead of patterns |
| Is the fraud label in the history? | ❌ No | That would be cheating — the label is the answer |
| Special gap feature? | ✅ Yes — log(time since previous transaction) | Rapid transactions can signal fraud |

### Leakage tests — all passed

We wrote **33 automated tests** to verify that no future information could leak into training. All 33 passed. ✅

---

## Part 2 — E2a: The Unscaled GRU (Control Experiment)

### What is a GRU?

A GRU (Gated Recurrent Unit) is a type of neural network that reads a sequence of inputs step-by-step and "remembers" patterns across steps — like reading a sentence word by word and understanding meaning from the order.

Our GRU:
```
Input:  [4 history steps × 406 features]
         ↓
GRU:    406 → 64 hidden units
         ↓
Output: 1 probability score (fraud / not fraud)
Total:  90,689 trainable parameters
```

### What happened with E2a?

We trained this GRU **without scaling the features** (i.e., raw feature values from Phase 1 were fed directly in).

**Result:** The GRU achieved a validation PR-AUC of **0.0345**.

This is almost exactly equal to the fraud rate in the training data (~3.4%). A model that predicts "fraud probability = 3.4%" for every single transaction — ignoring all input — would score ~0.035.

**In plain terms: the unscaled GRU learned essentially nothing.**

### Why keep E2a if it failed?

E2a is a **control experiment**. It tells us the model architecture works but needs scaled inputs. Without E2a, we couldn't compare to E2b in a principled way.

---

## Part 3 — E2b: The Scaled GRU (Primary Experiment)

### What changed?

One thing: we added a **StandardScaler** before feeding features into the GRU.

StandardScaler transforms each feature to have:
- Mean = 0
- Standard deviation = 1

This is important because neural networks trained with gradient descent are sensitive to feature scale. When some features have values in the thousands and others are 0 or 1, the network gets confused. Scaling fixes this.

**Critical rule:** The scaler was fitted **only on training data**. Validation and test data were transformed using the training statistics — never refitted. This prevents any information from the future leaking into the scaling step.

### Training results

| Epoch | Training Loss | Validation Loss | Validation PR-AUC |
|-------|-------------|----------------|------------------|
| 1 | higher | higher | — |
| **2** | **1.1071** | **1.1311** | **0.1377** ← Best |
| 3 | improving | rising | lower |
| ... | | | |
| 7 | — | — | (early stop triggered) |

**Best checkpoint = Epoch 2** (saved and frozen, SHA-256: `31b17bb46d2c9257`)

The model started overfitting after epoch 2 (validation loss rising while training loss was still improving). Early stopping caught this automatically.

### What improved?

| | E2a (unscaled) | E2b (scaled) |
|--|---------------|-------------|
| Validation PR-AUC | 0.0345 | **0.1377** |
| Performance vs. random | ≈ same as random | ~4× above random |

Scaling was associated with a substantial improvement. The architecture itself was identical.

---

## Part 4 — STEP 9: The Final Test Evaluation

### The golden rule

A proper research evaluation uses the test set **exactly once**. If you look at test results and then keep tweaking your model, you're essentially using the test set for training. We waited until everything was locked before opening the test set.

### What we did

1. Loaded the **frozen E1 model** (untouched since Phase 1)
2. Loaded the **frozen E2b checkpoint** (epoch 2, untouched)
3. Loaded the **frozen E2b scaler** (trained statistics only)
4. Found all **common test transactions**: 76,520 transactions that both E1 and E2b could evaluate
5. Applied a **decision threshold** selected from validation data (not test data)
6. Evaluated both models on the same 76,520 transactions
7. Ran **2,000 paired bootstrap resamples** to measure statistical reliability

### The common test population

```
E1 test set:       78,542 transactions
E2b test set:      76,520 transactions  (some cards didn't have enough history)
Common population: 76,520 transactions  ← Used for fair comparison
Coverage:          97.43%
```

The 2,022 transactions that E2b missed simply didn't have 4 prior transactions from the same cardholder — a documented, expected limitation.

---

## Part 5 — STEP 9 Results

### Full test metrics — same 76,520 transactions

| Metric | E1 (LightGBM) | E2b (GRU) | Simple explanation |
|--------|-------------|---------|-------------------|
| **PR-AUC** | **0.5267** | **0.1683** | Main metric. Higher = better fraud detection under imbalance. |
| ROC-AUC | 0.8981 | 0.7422 | How well it ranks fraud above legit. |
| Precision | 0.5790 | 0.1913 | Of all flagged fraud alerts, % that are real fraud |
| Recall | 0.4900 | 0.3337 | Of all actual frauds, % that were caught |
| F1 | 0.5308 | 0.2432 | Balance of precision and recall |
| False Alert Rate (FPR) | **0.0130** | 0.0516 | % of legit transactions wrongly flagged |
| Missed Fraud (FNR) | 0.5100 | 0.6663 | % of fraud transactions missed |
| Brier Score | **0.0324** | 0.2377 | Probability accuracy (lower = better) |
| ECE | **0.0523** | 0.4247 | Calibration quality (lower = better) |

### Confusion matrices

**E1 — What happened with each transaction:**

| | Predicted Legit | Predicted Fraud |
|--|--|--|
| **Actually Legit** (73,820) | ✅ 72,858 correct | ❌ 962 false alarms |
| **Actually Fraud** (2,700) | ❌ 1,377 missed | ✅ 1,323 caught |

**E2b — What happened:**

| | Predicted Legit | Predicted Fraud |
|--|--|--|
| **Actually Legit** (73,820) | ✅ 70,011 correct | ❌ 3,809 false alarms |
| **Actually Fraud** (2,700) | ❌ 1,799 missed | ✅ 901 caught |

### Operational metrics

*"If I can only investigate X transactions per day, how many frauds can I catch?"*

| Alert budget | E1 catches | E2b catches |
|------------|-----------|-----------|
| Top 100 alerts | **98 real frauds** | 48 real frauds |
| Top 500 alerts | **452 real frauds** | 214 real frauds |
| Top 1,000 alerts | **831 real frauds** | 365 real frauds |

### Statistical test — Paired Bootstrap

We ran 2,000 paired bootstrap resamples (randomly resampling the same 76,520 transactions 2,000 times) to measure whether the difference is real or just due to random chance.

```
Difference (E2b − E1) in PR-AUC = −0.3584
95% Confidence Interval          = [−0.3756, −0.3395]
Does the interval include zero?   NO

Meaning: Across all 2,000 resamples, E1 consistently
         outperformed E2b. This is not sampling noise.
```

---

## Part 6 — What the Results Mean

### Plain language conclusion

> Under the tested configuration — a 4-step GRU with 406 standardized features — the standalone temporal model (E2b) did not outperform the tabular baseline (E1) on the common test population.

### What this does NOT mean

❌ NOT: "GRUs don't work for fraud detection"  
❌ NOT: "History is useless"  
❌ NOT: "We should give up on neural networks"

### What this DOES mean

✅ "Under these specific settings, the GRU alone doesn't beat the tabular model"  
✅ "The tabular model may already capture most of what the 4-step history can offer"  
✅ "The next question is: can history ADD value ON TOP OF the tabular model?" → That's E3 Hybrid

Think of it this way:
- E1 (Phase 1) = reading one page of a book and summarizing it
- E2 (Phase 2) = reading the last 4 pages only
- E3 (Phase 3) = reading the last 4 pages AND the current page together

E3 hasn't been run yet — but it's the natural next experiment.

---

## Part 7 — STEP 10: Final Phase 2 Freeze

After the test results were obtained, we did a **formal audit** to officially close Phase 2:

### What was verified

| Check | Result |
|-------|--------|
| E1 predictions file unchanged (SHA-256) | ✅ `6a73279d81491437` MATCH |
| E2b model checkpoint unchanged (SHA-256) | ✅ `31b17bb46d2c9257` MATCH |
| E2b scaler unchanged (SHA-256) | ✅ `fe294184277da80f` MATCH |
| E1 metrics file unchanged (SHA-256) | ✅ `6f3ab9c86f9b9f47` MATCH |
| All 20 test metrics verified | ✅ PASS |
| Paired bootstrap verified | ✅ PASS |
| 80/80 automated tests passing | ✅ PASS |
| Known warnings documented | ✅ 2 minor warnings (no effect on results) |

### Phase 2 is officially frozen 🔒

Nothing in Phase 1 or Phase 2 can be changed. All artifacts, metrics, and source code are locked.

---

## Part 8 — Dataset Suitability Audit (Parallel Track)

While Phase 2 was running, we also audited all datasets available in the repository.

### What datasets do we have?

| Dataset | Type | Rows | Fraud% | What it's for |
|---------|------|------|--------|--------------|
| **IEEE-CIS** | Real-world (anonymized) | 590,540 | 3.5% | Primary benchmark — E1, E2, E3 |
| **ULB Credit Card** | Real-world (PCA-anonymized) | 284,807 | **0.17%** | External validation — extreme imbalance |
| **BAF** | Realistic synthetic | 6,000,000 | 1.1% | Concept drift experiments |
| **PaySim** | Simulated mobile money | 6,362,620 | 0.13% | Streaming demonstration |
| **10K toy** | Unknown toy dataset | 10,000 | 1.5% | ❌ Not used — too small, not a benchmark |

### Key finding: ULB is NOT the same as the "Credit Card Fraud" folder

We have a folder called `Credit card Fraud detection/` with a 10K file. **This is completely different from ULB.** The 10K file is a small toy dataset with no published provenance. ULB (284,807 rows, PCA features) is a well-known benchmark. They happen to be about credit cards but they are entirely different datasets.

### Key finding: IEEE-CIS V-features ≠ ULB V-features

```
IEEE-CIS: V1, V2, ... V339
  → Vesta-proprietary ENGINEERED features (not PCA)
  → 339 features
  
ULB: V1, V2, ... V28  
  → PCA principal components of hidden original features
  → 28 features
  
SAME NAMES. COMPLETELY DIFFERENT MEANING.
Must NEVER be combined or shared.
```

### Role assignments

```
IEEE-CIS → Phase 1, 2, 3 (primary)
BAF      → Phase 5 Concept Drift (main role)
PaySim   → Phase 8 Kafka Streaming
ULB      → Phase 4 External Validation
10K toy  → Not used in research
```

---

## Part 9 — Complete Research Status

### What is done

```
Phase 1    E1 LightGBM (tabular baseline)         ✅ COMPLETE
Phase 1.5  E1 hardening (bootstrap, calibration)  ✅ COMPLETE
Phase 2    E2a GRU unscaled (control)              ✅ COMPLETE
Phase 2    E2b GRU scaled (primary)                ✅ COMPLETE
Phase 2    STEP 9 — Final test evaluation          ✅ COMPLETE
Phase 2    STEP 10 — Audit + Freeze               🔒 FROZEN
Dataset    Suitability Audit (all 5 datasets)     ✅ COMPLETE
```

### What is next

```
Phase 3    E3 Hybrid (GRU + tabular combined)     ⏳ READY TO START
Phase 4    Cross-dataset (ULB-E1, BAF-E1)         ⏳ PLANNED
Phase 5    Concept Drift (BAF Variants)            ⏳ PLANNED
Phase 6    Adaptive Retraining                     ⏳ PLANNED
Phase 7    Explainability (SHAP)                   ⏳ PLANNED
Phase 8    Kafka + Spark Streaming                 ⏳ PLANNED
Phase 9    FastAPI Serving                         ⏳ PLANNED
Phase 10   End-to-end Integration                  ⏳ PLANNED
```

### The big picture

```
[Sprint 1]
   Phase 1 + 1.5
   E1 LightGBM
   PR-AUC = 0.5317
        │
        ▼
[Sprint 2 — THIS SPRINT]
   Phase 2
   E2a (unscaled) → near-random (PR-AUC = 0.0345)
   E2b (scaled)   → learning   (Val PR-AUC = 0.1377)
   
   STEP 9 Final Test:
   E1: 0.5267   E2b: 0.1683
   Δ = −0.3584  CI: [−0.3756, −0.3395]
   
   + Dataset Audit (5 datasets profiled)
   Phase 2 🔒 FROZEN
        │
        ▼
[Sprint 3 — NEXT]
   Phase 3: E3 Hybrid
   "Does history ADD to the tabular model?"
```

---

## Part 10 — Key Numbers to Remember

| What | Number |
|------|--------|
| E1 PR-AUC (test, full population) | **0.5317** |
| E1 PR-AUC (common 76,520) | **0.5267** |
| E2b PR-AUC (test, common 76,520) | **0.1683** |
| Difference (E2b − E1) | **−0.3584** |
| 95% CI for difference | **[−0.3756, −0.3395]** |
| E1 Precision@Top-100 | **98%** (of top 100 alerts, 98 are real fraud) |
| E2b Precision@Top-100 | 48% |
| Common test transactions | **76,520** |
| Sequences trained on | **398,312** |
| History steps per sequence | **4** |
| Features per step | **406** |
| GRU parameters | **90,689** |
| Tests passing | **80 / 80** |
| Datasets audited | **5** |
| Phase 2 freeze status | 🔒 **FROZEN** |

---

## Part 11 — What We Learned in Sprint 2

1. **Neural networks need properly scaled inputs.** E2a showed this directly — without scaling, the GRU learned nothing. E2b showed that scaling enabled learning.

2. **A strong tabular model is hard to beat with a simple GRU.** E1's 406 features include engineered aggregations that already contain some temporal information. A 4-step GRU processing the same features may not find much to add.

3. **The experiment that matters is E3.** E2 tested "GRU alone vs. E1 alone." E3 tests "GRU + E1 together vs. E1 alone." This is the scientifically meaningful next question.

4. **BAF is the right dataset for concept drift.** The dataset audit confirmed that BAF (NeurIPS 2022, 6 million rows, 5 drift variants) is purpose-built for concept drift research — much better suited than ULB's 48-hour window.

5. **Leakage prevention at this level is research-grade.** 33 temporal leakage tests, 22 preprocessing tests, single test evaluation, train-only scaler fitting, validation-only threshold selection, SHA-256 hash verification — this methodology is stronger than many published fraud detection papers.

---

## Part 12 — Glossary (For Non-Technical Readers)

| Term | Plain Explanation |
|------|-----------------|
| PR-AUC | Area under the Precision-Recall curve. Measures how well the model ranks fraud above legitimate transactions. A random model scores ~0.035 (fraud rate). E1 scored 0.5317. |
| GRU | A type of neural network that reads sequences step-by-step, remembering patterns — like understanding a sentence by reading it word by word. |
| StandardScaler | A preprocessing step that rescales features so they all have the same average and spread. Essential for neural networks. |
| Paired Bootstrap | A statistical technique: randomly resample the test data 2,000 times to measure whether the performance difference is real or could be due to luck. |
| 95% CI | Confidence interval. If the CI for a difference is [−0.38, −0.34], we're 95% confident the true difference is in that range. If it doesn't include zero, the difference is statistically reliable. |
| Temporal Leakage | When future information accidentally gets used during training. Like studying tomorrow's exam questions. Completely invalidates results. |
| Frozen | The experiment is locked. Models, predictions, thresholds, and metrics cannot be changed. SHA-256 hashes verify nothing was modified. |
| Concept Drift | When fraud patterns change over time, making models trained on past data gradually less accurate. |
| card1 | An anonymized identifier grouping transactions from the same cardholder. Used to build transaction history sequences. |
| SHA-256 | A mathematical fingerprint of a file. If the file changes even by one byte, the fingerprint changes completely. Used to prove nothing was modified. |

---

*Sprint 2 Report — 2026-09-18*  
*All numbers from frozen artifacts. No values invented.*  
*Phase 2 is frozen. Next: E3 Hybrid (Sprint 3).*
