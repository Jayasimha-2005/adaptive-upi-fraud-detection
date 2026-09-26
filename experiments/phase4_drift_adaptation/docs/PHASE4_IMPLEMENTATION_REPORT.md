# PHASE 4 IMPLEMENTATION REPORT

**Project:** Adaptive Financial Fraud Detection with Temporal Modeling,
Concept Drift Adaptation, Explainable AI, and Real-Time Stream Processing

**Phase:** 4 — Concept Drift Detection + Adaptive Retraining

**Protocol version:** 1.1 (FINAL)
**Run ID:** 20260919_162551
**Date:** 2026-09-19
**Status:** COMPLETE

---

## 1. Research Question

> **Does drift-triggered adaptive retraining maintain fraud-detection
> performance better than a static model under temporal distribution
> changes in the BAF dataset?**

The scientific variable under test is **adaptation** — specifically,
whether expanding the training window after a pre-specified, label-free
drift trigger (PSI ≥ 0.10 in ≥ 6 of 29 features) produces a measurably
better fraud detector than keeping the model frozen on its initial
training data.

---

## 2. Hypotheses

- **H₀ (null):** Drift-triggered retraining produces no statistically
  significant change in PR-AUC compared to the static model.
- **H₁ (alternative):** Drift-triggered retraining produces a statistically
  significant improvement in PR-AUC.

---

## 3. Dataset

| Property | Value |
|----------|-------|
| Name | BAF Base |
| File | `Datasets/BAF/Base.csv` |
| SHA256 (first 16 hex) | `7bf10a37ce07e72e` |
| Rows | 1,000,000 |
| Columns | 32 |
| Temporal column | `month` |
| Periods | 0–7 (8 months) |
| Target | `fraud_bool` (binary) |
| Missing values | None |
| Duplicate rows | None |

---

## 4. Dataset Integrity

All per-period row counts and fraud counts verified against protocol
expectations before any model training:

| Month | Rows | Fraud | Fraud rate |
|-------|------|-------|-----------|
| 0 | 132,440 | 1,500 | 1.133% |
| 1 | 127,620 | 1,198 | 0.939% |
| 2 | 136,979 | 1,198 | 0.875% |
| 3 | 150,936 | 1,392 | 0.922% |
| 4 | 127,691 | 1,452 | 1.137% |
| 5 | 119,323 | 1,411 | 1.183% |
| 6 | 108,168 | 1,450 | 1.341% |
| 7 | 96,843 | 1,428 | 1.475% |

Fraud rate increases steadily from 0.875% (Month 2) to 1.475% (Month 7),
confirming a temporal distribution shift that motivates the experiment.

---

## 5. Feature Set

**29 model-input features** (24 numerical + 5 categorical).
Excluded: `fraud_bool`, `month`, `device_fraud_count` (constant = 0).

**Numerical (24):** income, name_email_similarity,
prev_address_months_count, current_address_months_count, customer_age,
days_since_request, intended_balcon_amount, zip_count_4w, velocity_6h,
velocity_24h, velocity_4w, bank_branch_count_8w,
date_of_birth_distinct_emails_4w, credit_risk_score, email_is_free,
phone_home_valid, phone_mobile_valid, bank_months_count, has_other_cards,
proposed_credit_limit, foreign_request, session_length_in_minutes,
keep_alive_session, device_distinct_emails_8w

**Categorical (5):** payment_type, employment_status, housing_status,
source, device_os (LightGBM native categorical — no ordinal encoding)

---

## 6. Temporal Split

| Period | Months | Rows | Purpose |
|--------|--------|------|---------|
| Initial training | 0–3 | 547,975 | Model v1 training |
| Validation | 4 | 127,691 | Threshold selection ONLY |
| Monitoring Window 1 | 5 | 119,323 | PSI drift detection |
| Evaluation 1 + Monitor 2 | 6 | 108,168 | Model evaluation + PSI |
| Protected final eval | 7 | 96,843 | Final comparison |

No random splits. All splits are deterministic on the `month` column.

---

## 7. Static Model (Model A)

**Algorithm:** LightGBM (LGBMClassifier)
**Training data:** Months 0–3 (547,975 rows, 5,288 fraud, 0.965% rate)
**Configuration (locked):**
```
n_estimators=500, learning_rate=0.05, max_depth=6,
num_leaves=63, min_child_samples=50, subsample=0.8,
colsample_bytree=0.8, class_weight="balanced",
random_state=42
```
**Categorical handling:** LightGBM native (not ordinal encoder)
**Training time:** 28.3s
**Model hash (SHA256[:16]):** `2cb7bc1247114aaa`
**Static model is NEVER retrained.** Its weights are frozen at v1.

---

## 8. Adaptive Model (Model B)

Same algorithm and hyperparameters as static model.
Only the training data changes at each retraining event.

**Adaptation path executed: PATH D**

```
v1 (months 0-3)
    ↓ Month 5 PSI triggered
v2 (months 0-5)
    ↓ Month 6 PSI triggered
v3 (months 0-6)  ← final version for Month 7 evaluation
```

| Version | Training months | Rows | Fraud | Training time |
|---------|----------------|------|-------|--------------|
| v1 | 0–3 | 547,975 | 5,288 | 28.3s |
| v2 | 0–5 | 794,989 | 8,151 | 16.5s |
| v3 | 0–6 | 903,157 | 9,601 | 21.9s |

---

## 9. PSI Methodology

**Reference distribution:** Months 0–3 (built once, never reset)
**Numerical features:** 10 quantile bins per feature, edges fitted on
months 0–3, reused for both monitoring windows
**Categorical features:** Frequency distribution from months 0–3;
`__UNSEEN__` bucket for categories not in reference
**Smoothing:** ε = 1×10⁻⁶
**Feature trigger threshold:** PSI ≥ 0.10
**Aggregate trigger rule:** ≥ 6 of 29 features

---

## 10. Threshold Methodology

- Model v1 applied to Month 4 to generate fraud probabilities
- 200 thresholds swept from 0.01 to 0.99
- F1-max criterion selects threshold
- **Frozen threshold: 0.8964**
- Validation F1 at this threshold: 0.2458
- Applied identically to v1, v2, v3 predictions

> Note: The high threshold (0.8964) is characteristic of the low base
> rate (~1%) and the class_weight="balanced" training. The model assigns
> high probabilities to a small fraction of transactions. This is expected.

---

## 11. Drift Results

### Month 5 (Monitoring Window 1)

| Metric | Value |
|--------|-------|
| Features with PSI ≥ 0.10 | **17 / 29** |
| Trigger threshold | 6 / 29 |
| Triggered | **YES** |
| n_observations | 119,323 |

**Top-5 shifted features:**
1. `prev_address_months_count` — PSI = 19.11 (extreme shift)
2. `velocity_4w` — PSI = 6.66
3. `bank_months_count` — PSI = 6.21
4. `income` — PSI = 4.47
5. `velocity_24h` — PSI = 3.54

### Month 6 (Monitoring Window 2)

| Metric | Value |
|--------|-------|
| Features with PSI ≥ 0.10 | **18 / 29** |
| Trigger threshold | 6 / 29 |
| Triggered | **YES** |
| n_observations | 108,168 |

**Top-5 shifted features:**
1. `prev_address_months_count` — PSI = 18.82 (sustained extreme shift)
2. `velocity_4w` — PSI = 6.66
3. `bank_months_count` — PSI = 6.19
4. `income` — PSI = 4.44
5. `velocity_24h` — PSI = 4.38

---

## 12. Retraining Events

| Event | Month | Trigger | Retrained to |
|-------|-------|---------|-------------|
| 1 | 5 | 17/29 ≥ 6 ✓ | v2 (months 0–5) |
| 2 | 6 | 18/29 ≥ 6 ✓ | v3 (months 0–6) |

Both monitoring windows triggered. Experiment followed **Path D** (all
possible retraining events occurred).

---

## 13. Model Versions Used

| Period | Static model | Adaptive model |
|--------|-------------|----------------|
| Month 6 | v1 | v2 |
| Month 7 | v1 | v3 |

---

## 14. Month 6 Results

| Metric | Static v1 | Adaptive v2 | Δ (A−S) |
|--------|-----------|------------|---------|
| **PR-AUC** | **0.1657** | **0.1725** | **+0.0068** |
| ROC-AUC | 0.8481* | — | — |
| F1 | — | — | — |
| Recall@FPR=1% | — | — | — |

*Full metric tables in `artifacts/metrics/comparison_month6.json`

---

## 15. Month 7 Results

| Metric | Static v1 | Adaptive v3 | Δ (A−S) |
|--------|-----------|------------|---------|
| **PR-AUC** | **0.1609** | **0.2010** | **+0.0402** |
| P@100 | — | — | — |
| P@500 | — | — | — |
| P@1000 | — | — | — |

*Full metric tables in `artifacts/metrics/comparison_month7.json`

---

## 16. Paired Bootstrap

**Method:** Paired bootstrap, 2000 resamples, seed=42, PR-AUC metric

### Month 6
| Item | Value |
|------|-------|
| Static PR-AUC | 0.165667 |
| Adaptive PR-AUC | 0.172467 |
| Observed delta | +0.006800 |
| 95% CI | [−0.002316, +0.015793] |
| Interpretation | CI crosses zero — no statistically significant difference |

### Month 7
| Item | Value |
|------|-------|
| Static PR-AUC | 0.160858 |
| Adaptive PR-AUC | 0.201027 |
| Observed delta | **+0.040169** |
| 95% CI | **[+0.026752, +0.053843]** |
| Interpretation | **CI does not cross zero — adaptive significantly higher** |

---

## 17. Confidence Interval (Primary Comparison)

**Primary metric:** PR-AUC on Month 7 (most temporally displaced evaluation)

- Adaptive − Static = **+0.0402**
- 95% CI: **[+0.0268, +0.0538]**
- The CI is entirely above zero.
- Result is statistically significant at the 95% level.

---

## 18. Figures

All 13 protocol-required figures saved to `figures/`:

| File | Description |
|------|-------------|
| `fig01_fraud_prevalence_by_month.png` | Fraud rate by month 0–7 |
| `fig02_psi_by_feature_month5.png` | Month 5 PSI per feature |
| `fig03_psi_by_feature_month6.png` | Month 6 PSI per feature |
| `fig04_psi_trigger_count.png` | PSI-triggered features by window |
| `fig05_adaptation_timeline.png` | v1 → v2 → v3 timeline |
| `fig06_prauc_comparison.png` | PR-AUC comparison (M6 + M7) |
| `fig07_pr_curve_month6.png` | PR curve Month 6 |
| `fig08_pr_curve_month7.png` | PR curve Month 7 |
| `fig09_roc_curve_month6.png` | ROC curve Month 6 |
| `fig10_roc_curve_month7.png` | ROC curve Month 7 |
| `fig11_precision_at_topk_month7.png` | P@K Month 7 |
| `fig12_recall_at_fpr_month7.png` | Recall@FPR Month 7 |
| `fig13_bootstrap_delta_month7.png` | Bootstrap delta distribution |

---

## 19. Leakage Audit

All 10 leakage categories audited in `PHASE4_FINAL_AUDIT.md`.
**Result: 10/10 CLEAR — no violations.**

---

## 20. Reproducibility Information

| Item | Value |
|------|-------|
| Python | 3.x (Windows, Miniconda) |
| LightGBM | installed version |
| Random seed | 42 |
| Dataset hash (SHA256[:16]) | `7bf10a37ce07e72e` |
| Protocol version | 1.1 |
| Model v1 hash | `2cb7bc1247114aaa` |
| Model v2 hash | `e4f249253fab9125` |
| Model v3 hash | `cd3908ae85418ed4` |
| Threshold artifact hash | `c04cb31bb1abf9de` |
| Integrity tests | 52/52 PASSED |
| Preflight audit | 84/84 PASSED |

---

## 21. Limitations

1. **BAF is synthetic** — conclusions are scoped to the BAF benchmark and
   may not generalise to real fraud datasets with different distributional
   properties.
2. **Only 8 temporal periods** — limited temporal resolution. Long-term
   drift behaviour cannot be assessed.
3. **Single dataset** — no cross-dataset validation of the adaptation mechanism.
4. **Single drift detector** — only PSI was used as the primary trigger. KS,
   JS-divergence, and label-based detectors were not compared.
5. **Single base model** — only LightGBM tested. Whether adaptation benefits
   depend on the model class is unknown.
6. **Single seed** — no cross-seed variance estimation.
7. **Expanding window only** — sliding window retraining was not tested.
8. **PSI threshold fixed a priori** — the sensitivity of results to
   alternative trigger thresholds (e.g., 0.05, 0.20) is not assessed.
9. **`prev_address_months_count` dominates PSI** — PSI = 19.11 in Month 5
   for a single feature. Whether this feature drives the adaptive advantage
   is not separately quantified.
10. **No entity tracking** — transactions are treated independently.
    Cardholder-level temporal patterns are not modelled.

---

## 22. Scientific Interpretation

### Month 6 (Adaptive v2 vs Static v1)

The adaptive model (v2, retrained on months 0–5 after a Month 5 drift
trigger) showed a small numerical improvement in PR-AUC:
Δ = +0.0068 (95% CI: −0.002, +0.016).

The 95% confidence interval crosses zero. **This result does not
constitute a statistically significant improvement at the 95% level.**
The Month 6 result is inconclusive: the observed positive delta may
be explained by sampling variation.

### Month 7 (Adaptive v3 vs Static v1)

The adaptive model (v3, retrained on months 0–6 after both Month 5 and
Month 6 drift triggers) showed a larger improvement in PR-AUC:
Δ = +0.0402 (95% CI: +0.027, +0.054).

The 95% confidence interval is entirely above zero. **This result is
statistically significant.** By Month 7 — the period with the highest
observed distribution shift — the adaptive model's advantage over the
frozen static model is both numerically substantial (+25% relative
improvement in PR-AUC) and statistically reliable at the 95% level.

**The experiment provides evidence that drift-triggered adaptive
retraining can improve fraud detection performance under sustained
temporal distribution shift, particularly as the evaluation horizon
extends further from the static model's training period.**

---

## 23. What This Experiment Does NOT Establish

- This experiment does **not** establish that adaptive retraining is
  universally superior to static modelling.
- The result applies specifically to **BAF Base**, **LightGBM**,
  **PSI with a 6/29 trigger**, and **an expanding-window retraining policy**.
- The advantage is observed at **Month 7** (the most displaced evaluation
  point). At **Month 6** (one retraining event earlier), the result is not
  statistically significant.
- This experiment does **not** establish that PSI is the optimal drift detector
  for this problem.
- The experiment does **not** establish the optimal PSI threshold, trigger
  count, or retraining frequency.
- The experiment does **not** establish generalisability to real-world
  production fraud systems.

---

## PHASE 4 FINAL STATUS

```
PHASE 4 STATUS:       COMPLETE
DATASET:              BAF Base
DATASET HASH:         7bf10a37ce07e72e
PROTOCOL:             v1.1
MODEL A:              Static v1 (frozen, months 0-3)
MODEL B:              Drift-Triggered Adaptive
MONTH 4 THRESHOLD:    0.8964 (F1-max, v1 predictions, FROZEN)
MONTH 5 PSI:          17/29 features >= 0.10 | TRIGGERED
MONTH 5 RETRAIN:      YES -> v2 (months 0-5)
ADAPTIVE M6 VERSION:  v2
MONTH 6 RESULTS:      Static PR-AUC=0.1657 | Adaptive PR-AUC=0.1725 | Delta=+0.0068
MONTH 6 PSI:          18/29 features >= 0.10 | TRIGGERED
MONTH 6 RETRAIN:      YES -> v3 (months 0-6)
ADAPTIVE M7 VERSION:  v3
MONTH 7 RESULTS:      Static PR-AUC=0.1609 | Adaptive PR-AUC=0.2010 | Delta=+0.0402
PRIMARY PR-AUC DELTA: Adaptive - Static = +0.0402
95% CI:               [+0.0268, +0.0538]
BOOTSTRAP:            2000 resamples, seed=42
INTEGRITY TESTS:      52/52 PASSED
LEAKAGE AUDIT:        10/10 CLEAR — PASS
ARTIFACT HASHING:     COMPLETE
FINAL INTERPRETATION: Adaptive retraining produced a statistically significant
                      improvement in PR-AUC at Month 7 (Δ=+0.0402, CI excludes
                      zero). Month 6 result is inconclusive (CI crosses zero).
LIMITATIONS:          See Section 21 above.
```
