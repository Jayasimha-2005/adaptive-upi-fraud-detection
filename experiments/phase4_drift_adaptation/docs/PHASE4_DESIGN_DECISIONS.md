# PHASE 4 DESIGN DECISIONS
## Rationale for Every Key Protocol Choice

**Protocol version:** 1.1 (FINAL — approved for implementation)  
**Date:** 2026-09-19  
**Status:** FINAL — no further protocol changes permitted before implementation

---

## D1 — Primary Dataset: BAF Base (not IEEE-CIS)

**Decision:** Use BAF Base.csv as the primary dataset for Phase 4.

**Why not IEEE-CIS:**
- IEEE-CIS has a single temporal axis (transaction timestamps) but was used for predictive modeling (E1–E3). Re-using it for drift would conflate the two roles.
- IEEE-CIS is a transaction-level dataset; BAF is application-level — each addresses a different fraud domain.
- The research story is cleaner with dataset roles clearly separated: IEEE-CIS = predictive foundation, BAF = temporal adaptation.

**Why BAF Base specifically:**
- Base variant has the simplest row distribution (~96K–151K per period), cleanest fraud rate trend, and is the intended "primary" variant per the BAF paper.
- Confirmed from actual data audit: fraud rate increases monotonically from 0.875% (month 2) to 1.475% (month 7) — genuine drift signal.

---

## D2 — Temporal Split: Sequential Adaptation Loop

**Decision:** Months 0–3 training, Month 4 validation, Month 5 monitoring, Month 6 evaluation + monitoring, Month 7 protected final evaluation.

**Why sequential adaptation (not one-shot):**
The research question is *"Can a model continuously adapt?"* A single month-5 trigger followed by evaluating months 6 and 7 without further adaptation is closer to testing *"Does one retraining event help?"* — a narrower question. The sequential loop (two monitoring windows) creates a genuinely adaptive longitudinal experiment within the available 8-period constraint.

**Rationale for period assignments:**
- Months 0–3 form the "stable" reference period (fraud rates 0.875%–1.133%, KS 0.087–0.094 — low and consistent).
- Month 4 separates training from the monitoring phase and serves as the threshold-selection validation set.
- Month 5 shows the first meaningful KS acceleration (0.142), making it a natural first monitoring window.
- Month 6 serves dual role: evaluate the month-5 adaptation decision, then use month-6 data for the second monitoring window before month 7.
- Month 7 has the highest distribution shift (KS=0.182, fraud rate=1.475%) — the most challenging evaluation period.

---

## D3 — Base Model: LightGBM

**Decision:** LightGBM as the base model for both static and adaptive variants.

**Rationale:**
- Established as the strongest model in Phase 1 (E1 PR-AUC=0.531731 on IEEE-CIS).
- The research question is about ADAPTATION, not ARCHITECTURE. Using the same algorithm for both static and adaptive isolates adaptation as the only variable.
- LightGBM handles categorical features natively, handles class imbalance well, and trains fast — all appropriate for a repeated-retraining scenario.
- A neural architecture would require different training time, learning rate scheduling, and convergence behavior, confounding the adaptation comparison.

---

## D3a — Class Imbalance: `class_weight="balanced"`, Not `scale_pos_weight`

**Decision:** Use `class_weight="balanced"` for all model versions. `scale_pos_weight` is NOT used.

**Why a single deterministic method is required:**
The static and adaptive models must use exactly the same imbalance treatment. An "OR" instruction creates a risk of two different implementations being compared — conflating the imbalance method with adaptation as experimental variables.

`class_weight="balanced"` is preferred because:
- It is a single keyword with deterministic behavior (LightGBM computes class weights from the training distribution automatically)
- It is consistent across all model versions regardless of training data size
- `scale_pos_weight` requires a manual ratio computation that could be computed differently in different code paths

---

## D3b — Categorical Encoding: LightGBM Native, Not Ordinal Encoder

**Decision:** The five categorical columns (`payment_type`, `employment_status`, `housing_status`, `source`, `device_os`) are passed as LightGBM `categorical_feature`. No `OrdinalEncoder` or `LabelEncoder` is used.

**Why native categorical:**
- LightGBM natively optimizes categorical splits using the "many-to-one" method, which is strictly better than ordinal encoding for tree models (ordinal encoding imposes a spurious ordering).
- Eliminates an external preprocessing component that would otherwise need to be separately fitted, serialized, and refitted at each retraining event.
- Native handling requires only that the column dtype is `category` or the column name is listed in `categorical_feature` — no encoder object to manage.

**Per-version category handling:**
- v1: LightGBM builds category representation from months 0–3 training data
- v2: Representation uses months 0–5 (refitted with expanded training set at Month-5 trigger)
- v3: Representation uses months 0–6 (refitted with expanded training set at Month-6 trigger)
- No future-period categories are visible when fitting any version

---

## D4 — Drift Detector: PSI

**Decision:** Population Stability Index (PSI) as the primary drift detector.

**Why PSI over alternatives:**
| Detector | Pros | Cons | Verdict |
|----------|------|------|---------|
| PSI | Industry standard, interpretable, no distribution assumptions | Sensitive to binning choice | ✅ Primary |
| KS test | Rigorous statistical test per feature | Multiple testing correction needed | ✅ Secondary diagnostic only |
| Jensen-Shannon divergence | Handles categorical natively | Less widely used in industry | ❌ Not primary |
| Page-Hinkley | Online/streaming | Not well-suited to batch monthly detection | ❌ Not applicable |
| ADWIN | Adaptive windowing | Requires streaming label feedback | ❌ Violates label-free assumption |

PSI is chosen because it is the de facto standard in banking model risk management — using it aligns the experiment with real production monitoring practice.

---

## D5 — PSI Threshold: PSI ≥ 0.10, at least 6 of 29 features

**Decision:** Individual feature threshold = PSI ≥ 0.10; aggregate trigger = at least 6 of the 29 model-input features exceed this threshold.

**Why PSI ≥ 0.10 (as operational warning, not universal law):**
- PSI < 0.10: no significant population shift (common convention)
- PSI 0.10–0.25: moderate shift — worth monitoring (not automatically requiring action)
- PSI > 0.25: significant shift — action typically warranted

PSI = 0.10 is adopted as a **pre-specified operational warning threshold** based on commonly used credit-risk model monitoring conventions. It is **not** treated as a universal statistical significance threshold — PSI is sensitive to the number of bins and binning method, and does not have a sampling distribution in the way a hypothesis test does. This limitation is explicitly acknowledged.

**Why 6/29 (exact integer rule, not 20% fraction):**
- Denominator = 29 (all model-input features: 24 numerical + 5 categorical, after dropping `device_fraud_count`)
- 20% × 29 = 5.8 → rounded up to 6 for an unambiguous integer trigger
- This removes any ambiguity about whether 20% applies to numerical features only (24) or all features (29)
- A single-feature trigger is too noisy; requiring 6 features provides robustness against individual feature outliers

---

## D6 — PSI Implementation Fully Specified

**Decision:** PSI computation is fully specified before implementation to ensure reproducibility and consistency across numerical and categorical features.

**Numerical (24 features):** 10 quantile bins; edges fitted on months 0–3 only and reused without re-fitting for all future windows; out-of-range values clipped to outermost reference bins; ε = 1e-6 smoothing.

**Categorical (5 features):** Frequency-based PSI against months 0–3 reference; unseen categories assigned to `__UNSEEN__` bucket; same ε smoothing.

**Why these specs matter:**
- Without fixed bin edges, PSI values are not comparable across monitoring windows.
- Without an `__UNSEEN__` bucket, a new category in the monitoring window causes a division-by-zero or missing-bucket error.
- Smoothing prevents log(0) errors when a category is absent in one window.
- Specifying these before implementation prevents ad-hoc choices driven by monitoring data.

---

## D7 — Reference Distribution: Fixed Initial (Months 0–3)

**Decision:** The PSI reference distribution is always months 0–3, even after retraining.

**Alternative rejected:** Rolling reference (compare month N to month N-1).

**Why fixed initial:**
- Answers the question "how different is the current distribution from what the original model was trained on?"
- Rolling reference loses the connection to the model's actual training environment.
- More conservative: if distribution stabilizes after retraining, the detector might still alarm on a fixed reference — this is acceptable and interpretable.
- After retraining with months 0–5, a new reference could optionally be set for future protocols, but is not changed within the 8-period experiment.

---

## D8 — Retraining Policy: Expanding Window with Sequential Triggers

**Decision:** When drift is detected after monitoring month M, retrain on all data months 0–M. Two sequential retraining events are possible (one at month 5, one at month 6).

**Alternatives considered:**
| Policy | Description | Rejected reason |
|--------|-------------|----------------|
| Expanding window | Use all data available | ✅ Chosen — maximizes data, most reproducible |
| Rolling K-window | Use only most recent K months | Requires choosing K, not pre-specified |
| Initial + recent | Months 0–3 plus monitoring month only | Skips intermediate months arbitrarily |

**Why no global cap on triggers:**
- Capping at 1 trigger prevents the second sequential monitoring window from operating, which defeats the sequential adaptation design.
- The correct control is *at most one retraining event per monitoring window*, not a global maximum.
- With 2 monitoring windows, the maximum natural number of retraining events is 2 — this is not excessive.

**Label availability:** Fraud labels for a completed monitoring period are assumed available before the next retraining event. This is explicitly stated as a deployment assumption.

---

## D8 — Label Availability: Unsupervised Drift Detection

**Decision:** Feature distributions only for drift trigger. Fraud labels not used for triggering.

**Why:**
- In production, fraud labels arrive with delay (chargeback process).
- Using labels for drift detection makes the experiment less realistic.
- Supervised performance monitoring is a separate, interesting experiment but is NOT the primary protocol.
- PSI operates purely on input feature distributions — no labels needed.

---

## D9 — Preprocessing Refit Policy After Retraining

**Decision:** Refit categorical encoder and any scalers on the expanded training set (months 0–5) when retraining occurs.

**Rationale:**
- Using a stale encoder from months 0–3 on month 5 data could introduce unknown categories.
- Refitting is safe — the new training set is a superset of the old one.
- This must be documented as part of the retraining event log.

---

## D10 — LightGBM Configuration: Frozen Before Month-5 Monitoring

**Decision:** The LightGBM configuration is fixed before any Month-5 monitoring or evaluation. No hyperparameter tuning, model-selection procedure, or configuration change is permitted at any point during Phase 4.

**Rationale:**
- The research question is adaptation, not model optimization.
- Hyperparameter search would confound the adaptation benefit (improvement from tuning vs from adaptation).
- Allowing "BAF-specific refinement" before Month-5 creates a window for inadvertent optimization on training data, conflating HPO benefit with adaptation benefit.
- A fixed configuration applied identically to static and adaptive models ensures adaptation is the only experimental variable.

---

## D10a — Threshold Policy: Frozen from v1, Never Re-selected

**Decision:** The F1-max classification threshold is derived from Model v1's predictions on Month 4 and then permanently fixed for all model versions (v1, v2, v3). No retrained model version may re-optimize the threshold.

**Why re-selection is invalid after retraining:**

When v2 is trained on months 0–5, Month 4 is part of its training data. Evaluating v2 on Month 4 to select a threshold constitutes validation contamination — the model has already seen and been fitted on those exact rows. The threshold selected this way would be inflated relative to truly held-out data.

The correct and scientifically defensible approach:

```
Month 4 → v1 predictions → F1-max → 🔒 FROZEN
                                       ↓
                             Applied to v2 and v3 unchanged
```

PR-AUC and ROC-AUC are threshold-independent and are unaffected by this constraint.

---

## D11 — Statistical Test: Paired Bootstrap

**Decision:** 2,000-resample paired bootstrap, seed=42, on evaluation period observations.

**Rationale:**
- Same methodology as Phases 1–3 (consistent across entire project).
- Paired because static and adaptive are evaluated on the same rows in the same period.
- Bootstrap is appropriate for PR-AUC estimation under imbalance.
- Seed=42 matches all prior experiments.

---

## D12 — BAF Variants: Secondary Stress Tests Only

**Decision:** Primary experiment uses Base only. Variants I–V reserved for secondary stress tests if needed.

**Rationale:**
- Variants share the same 1M rows with modified data generation (bias scenarios).
- Primary contribution is the adaptation comparison under natural temporal drift (Base).
- Variant experiments should be labeled explicitly (e.g., "Phase 4 Variant III Stress Test") and not mixed with primary results.
- Variants III–V have different period distributions (see audit), making direct comparison to Base harder.

---

## D13 — Optional Model C: Periodic Retraining

**Decision:** Include as an optional secondary control, not as a primary comparison.

**Definition:** Retrain after every monitoring period unconditionally (month 5 → retrain, month 6 → retrain).

**Value:** Separates "retraining helps" from "DRIFT-TRIGGERED retraining helps." If periodic retraining also improves performance, the benefit may not be specific to drift detection.

**Condition for inclusion:** Must be defined and fixed before evaluation. If included, report:
- Model C vs Static
- Model B (drift-triggered) vs Static
- Do not create "adaptive wins" by cherry-picking comparisons.
