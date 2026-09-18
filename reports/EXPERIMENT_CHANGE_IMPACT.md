# Experiment Change Impact Report
**Generated:** 2026-09-18 | **Audit only. No changes implemented.**

---

## 1. Experiment Change Matrix

| Existing Experiment | Dataset | Modify for ULB? | New Experiment Needed? | Reason |
|--------------------|---------|-----------------|-----------------------|--------|
| E1 LightGBM | IEEE-CIS | **NO** | ULB-E1 (new) | Different domain, different features, independent pipeline |
| E1 Phase 1.5 Hardening | IEEE-CIS | **NO** | ULB-E1.5 (optional) | Same rationale; hardening methodology reusable conceptually |
| E2a GRU (unscaled) | IEEE-CIS | **NO** | Not applicable | ULB has no entity ID — locked sequence spec cannot apply |
| E2b GRU (scaled) | IEEE-CIS | **NO** | Not applicable | Same reason; no entity ID in ULB |
| E3 Hybrid | IEEE-CIS | **NO** | ULB-E3 (optional) | Would need ULB tabular baseline first |
| Concept Drift | IEEE-CIS + BAF | **NO** | BAF is already present | BAF variants purpose-built for drift; ULB insufficient (48h) |
| Adaptive Retraining | TBD | **NO** | BAF-based | BAF 8-month ordinal time; ULB 48h insufficient |
| Explainability | IEEE-CIS | **NO** | Optional ULB extension | SHAP on ULB meaningful only after ULB model trained |
| Kafka / Spark Streaming | Any | **NO** | ULB can be replayed | ULB Time column allows time-ordered streaming replay |
| FastAPI Serving | Any | **NO** | Endpoint-agnostic | Serving layer is dataset-independent |
| Integration | All | **NO** | No change | Integration tests are methodology-level, not dataset-specific |

---

## 2. Q&A: 19 Final Questions

**Q1: Is the ULB dataset valid and usable?**  
Yes. 284,807 rows, 0 missing values, `Time` is monotone increasing. **1,081 duplicate rows exist** — a deduplication decision is needed before any modeling. The dataset is a well-established published benchmark.

**Q2: Is it actually different from any existing credit-card dataset?**  
Yes. The `Credit card Fraud detection` folder contains `credit_card_fraud_10k.csv` — a 10,000-row **toy/synthetic dataset** with no PCA V features, no entity identifiers, no real timestamps, and manufactured feature values. It is completely different from ULB and is **not a published benchmark**. The two datasets share no columns, structure, or origin.

**Q3: Is ULB useful for this research project?**  
Yes, in a limited but legitimate role: external validation of the tabular baseline methodology and extreme class imbalance demonstration (0.17% fraud rate vs. 3.5% in IEEE-CIS).

**Q4: What exact research question does ULB help answer?**  
"Does the tabular LightGBM fraud detection methodology generalize to a different fraud domain (European credit card transactions, PCA-anonymized features, extreme 578:1 class imbalance) without modification?"

**Q5: Does ULB require changing E1?**  
**No.** E1 is an IEEE-CIS experiment. ULB creates a separate, independent pipeline.

**Q6: Does ULB require changing E2a?**  
**No.** E2a uses IEEE-CIS entity-grouped sequences. ULB has no entity identifier.

**Q7: Does ULB require changing E2b?**  
**No.** Same reason as E2a.

**Q8: Does ULB require changing E3?**  
**No.** E3 (hybrid) is currently not implemented. ULB would require its own tabular baseline before any hybrid.

**Q9: Should ULB be used for concept drift?**  
**No.** 48-hour temporal coverage is insufficient to observe or study meaningful concept drift. BAF (6 × 1M rows with 8 ordinal months of data and purpose-designed drift variants) is the correct dataset for concept drift research.

**Q10: Should ULB be used for adaptive retraining?**  
**No.** Adaptive retraining requires multiple temporal windows with distribution shift. ULB covers 48 hours — too narrow for any meaningful retraining experiment.

**Q11: Should ULB be used for streaming?**  
**Marginally.** ULB's `Time` column enables time-ordered transaction replay for streaming demonstrations. It can serve as a secondary streaming dataset alongside IEEE-CIS or BAF. But it should not be the primary streaming benchmark due to its 48h duration.

**Q12: Should ULB be used for external validation?**  
**Yes — this is its primary research role.** Testing whether a model pipeline trained on IEEE-CIS (e-commerce, US) generalizes to ULB (credit cards, Europe, PCA features) is a meaningful cross-dataset validation experiment.

**Q13: Should any datasets be merged?**  
**No.** IEEE-CIS V-features and ULB V-features are semantically distinct. BAF features are entirely different (interpretable bank account features). PaySim features are mobile money simulation. Merging any of these would be methodologically invalid. Each dataset has an independent pipeline.

**Q14: What preprocessing is dataset-specific?**  
All fitted preprocessing objects are dataset-specific:
- Missing-value imputers: Dataset-specific (ULB has no NaN; IEEE-CIS has extensive NaN)
- Scalers (StandardScaler, etc.): Must be fitted independently per dataset
- Label/ordinal encoders: Dataset-specific
- Feature manifests: Dataset-specific
- E1 `preprocessing.joblib` must not be reused for ULB or BAF

Methodology (train-only fitting, val/test frozen transformation) is reusable as a pattern, not as an object.

**Q15: What NEW experiments should be added?**  
| New Experiment | Dataset | Purpose |
|---------------|---------|---------|
| ULB-E1 | ULB | Tabular LightGBM baseline on ULB (external validation) |
| BAF-E1 | BAF Base | Tabular baseline on BAF Base for concept drift anchor |
| BAF Concept Drift | BAF Variants I–V | Drift detection and adaptation |
| PaySim-E1 | PaySim | Optional: mobile money tabular baseline |

**Q16: What existing experiments must remain completely unchanged?**  
E1 LightGBM, E1 Phase 1.5, E2a GRU, E2b GRU, STEP 9 evaluation — **all frozen as-is**.

**Q17: What is the recommended final research architecture?**  
See `DATASET_SUITABILITY_AUDIT.md` Section on Proposed Experiment Structure.

---

## 3. "Credit card Fraud detection" 10K Dataset — Verdict

| Property | Value |
|---------|-------|
| Rows | 10,000 |
| Columns | 10 |
| Type | **Synthetic toy** |
| Published benchmark? | **No** |
| Entity identifier | **None** |
| Real timestamps | **None** (transaction_hour only) |
| V features | **None** |
| Recommended research role | **NOT USEFUL** for rigorous experiments |
| Action | Leave in place, document as toy dataset, do not use in primary experiments |

---

## 4. Recommended Experiment Structure

```
PHASE 1      IEEE-CIS tabular baseline (E1)                  ✅ COMPLETE
PHASE 1.5    IEEE-CIS robustness hardening                    ✅ COMPLETE
PHASE 2      IEEE-CIS temporal GRU (E2a, E2b)                ✅ COMPLETE

PHASE 3      IEEE-CIS hybrid model                           [NEXT — pending approval]

PHASE 4      Cross-dataset tabular evaluation
               ├── ULB-E1   (tabular LightGBM on ULB)
               ├── BAF-E1   (tabular LightGBM on BAF Base)
               └── PaySim-E1 (optional)

PHASE 5      Concept drift
               ├── BAF Variants I–V (purpose-designed drift)
               └── IEEE-CIS temporal concept drift (if applicable)

PHASE 6      Adaptive retraining
               └── BAF-based (8-month ordinal time windows)

PHASE 7      Explainability (SHAP)
               └── IEEE-CIS E1 / E2b (primary)

PHASE 8      Kafka + Spark streaming
               └── IEEE-CIS + ULB replay

PHASE 9      FastAPI serving

PHASE 10     End-to-end integration
```

**No dataset merging at any phase.**

---

*Experiment Change Impact — 2026-09-18 | Audit only. No training. No model changes.*
