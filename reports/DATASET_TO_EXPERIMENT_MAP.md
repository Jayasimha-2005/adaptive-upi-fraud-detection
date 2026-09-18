# Dataset-to-Experiment Map
**Generated:** 2026-09-18 | Audit only. No training.

---

## Overview

Each dataset has an independent processing pipeline. No datasets are merged.  
The methodological approach (chronological split, train-only preprocessing, validation-only threshold) is shared across all datasets — but the fitted objects (scalers, encoders, models) are strictly dataset-specific.

---

## Dataset-to-Experiment Table

| Dataset | Primary Role | Secondary Role | Experiments | Why |
|---------|-------------|---------------|-------------|-----|
| **IEEE-CIS** | PRIMARY BENCHMARK | Temporal modeling anchor | E1, E1.5, E2a, E2b, E3, Explainability, Streaming | Real-world-derived, entity IDs (card1), fine-grained timestamps, 406 features, 590K transactions — ideal for rigorous baseline and temporal comparison |
| **BAF** | CONCEPT DRIFT | Adaptive retraining | BAF-E1 baseline, BAF Variants I–V drift, Adaptive retraining | Purpose-designed for concept drift (NeurIPS 2022), 6×1M rows, 8 ordinal months, 5 drift variant types, consistent schema across variants |
| **PaySim** | STREAMING DEMONSTRATION | Domain validation | PaySim-E1 (optional), Kafka streaming replay | Named entity (`nameOrig`), ordered `step` column, 6.36M transactions — enables realistic streaming simulation. Mobile money domain adds breadth. |
| **ULB** | EXTERNAL VALIDATION | Extreme imbalance benchmark | ULB-E1 tabular baseline | Real-world European credit card data, 0.17% fraud (578:1), PCA-anonymized, independent domain — validates tabular methodology generalization |
| **10K toy** | NOT USED | — | None | Too small (10K), no entity ID, no real timestamps, not a published benchmark, unknown provenance |

---

## Per-Dataset Experiment Specifications

### IEEE-CIS

```
Primary dataset for all foundational experiments.

E1 LightGBM:
  Status: COMPLETE
  Data:   590,540 transactions, 406 features, chronological split
  Model:  LightGBM (is_unbalance=True, 406 features)
  Result: Test PR-AUC = 0.531731

E1.5 Phase 1.5 Hardening:
  Status: COMPLETE
  Data:   Same E1 test set (78,542)
  Tests:  Bootstrap, operational metrics, calibration, encoding comparison
  Result: PR-AUC CI [0.5126, 0.5504]; ECE=0.0524; Precision@Top100=0.98

E2a GRU (unscaled):
  Status: COMPLETE
  Data:   550,559 sequences (398,312 train / 75,727 val / 76,520 test)
  Model:  FraudGRU(406, 64, 1, dropout=0.2), unscaled input
  Result: Val PR-AUC = 0.0345 (near random). Not test-evaluated.

E2b GRU (StandardScaler):
  Status: COMPLETE — FROZEN
  Data:   Same sequences, StandardScaler (train-only)
  Model:  Same architecture, scaled input
  Result: Val PR-AUC = 0.1377 (epoch 2), Test PR-AUC = 0.1683
          Δ PR-AUC (vs E1) = −0.3584, 95% CI [−0.3756, −0.3395]

E3 Hybrid (PLANNED — NOT STARTED):
  RQ:     Does temporal info complement tabular features?
  Design: GRU embedding + E1 tabular features → MLP head
  Data:   Same 76,520 common test population
  Baseline: E1 (frozen), E2b (frozen)

Explainability (PLANNED):
  SHAP on E1 and E3
  Feature importance, partial dependence, behavioral interpretation

Streaming (PLANNED):
  Ordered replay of IEEE-CIS transactions in Kafka
  E1/E3 inference in Spark Streaming
```

### BAF (Bank Account Fraud)

```
Primary dataset for concept drift and adaptive retraining experiments.

BAF-E1 Baseline (PLANNED):
  Status: Not started
  Data:   BAF Base.csv (1M rows, 32 features, fraud_bool target)
  Model:  LightGBM (same methodology as IEEE-CIS E1)
  Purpose: Establish clean tabular baseline for drift comparison

BAF Concept Drift — Variants I–V (PLANNED):
  Status: Not started
  Data:   Base → V1 → V2 → V3 → V4 → V5 (sequential variants)
  Design: Train on Base, evaluate on each Variant
          Measure PR-AUC degradation across variants
  Purpose: Quantify impact of different drift types on model performance

BAF Adaptive Retraining (PLANNED):
  Status: Not started
  Data:   8 ordinal months (month 0–7)
  Design: Monthly retraining windows; detect drift → retrain → measure recovery
  Purpose: Demonstrate adaptive learning under measured drift
```

### PaySim

```
Streaming demonstration and optional domain baseline.

PaySim-E1 (OPTIONAL):
  Status: Not started
  Data:   6.36M rows, 11 features, isFraud target
  Note:   Only CASH_OUT and TRANSFER contain fraud — may need type filtering

PaySim Streaming (PLANNED):
  Status: Not started
  Data:   743-step time series (simulated hours)
  Design: Replay transactions in step order via Kafka
          E1-equivalent model inference via Spark Streaming
  Purpose: Demonstrate real-time scoring pipeline
```

### ULB

```
External validation and extreme imbalance demonstration.

ULB-E1 (PLANNED):
  Status: Not started
  Data:   284,807 rows (283,726 unique), 31 features, Class target
  Preprocessing: New scaler (ULB-specific); deduplication decision needed
  Model:  LightGBM with extreme imbalance handling (578:1)
  Purpose: Test whether tabular methodology generalizes to different fraud domain
  Important: ULB V1–V28 ≠ IEEE-CIS V1–V339. Independent pipelines required.
  Limitation: 48-hour temporal window — no temporal or drift experiments
```

### 10K Toy Dataset

```
NOT USED.
  Excluded from all experiments.
  Reasons: 10K rows insufficient, no entity ID, no real timestamps,
           not a published benchmark, unknown provenance.
  Action:  Leave file in place, document exclusion.
```

---

## Feature Space Compatibility

| Dataset A | Dataset B | V-feature compatibility | Can combine? |
|----------|----------|------------------------|-------------|
| IEEE-CIS (V1–V339) | ULB (V1–V28) | **INCOMPATIBLE** — different semantics | ❌ NO |
| IEEE-CIS (406 features) | BAF (32 features) | **INCOMPATIBLE** — completely different feature spaces | ❌ NO |
| IEEE-CIS | PaySim | **INCOMPATIBLE** — different domains and features | ❌ NO |
| BAF Base | BAF Variant I–V | **COMPATIBLE** — same schema, same 32 features | ✅ YES (for drift) |

---

## Preprocessing Reusability

| Component | Approach | Reusable Concept? | Fitted Objects Shareable? |
|---------|---------|------------------|--------------------------|
| Chronological split | Split by time boundary | ✅ Reusable concept | N/A |
| Missing value imputer | Train-only fit | ✅ Reusable concept | ❌ Dataset-specific |
| Categorical encoder | Target encoding (train-only) | ✅ Reusable concept | ❌ Dataset-specific |
| StandardScaler | Train-only fit | ✅ Reusable concept | ❌ Dataset-specific |
| Feature selection | Dataset-specific | ❌ Dataset-specific | ❌ Dataset-specific |
| pos_weight / is_unbalance | Train fraud rate | ✅ Reusable concept | ❌ Computed per dataset |
| Bootstrap protocol | 2000 resamples, seed=42 | ✅ Reusable concept | N/A |

---

*Dataset-to-Experiment Map — 2026-09-18 | Audit only. No training.*
