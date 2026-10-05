# Target Repository Audit: `adaptive-upi-fraud-detection`
**Baseline Branch:** `Upto_Phase-4`  
**Integration Working Branch:** `integration/member1-member2`  
**Target Commit HEAD:** `8d600ba` ("Update README: complete project overview, all phases, teammate guidance, model comparison")  
**Audit Date:** October 2026  
**Auditor:** Senior Software & Research Integration Engineer

---

## 1. Executive Summary & Objective

This document provides a comprehensive structural, methodological, and artifact audit of the canonical research repository `adaptive-upi-fraud-detection`. The repository contains a rigorous, multi-phase machine learning fraud detection research track that progresses through Phase 1 (LightGBM baseline), Phase 1.5 (robustness & calibration hardening), Phase 2 (temporal sequence modeling with GRU), Phase 3 (tabular-temporal hybrid architectures), and Phase 4 (concept drift monitoring and temporal adaptation on Bank Account Fraud).

All Phase 1–4 research artifacts, models, thresholds, evaluation tables, and freeze manifests are **strictly immutable**. This audit establishes the baseline against which the real-time event streaming and distributed processing layers from Member 1 (Apache Kafka) and Member 2 (Apache Spark & Apache Flink) will be merged.

---

## 2. Directory Structure Inventory

The target repository at `c:\Users\Harini\Documents\GitHub\Jayasimha-github\adaptive-upi-fraud-detection` contains the following top-level directory structure on `Upto_Phase-4`:

```text
adaptive-upi-fraud-detection/
├── .gitignore                      # Git exclusion rules (datasets, logs, temp files)
├── README.md                       # Canonical repository guide & teammate instructions
├── requirements.txt                # Core research Python dependencies
├── download_dataset.py             # Script to download datasets (IEEE-CIS, BAF, PaySim)
├── configs/                        # Experiment configurations
│   └── phase1_lightgbm.yaml        # Phase 1 LightGBM hyperparameters & training config
├── src/                            # Modular research source code
│   ├── audit/                      # Data auditing & feature provenance utilities
│   ├── data/                       # Dataset loading, temporal splitting, validation
│   ├── evaluation/                 # Metrics, bootstrap confidence intervals, calibration
│   ├── features/                   # IEEE-CIS feature engineering pipeline
│   ├── models/                     # LightGBM baseline model wrapper
│   └── utils/                      # Config parsing & logging utilities
├── dataset_analysis/               # Comprehensive forensic analysis across 5 candidate datasets
│   ├── column_dictionary.csv       # Column dictionaries across datasets
│   ├── dataset_comparison.csv      # Statistical comparison matrix
│   ├── FINAL_DATASET_RECOMMENDATION.md # Research verdict on dataset roles
│   ├── ieee_cis/                   # Detailed profiles of IEEE-CIS tables
│   └── visualizations/             # 11 forensic EDA visualizations (.png)
├── analysis_scripts/               # Exploratory data analysis scripts for datasets
├── experiments/                    # Execution pipelines and immutable artifacts
│   ├── run_phase1.py               # Phase 1 execution runner
│   ├── run_phase15_extend.py       # Phase 1.5 extension runner
│   ├── E1_lightgbm/                # FROZEN E1 MODEL ARTIFACTS
│   │   ├── model.txt               # Canonical LightGBM booster model (6.64 MB)
│   │   ├── preprocessing.joblib    # Preprocessing pipeline (0.06 MB)
│   │   ├── predictions.parquet     # Test set predictions (3.19 MB)
│   │   ├── feature_names.json      # 406 canonical feature names
│   │   ├── metrics.json            # Final test evaluation metrics
│   │   └── feature_importance.csv  # Split & gain feature importances
│   ├── E1_encoding_comparison/     # Categorical encoding benchmark results
│   ├── E3_hybrid/                  # Phase 3 hybrid models, tests, docs, artifacts
│   └── phase4_drift_adaptation/    # Phase 4 concept drift & adaptation experiment
├── phase2_gru/                     # Phase 2 temporal GRU modeling module
│   ├── artifacts/                  # E2a (unscaled) & E2b (scaled) model checkpoints (.pt)
│   ├── reports/                    # Sequence generation specs & paired comparison reports
│   ├── src/                        # GRU sequence builder, dataset, model, trainer
│   └── tests/                      # 80 automated unit & integrity tests
├── reports/                        # Research completion reports & sprint deliverables
│   ├── phase1/                     # LightGBM baseline & operational reports (.md, .pdf)
│   ├── phase2/                     # Phase 2 freeze manifest & sequence specifications
│   ├── DATASET_FORENSICS.json      # Machine-readable dataset suitability audit
│   └── RESEARCH_STATUS_COMPLETE_REVIEW.md # Comprehensive multi-phase research review
├── tests/                          # Root integration & integrity tests
│   └── test_phase1.py              # 10 automated test cases for Phase 1
└── Datasets/                       # Local raw & processed datasets (never committed to Git)
```

---

## 3. Research Track & Phase Roles

The research methodology follows strict role separation across datasets and experiments:

### Phase 1: Canonical Baseline Model (E1 LightGBM)
- **Dataset:** IEEE-CIS Fraud Detection (`train_transaction.csv` joined with `train_identity.csv`).
- **Algorithm:** LightGBM gradient boosted decision trees.
- **Features:** 406 features (numerical + categorical frequency/target encoded).
- **Split Protocol:** Temporal split (first 80% train, next 10% validation, final 10% out-of-time test).
- **Performance:** PR-AUC = 0.5267, ROC-AUC = 0.8981, F1 = 0.5308 (at optimal threshold = 0.616521).
- **Role:** The **authoritative canonical model** for tabular fraud classification. Teammates must use this frozen model artifact for inference scoring.

### Phase 1.5: Hardening & Robustness Analysis
- **Scope:** Bootstrapped confidence intervals (2,000 resamples), Brier score calibration (0.032), label leakage audit, and zero-day feature provenance verification.

### Phase 2: Temporal Deep Learning (E2 GRU)
- **Dataset:** IEEE-CIS with per-card sequence generation (historical window = 4 previous transactions).
- **Architecture:** PyTorch 2-layer GRU (hidden size 64) with classification head.
- **Performance:** PR-AUC = 0.1683, ROC-AUC = 0.7422, F1 = 0.2432.
- **Finding:** GRU alone underperforms tabular LightGBM due to sparsity and non-uniform transaction intervals.
- **Artifact:** Frozen checkpoint `phase2_gru/artifacts/E2b_scaled/model/gru_best.pt`, scaler `standard_scaler.pkl`.

### Phase 3: Tabular-Temporal Hybrids (E3A, E3B, E3C)
- **Architectures:** 
  - E3A: Feature concatenation (tabular + GRU hidden states).
  - E3B: GRU history with tabular skip connection.
  - E3C: Probability ensemble.
- **Finding:** Every hybrid underperformed standalone LightGBM (E3A PR-AUC = 0.4271, E3B = 0.1654, E3C = 0.2974). Standalone E1 remains the champion model.

### Phase 4: Concept Drift Adaptation (BAF Base Dataset)
- **Dataset:** Bank Account Fraud (BAF) Suite (1M records across 8 months).
- **Protocol:** Months 0–3 training, Month 4 validation/threshold locking, Months 5–6 Population Stability Index (PSI) drift monitoring, Month 7 out-of-time adaptation test.
- **Result:** PSI triggered drift alerts on 6 features at Month 5 and Month 6. Drift-triggered retraining improved Month 7 PR-AUC from 0.0890 to 0.1256 (bootstrap 95% CI strictly excludes zero).
- **Role:** Pure concept drift and retraining frequency research. Not for real-time serving.

---

## 4. Existing Streaming & Distributed Infrastructure Status

| Component | Status in `Upto_Phase-4` | Notes |
| :--- | :--- | :--- |
| **Kafka Code** | Absent | No Kafka producers, consumers, or configs exist in the `Upto_Phase-4` working tree. (Historical PR branches exist on remote `origin`). |
| **Spark Code** | Absent | No PySpark scripts, configs, or batch jobs exist in `Upto_Phase-4`. |
| **Flink Code** | Absent | No PyFlink scripts, jars, or streaming jobs exist in `Upto_Phase-4`. |
| **Integration Glue** | Absent | Currently no pipeline connecting streaming Kafka topics to model inference. |

---

## 5. Existing Test Suites & Baseline Health Verification

Before introducing any modifications, all test suites on `Upto_Phase-4` were executed and verified:

```text
================================================================================
TEST SUITE AUDIT EXECUTION RESULTS (Target Repository Baseline)
================================================================================
1. tests/test_phase1.py                              : 10 / 10 PASSED (100%)
2. phase2_gru/tests/ (Architecture & Preprocessing)  : 80 / 80 PASSED (100%)
3. experiments/phase4_drift_adaptation/tests/        : 52 / 52 PASSED (100%)
4. experiments/E3_hybrid/tests/ (Leakage & Hashes)   : 20 / 20 PASSED (100%)
--------------------------------------------------------------------------------
TOTAL AUTOMATED RESEARCH INTEGRITY TESTS             : 162 / 162 PASSED (100%)
================================================================================
```

Specifically verified test assertions:
- `test_T18_e1_predictions_hash`: Validates SHA-256 hash of `predictions.parquet`.
- `test_T19_e2b_checkpoint_hash`: Validates SHA-256 hash of `gru_best.pt`.
- `test_T20_e2b_scaler_hash`: Validates SHA-256 hash of `standard_scaler.pkl`.
- `phase4_drift_adaptation`: All 52 assertions on feature isolation, threshold locking, and temporal boundaries passed with a generated `INTEGRITY_CERTIFICATE.json`.

---

## 6. Dependency & Environment Analysis

### Python Dependencies (`requirements.txt`)
```text
pandas>=2.0.0
numpy>=1.21.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
pyarrow>=12.0.0
shap>=0.44.0
scipy>=1.10.0
joblib>=1.3.0
pyyaml>=6.0
matplotlib>=3.7.0
seaborn>=0.12.0
```
- **Python Version:** Active environment runs Python 3.13.13 on Windows 64-bit.
- **Deep Learning:** PyTorch 2.13.0.dev with CUDA 13.2 support is installed in the active Conda environment.
- **Distributed Computing Packages (PySpark, Apache-Flink, Kafka-Python):** Currently absent from root `requirements.txt` to keep the research baseline lightweight and isolated.

---

## 7. Configuration & Dataset References

### Configuration (`configs/phase1_lightgbm.yaml`)
- Specifies data paths, random seed (42), temporal split fractions (0.80, 0.10, 0.10), LightGBM objective (`binary`), evaluation metric (`average_precision`), and learning rate (0.03).

### Dataset Storage & Integrity Rules
- The physical directory `Datasets/` contains:
  - `BAF/` (Base.csv, Variant I-V)
  - `Credit card Fraud detection/credit_card_fraud_10k.csv`
  - `IEEE CIS-20260829T103704Z-1-001/IEEE CIS/` (train_transaction.csv, train_identity.csv)
  - `Paysim/paysim dataset.csv`
  - `ulb_creditcard/creditcard.csv`
  - `processed/ieee_cis/` (train.parquet, validation.parquet, test.parquet)
- All datasets are excluded via `.gitignore` and **must never be committed to Git**.

---

## 8. Target Repository Audit Conclusions

1. **Research Baseline is Completely Healthy:** All 162 research tests pass cleanly with zero regression.
2. **Clean Separation of Concerns:** Research code resides neatly under `src/`, `experiments/`, `phase2_gru/`, and `reports/`.
3. **Greenfield Real-Time Infrastructure:** Since no Kafka, Spark, or Flink code currently exists in `Upto_Phase-4`, the incoming modules from Member 1 and Member 2 will occupy dedicated top-level namespaces (`kafka/`, `spark/`, `flink/`) without colliding with research files.
4. **Safety Guarantees Established:** Research models, checkpoints, scalers, test predictions, and evaluation metrics are inventoried and locked as immutable.
