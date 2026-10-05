# Adaptive Financial Fraud Detection
### All Research Phases Complete — Git Checkpoint `7f6c99a`

> Research-grade adaptive fraud detection system: LightGBM baseline → GRU temporal → Hybrid → PSI-triggered drift adaptation.  
> **Teammates: Read Section "For Teammates" before doing anything.**

---

## ⚡ For Teammates — Read This First

### Which model should you use?

| Model | Dataset | PR-AUC | Use for serving? |
|-------|---------|--------|-----------------|
| **E1 LightGBM** | IEEE-CIS | **0.5267** | ✅ **YES — use this** |
| E2 GRU | IEEE-CIS | 0.1683 | ❌ No — much lower accuracy |
| E3A Hybrid | IEEE-CIS | 0.4271 | ❌ No — worse than E1 |
| E3B Hybrid | IEEE-CIS | 0.1654 | ❌ No — similar to E2 |
| E3C Hybrid | IEEE-CIS | 0.2974 | ❌ No — worse than E1 |
| Phase 4 Adaptive | **BAF** (different dataset) | 0.2010* | ❌ No — different dataset, incompatible |

> **\*Phase 4 PR-AUC is NOT comparable to E1.** They use completely different datasets (BAF vs IEEE-CIS).  
> A Phase 4 model given IEEE-CIS transactions will crash or give garbage output.  
> Phase 4 is a research experiment proving adaptive retraining works — it is not a serving replacement for E1.

### Rule: Use E1 only. Do not attempt to swap in E2, E3, or Phase 4 models.

---

### What each teammate needs from this repo

| Member | What you need | Where it is |
|--------|--------------|-------------|
| **Member 1 (Kafka)** | Transaction field schema | See "Transaction Schema" section below |
| **Member 2 (Spark)** | IEEE-CIS dataset field structure | `Datasets/IEEE CIS/` locally (not committed) |
| **Member 3 (Serving)** | `model.txt`, `preprocessing.joblib`, `feature_names.json` | `experiments/E1_lightgbm/` |

> Member 1 and Member 2 do **not** need any model file.  
> Only Member 3 directly loads the model.

---

### Transaction Schema (for Member 1 — Kafka)

The Kafka transaction generator should produce messages matching the IEEE-CIS transaction fields.
Key fields used by the model:

```json
{
  "TransactionID": 2987004,
  "TransactionDT": 86400,
  "TransactionAmt": 68.5,
  "ProductCD": "W",
  "card1": 13926,
  "card2": 391.0,
  "card3": 150.0,
  "card4": "discover",
  "card5": 142.0,
  "card6": "credit",
  "addr1": 315.0,
  "addr2": 87.0,
  "P_emaildomain": "gmail.com",
  "R_emaildomain": null,
  "C1": 1.0,
  "dist1": null,
  "M1": "T",
  "V1": 1.0
  // ... (406 features total after preprocessing)
}
```

Full feature list: `experiments/E1_lightgbm/feature_names.json` (406 features)  
Feature definitions: `src/features/ieee_cis_features.py`

---

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/Jayasimha-2005/adaptive-upi-fraud-detection.git
cd adaptive-upi-fraud-detection
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Datasets required

> **Raw datasets are NOT in this repo** — too large and subject to Kaggle Terms of Service.

| Phase | Dataset needed | Where to get it | Place it here |
|-------|---------------|-----------------|---------------|
| E1 (LightGBM) | IEEE-CIS | [Kaggle IEEE-CIS](https://www.kaggle.com/competitions/ieee-fraud-detection/data) | `Datasets/IEEE CIS/` |
| E2 (GRU) | IEEE-CIS | Same as above | `Datasets/IEEE CIS/` |
| E3 (Hybrid) | IEEE-CIS | Same as above | `Datasets/IEEE CIS/` |
| Phase 4 (Drift Adaptation) | BAF Base | [GitHub feedzai/bank-account-fraud](https://github.com/feedzai/bank-account-fraud) | `Datasets/BAF/` |

```
Datasets/
├── IEEE CIS/
│   ├── train_transaction.csv    ← required for E1, E2, E3
│   └── train_identity.csv       ← required for E1, E2, E3
└── BAF/
    └── Base.csv                 ← required for Phase 4 only
```

> **If you only want to use the trained model (not retrain), skip the datasets.**  
> All trained artifacts are already committed. See "Using the Trained Model" below.

---

## Using the Trained E1 Model (No Retraining Needed)

All E1 artifacts are in `experiments/E1_lightgbm/`:

```python
import lightgbm as lgb
import joblib
import json

# Load model
booster = lgb.Booster(model_file='experiments/E1_lightgbm/model.txt')

# Load preprocessor
preprocessor = joblib.load('experiments/E1_lightgbm/preprocessing.joblib')

# Load feature names
with open('experiments/E1_lightgbm/feature_names.json') as f:
    feature_names = json.load(f)  # 406 features

# Frozen threshold (F1-max on validation set)
THRESHOLD = 0.616521

# Predict on a preprocessed DataFrame
probs = booster.predict(X_processed)  # X_processed must have 406 features in correct order
decisions = (probs >= THRESHOLD).astype(int)  # 1 = FRAUD, 0 = LEGIT
```

---

## Complete Research Results — All Phases

### E1 — LightGBM Tabular Baseline 🔒 FROZEN
- **Dataset:** IEEE-CIS (Kaggle)
- **Status:** Frozen, committed, used by Member 3 for serving

```
PR-AUC  (test)      = 0.5317   [95% CI: 0.5126 – 0.5504]
ROC-AUC (test)      = 0.8990   [95% CI: 0.8924 – 0.9055]
F1                  = 0.5340
Threshold (frozen)  = 0.616521
Recall @ 1% FPR     = 0.4618
Precision @ Top-100 = 0.98
Brier Score         = 0.0323
```

---

### E2 — GRU Temporal Model 🔒 FROZEN
- **Dataset:** IEEE-CIS (same as E1)
- **Status:** Research experiment — proved GRU alone is weaker than LightGBM on this dataset

```
PR-AUC  (test)  = 0.1683   ← significantly below E1
ROC-AUC (test)  = 0.7422
Brier Score     = 0.2377   ← poorly calibrated
```

> ⚠️ **Do not use E2 for serving.** PR-AUC is 0.168 vs E1's 0.527. E2 is a research artifact only.

---

### E3 — Hybrid Model (LightGBM + GRU) 🔒 FROZEN
- **Dataset:** IEEE-CIS (same as E1)
- **Status:** Research experiment — proved combining E1 + E2 does not beat E1 alone

```
E3A:  PR-AUC = 0.4271  ← best hybrid, still below E1's 0.527
E3B:  PR-AUC = 0.1654  ← similar to E2 alone
E3C:  PR-AUC = 0.2974  ← between E2 and E1
```

> ⚠️ **Do not use E3 for serving.** All variants perform worse than E1.  
> Research finding: GRU component dragged the hybrid down. LightGBM alone is stronger on IEEE-CIS.

---

### Phase 4 — Concept Drift Adaptation 🔒 FROZEN
- **Dataset:** BAF Base (Bank Account Fraud — completely different from IEEE-CIS)
- **Status:** Research experiment — proved PSI-triggered adaptive retraining beats static model

```
IMPORTANT: Phase 4 uses BAF dataset. It is NOT compatible with IEEE-CIS preprocessing.
Do NOT give Phase 4 models to teammates for serving with IEEE-CIS data.
```

**Experimental design:**
```
Months 0–3  →  Train Static v1 (frozen)
Month 4     →  Validation (threshold selection → 0.8964, frozen)
Month 5     →  PSI check: 17/29 features drifted → Retrain → v2
Month 6     →  PSI check: 18/29 features drifted → Retrain → v3
Month 7     →  HELD-OUT evaluation (never seen during training)
```

**Month 7 final results:**

```
Static v1  PR-AUC  = 0.1609
Adaptive v3 PR-AUC = 0.2010
Delta              = +0.0402
95% paired bootstrap CI = [+0.0268, +0.0538]  (CI excludes zero)

84/84 preflight checks passed
52/52 integrity tests passed
```

> **Scientific finding:** A model that detects data drift (PSI) and retrains itself achieves meaningfully higher PR-AUC on unseen future data than a static frozen model. No p-value is claimed — the bootstrap CI excluding zero is the correct statistical statement.

---

## Project Structure

```
adaptive-upi-fraud-detection/
│
├── src/                              # Core source modules
│   ├── data/                         # validate.py, load.py, split.py
│   ├── features/                     # ieee_cis_features.py (preprocessor)
│   ├── models/                       # lightgbm_baseline.py
│   ├── evaluation/                   # metrics.py, bootstrap.py, calibration.py
│   └── audit/                        # feature_provenance.py
│
├── experiments/
│   ├── E1_lightgbm/                  # 🔒 E1 frozen artifacts
│   │   ├── model.txt                 # LightGBM model
│   │   ├── preprocessing.joblib      # Fitted preprocessor (406 features)
│   │   ├── feature_names.json        # All 406 feature names
│   │   ├── predictions.parquet       # Val + test predictions
│   │   ├── metrics.json              # All metrics + bootstrap CIs
│   │   └── split_meta.json           # Temporal split info
│   │
│   └── phase4_drift_adaptation/      # 🔒 Phase 4 frozen artifacts
│       ├── src/                      # pipeline.py, psi.py, model.py, ...
│       ├── artifacts/
│       │   ├── models/               # lgbm_v1.txt, lgbm_v2.txt, lgbm_v3.txt
│       │   ├── drift/                # psi_month5.json, psi_month6.json
│       │   ├── metrics/              # comparison + bootstrap results
│       │   ├── thresholds/           # frozen_threshold_v1.json
│       │   └── predictions/          # static + adaptive .npy files
│       ├── docs/                     # Protocol, design decisions, audit reports
│       ├── figures/                  # 13 research figures
│       └── tests/                   # preflight_audit.py, integrity_tests.py
│
├── phase2_gru/                       # 🔒 E2 frozen (research only)
│
├── Datasets/                         # NOT committed — download separately
│   ├── IEEE CIS/                     # For E1, E2, E3
│   └── BAF/                          # For Phase 4 only
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Key Numbers — Quick Reference

| Item | Value |
|------|-------|
| **E1 threshold (frozen)** | **0.616521** |
| E1 feature count | 406 |
| E1 PR-AUC (test) | 0.5317 |
| E1 ROC-AUC (test) | 0.8990 |
| Phase 4 threshold (frozen) | 0.8964 |
| Phase 4 Static M7 PR-AUC | 0.1609 |
| Phase 4 Adaptive M7 PR-AUC | 0.2010 |
| Phase 4 delta | +0.0402 |
| Phase 4 95% CI | [+0.0268, +0.0538] |
| Phase 4 v1 training rows | 547,975 |
| Git checkpoint | 7f6c99a |

---

## Git History

```
7f6c99a  Add Phase 4 — Concept Drift Adaptation (Protocol v1.1, BAF Base)
04a7203  Sprint 2 complete: Phase 2 GRU temporal modeling + Phase 2 freeze
b9a7d7d  Add README.md with setup guide, project structure, and teammate quickstart
```

---

## Team Architecture

```
YOUR REPO (this repo)
  E1 / E2 / E3 / Phase 4
  Model artifacts
         │
    GitHub push
         │
   ┌─────┼─────┐
   ▼     ▼     ▼
Member1 Member2 Member3
Kafka   Spark   FastAPI
                Docker
```

**This repo is the ML source of truth. Members 1–3 do NOT train models — they consume the frozen artifacts from this repo.**

---

## 🚀 Integrated Real-Time Infrastructure Track (Member 1 + Member 2)

The repository integrates the real-time event streaming and distributed data engineering infrastructure developed by **Member 1 (Apache Kafka)** and **Member 2 (Apache Spark & Apache Flink)** alongside the canonical research track:

```text
                                TRANSACTION REPLAY / SIMULATION
                                 (IEEE-CIS / PaySim / Synthetic)
                                               │
                                               ▼
                              ┌─────────────────────────────────┐
                              │     Kafka Producer Ingestion    │
                              │  - Key: card_id / user_id       │
                              │  - acks=all, idempotence=True   │
                              └────────────────┬────────────────┘
                                               │
                                               ▼
                              ┌─────────────────────────────────┐
                              │     Kafka: fraud-transactions   │
                              │     (6 Partitions, Murmur2 key) │
                              └────────┬───────────────┬────────┘
                                       │               │
                      ┌────────────────┴───┐       ┌───┴────────────────┐
                      │                    │       │                    │
                      ▼                    │       │                    ▼
    ┌───────────────────────────────────┐  │       │  ┌───────────────────────────────────┐
    │  Apache Flink 2.2 (Real-Time CEP) │  │       │  │ Apache Spark 3.5.9 (Batch/Stream) │
    ├───────────────────────────────────┤  │       │  ├───────────────────────────────────┤
    │ • key_by(user_id)                 │  │       │  │ • Columnar Parquet conversion     │
    │ • 5m & 10m Sliding Event Windows  │  │       │  │   (7.25x speedup, 927k rec/s)     │
    │ • Velocity Ratio Computation      │  │       │  │ • Historical feature engineering  │
    │ • Sub-second alert generation     │  │       │  │ • Micro-batch streaming connector │
    └─────────────────┬─────────────────┘  │       │  └─────────────────┬─────────────────┘
                      │                    │       │                    │
                      ▼                    │       │                    ▼
    ┌───────────────────────────────────┐  │       │  ┌───────────────────────────────────┐
    │      Kafka: fraud-features        │  │       │  │        Parquet Data Lake          │
    │   (Real-time feature vectors)     │  │       │  │   (Offline batch feature store)   │
    └─────────────────┬─────────────────┘  │       │  └─────────────────┬─────────────────┘
                      │                    │       │                    │
                      └─────────────────┐  │  ┌────┘                    │
                                        │  │  │                         │
                                        ▼  ▼  ▼                         │
========================================================================│=================
                            MODEL SCORING & RESEARCH BOUNDARY           │
========================================================================│=================
                                                                        │
                                ┌───────────────────────────────────┐   │
                                │   Model Serving Inference Engine  │   │
                                │   (Phase 1 LightGBM E1 Booster)   │   │
                                │   - Input: Canonical 406 features │   │
                                │   - Optimal Threshold: 0.616521   │   │
                                └─────────────────┬─────────────────┘   │
                                                  │                     │
                                                  ▼                     ▼
                                ┌───────────────────────────────────┐ ┌───────────────────┐
                                │        Transaction Verdict        │ │  Phase 4 Monitor  │
                                │  • APPROVED (p < 0.616521)        │ │  • PSI Drift Check│
                                │  • BLOCKED  (p >= 0.616521)       │ │  • Drift Retrain  │
                                └───────────────────────────────────┘ └───────────────────┘
```

### Component Structure & Roles

| Directory | Lead Role | Subsystem / Role | Key Commands |
| :--- | :--- | :--- | :--- |
| `kafka/` | Member 1 | Event Ingestion, Producers, Partition Affinity, At-Least-Once Delivery | `python kafka/producer/ieee_cis_replay_producer.py --rate 500` |
| `spark/` | Member 2 | Columnar Parquet Optimization (7.25x), Historical Feature Store, Micro-Batch Streaming | `python spark/benchmarks/run_benchmarks.py` |
| `flink/` | Member 2 | Sub-Second CEP, 5m/10m Sliding Window Velocity Ratios, Event-Time Watermarking | `python flink/streaming/stream_processor.py` |
| `cluster/` | Member 1 | 3-Node KRaft Distributed Broker Configurations | `server-1.properties` to `server-3.properties` |
| `tests/integration/` | Combined | 14-Step End-to-End Multi-Process Integration Test Suite | `python tests/integration/run_full_e2e_integration_test.py` |

### How to Run the End-to-End Pipeline

1. **Run Integration Unit & Contract Tests:**
   ```powershell
   python -m unittest kafka/tests/test_spark_flink_integration.py
   pytest spark/tests/
   pytest flink/tests/
   ```

2. **Run Concurrent Multi-Process Streaming Pipeline (Producer + Spark + Flink):**
   ```powershell
   python run_kafka_spark_flink_pipeline.py --records 1000 --rate 500 --mode all
   ```

3. **Run 14-Step Full Integration Verification Suite:**
   ```powershell
   python tests/integration/run_full_e2e_integration_test.py
   ```

---

## What NOT to Do

| Action | Why not |
|--------|---------|
| Use E2 (GRU) for serving | PR-AUC = 0.168, badly calibrated |
| Use E3 (Hybrid) for serving | All variants worse than E1 |
| Use Phase 4 model with IEEE-CIS data | Different dataset — will crash or give wrong output |
| Modify any frozen artifact | All experiments are locked at `7f6c99a` |
| Commit raw datasets | Gitignored — too large, Kaggle ToS |
