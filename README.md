# Adaptive Financial Fraud Detection
### Phase 1 + 1.5 Complete — Sprint 1

> Research-grade fraud detection system using LightGBM, bootstrap confidence intervals, operational metrics, and a locked Phase 2 GRU specification.

---

## Quick Start for Teammates

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

### 4. Get the datasets (required only if retraining from scratch)
Download from Kaggle — [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data):
- `train_transaction.csv` → place in `Datasets/IEEE CIS/`
- `train_identity.csv`   → place in `Datasets/IEEE CIS/`

> **If you only want to use the trained model (not retrain), skip this step.**  
> The model, preprocessor, and predictions are already in `experiments/E1_lightgbm/`.

---

## What's Already Done (Sprint 1)

| Phase | Status | Key output |
|-------|--------|-----------|
| Phase 1 — LightGBM baseline | ✅ Complete | PR-AUC = 0.5317 |
| Phase 1.5 — Research hardening | ✅ Complete | Bootstrap CI, Recall@FPR, Calibration |
| Phase 2 — GRU sequence model | 🔲 Not started | Spec locked |

---

## Using the Trained Model (No Retraining Needed)

All artifacts are in `experiments/E1_lightgbm/`:

```python
import lightgbm as lgb
import joblib
import pandas as pd

# Load model
booster = lgb.Booster(model_file='experiments/E1_lightgbm/model.txt')

# Load preprocessor
preprocessor = joblib.load('experiments/E1_lightgbm/preprocessing.joblib')

# Load existing predictions (val + test)
preds = pd.read_parquet('experiments/E1_lightgbm/predictions.parquet')
print(preds.head())
# Columns: TransactionID, TransactionDT, isFraud, fraud_probability, prediction, split

# Load all metrics (including bootstrap CIs)
import json
with open('experiments/E1_lightgbm/metrics.json') as f:
    metrics = json.load(f)
print("PR-AUC:", metrics['test']['pr_auc'])
print("Bootstrap CI:", metrics['phase15_bootstrap']['test']['pr_auc'])
```

---

## Running the Full Pipeline from Scratch

```bash
# Run Phase 1 (requires raw CSVs in Datasets/IEEE CIS/)
python experiments/run_phase1.py --config configs/phase1_lightgbm.yaml

# Run Phase 1.5 statistical extension (reads existing predictions, no retraining)
python experiments/run_phase15_extend.py --config configs/phase1_lightgbm.yaml

# Run feature provenance audit
python src/audit/feature_provenance.py

# Run encoding comparison (V1/V2/V3)
python src/audit/encoding_comparison.py

# Run external integrity tests
python tests/test_phase1.py
```

---

## Project Structure

```
adaptive-upi-fraud-detection/
│
├── src/                          # All source modules
│   ├── data/                     # validate.py, load.py, split.py
│   ├── features/                 # ieee_cis_features.py (preprocessor)
│   ├── models/                   # lightgbm_baseline.py
│   ├── evaluation/               # metrics.py, bootstrap.py, calibration.py
│   └── audit/                    # feature_provenance.py, encoding_comparison.py
│
├── experiments/
│   ├── run_phase1.py             # Phase 1 full pipeline
│   ├── run_phase15_extend.py     # Phase 1.5 statistical extension
│   └── E1_lightgbm/             # All trained artifacts
│       ├── model.txt             # LightGBM model (1000 trees)
│       ├── preprocessing.joblib  # Fitted preprocessor
│       ├── predictions.parquet   # Val + test predictions
│       ├── metrics.json          # All metrics + bootstrap CIs
│       ├── feature_names.json    # 406 feature names
│       ├── split_meta.json       # Temporal split boundaries
│       └── *.png                 # PR curve, ROC, SHAP, calibration plots
│
├── reports/
│   ├── sprint-1.md               # ← READ THIS FIRST — complete story
│   ├── phase1/                   # Phase 1 + 1.5 reports
│   │   ├── PHASE1_COMPLETION_REPORT.md
│   │   ├── operational_metrics_report.md
│   │   ├── feature_provenance_audit.md
│   │   ├── encoding_comparison.md
│   │   └── ...
│   └── phase2/
│       └── SEQUENCE_GENERATION_SPEC.md  # GRU design (LOCKED)
│
├── configs/
│   └── phase1_lightgbm.yaml      # All experiment parameters
│
├── tests/
│   └── test_phase1.py            # 10 integrity tests (all pass)
│
├── dataset_analysis/             # Dataset forensic outputs
├── requirements.txt
└── .gitignore
```

---

## Key Results (E1 LightGBM — Frozen Baseline)

```
PR-AUC  (test)  = 0.5317   [95% CI: 0.5126 – 0.5504]
ROC-AUC (test)  = 0.8990   [95% CI: 0.8924 – 0.9055]
Recall @ 1% FPR = 0.4618
Precision @ Top-1,000 = 0.8410
Brier Score     = 0.0323   (naive baseline: 0.0341)
Integrity tests = 24/24 PASSED
```

> ⚠️ **E1 is IMMUTABLE.** Do not modify the training data, preprocessor, or model.  
> All future experiments (Phase 2 GRU, etc.) compare against this frozen baseline.

---

## What to Read First

| Order | File | Why |
|-------|------|-----|
| 1 | `reports/sprint-1.md` | Complete story — problem, approach, results, next steps |
| 2 | `reports/phase1/PHASE1_COMPLETION_REPORT.md` | Full Phase 1 technical record |
| 3 | `reports/phase2/SEQUENCE_GENERATION_SPEC.md` | Read before writing any Phase 2 code |

---

## For Phase 2 (Next)

Before writing any GRU code, read the locked spec:
```
reports/phase2/SEQUENCE_GENERATION_SPEC.md
```

Then build:
1. `src/sequences/window_generator.py`
2. `tests/test_phase2_sequences.py` — 5 mandatory leakage tests must pass first
3. `src/models/gru_fraud.py`
4. `experiments/run_phase2_gru.py`

**Target to beat:** PR-AUC > 0.5504 (upper bound of E1's 95% CI)

---

## Dataset Note

Raw datasets are **not** in this repo (too large, Kaggle Terms of Service).

| Dataset | Source | Used for |
|---------|--------|---------|
| IEEE-CIS | [Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection) | Primary (Phase 1 + 2) |
| BAF | [GitHub](https://github.com/feedzai/bank-account-fraud) | Phase 3 drift experiments |
| PaySim | [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) | Phase 4 streaming demo |
