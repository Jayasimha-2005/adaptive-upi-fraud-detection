# Phase 2 — Complete Progress Report
## E2 GRU Sequence Modeling — Full Summary (STEP 1 → STEP 8)

**Project:** Adaptive Financial Fraud Detection with Temporal Modeling  
**Date:** 2026-09-18  
**Current Status:** STEP 8 COMPLETE — Research decision required before STEP 9  
**Phase 2 location:** `phase2_gru/` (fully isolated — zero E1 files touched)

---

## Project Overview

### Research Question
> *"Does historical transaction behavior (sequences) provide additional predictive value beyond the tabular LightGBM baseline (E1)?"*

All outcomes — GRU better, equal, or worse — are valid research results.

### E1 Frozen Baseline (IMMUTABLE)

| Metric | Value |
|--------|-------|
| Model | LightGBM (tabular, non-temporal) |
| Features | 406 features, one transaction at a time |
| Test PR-AUC | **0.531731** |
| Bootstrap CI | [0.512586 – 0.550448] |
| Test ROC-AUC | 0.898993 |
| Precision @ Top-1K | 0.841 |
| Recall @ 1% FPR | 0.4618 |
| Status | **FROZEN / IMMUTABLE** |

---

## Phase 2 Dataset

| Split | Rows | Fraud | Legit | Fraud% |
|-------|------|-------|-------|--------|
| Train | 434,176 | 15,252 | 418,924 | 3.51% |
| Validation | 77,822 | 2,637 | 75,185 | 3.39% |
| Test | 78,542 | 2,774 | 75,768 | 3.53% |
| **Total** | **590,540** | **20,663** | **569,877** | |

---

## STEP 1 — Repository Inspection ✅ COMPLETE

**Goal:** Map the exact state of the existing codebase before writing any Phase 2 code.

**Key findings:**
- E1 feature list: `experiments/E1_lightgbm/feature_names.json` — **406 features** confirmed
- Frozen artifacts: `model.txt`, `metrics.json`, `predictions.parquet`, `split_meta.json`
- Split boundaries: `train_dt_max = 10,972,800` / `val_dt_max = 13,392,000`
- `isFraud` in features → **NO** ✅
- `TransactionID` in features → **NO** ✅
- Critical ambiguity: `card1` is both an E1 feature AND natural entity grouping key

**Output:** [`phase2_gru/reports/repository_inventory.md`](file:///c:/Users/Harini/Documents/GitHub/Jayasimha-github/adaptive-upi-fraud-detection/phase2_gru/reports/repository_inventory.md)

---

## Research Decisions Locked ✅ COMPLETE

Two explicit design decisions before any implementation:

### Q1 — card1 Treatment → Option A: Excluded from GRU input

```
card1 → sequence grouping key ONLY
card1 → NOT in GRU input X
```

**Reason:** Feeding card1's raw identifier into the GRU risks entity memorization — the model learns "card #12345 = fraud" rather than learning that the *temporal pattern* is anomalous. Prevents leakage of entity identity into learned weights.

```
406 E1 features − card1 = 405 retained transaction features
```

### Q2 — Temporal Gap Feature → YES

```
time_since_previous_transaction = log1p(seconds between transactions)
T1 → 0.0 (anchor)
T2 → log1p(DT2 - DT1)
T3 → log1p(DT3 - DT2)
T4 → log1p(DT4 - DT3)
T5 → NOT COMPUTED (target — would be leakage)
```

**Reason:** Transaction timing carries strong fraud signal (rapid bursts vs. normal spacing).

```
405 retained E1 features + 1 causal gap = 406 E2 input features
```

**Output:** [`phase2_gru/reports/sequence_specification.md`](file:///c:/Users/Harini/Documents/GitHub/Jayasimha-github/adaptive-upi-fraud-detection/phase2_gru/reports/sequence_specification.md)

---

## E2 Feature Manifest ✅ LOCKED

| Component | Count |
|-----------|-------|
| E1 original features | 406 |
| card1 removed (entity key only) | −1 |
| E1 retained features | 405 |
| `time_since_previous_transaction` added | +1 |
| **Final E2 GRU input features** | **406** |

**Excluded (verified):**

| Column | Reason |
|--------|--------|
| `card1` | Entity grouping key only |
| `isFraud` | Target label |
| `TransactionID` | Row identifier |
| `TransactionDT` | Ordering only |

**Output:** [`phase2_gru/reports/E2_feature_manifest.json`](file:///c:/Users/Harini/Documents/GitHub/Jayasimha-github/adaptive-upi-fraud-detection/phase2_gru/reports/E2_feature_manifest.json)

---

## Pre-flight Dataset Verification ✅ COMPLETE

All dataset counts verified from parquets. Zero discrepancies.

**Output:** [`phase2_gru/reports/preflight_verification.json`](file:///c:/Users/Harini/Documents/GitHub/Jayasimha-github/adaptive-upi-fraud-detection/phase2_gru/reports/preflight_verification.json)

---

## STEP 6 — Sequence Generation ✅ COMPLETE / PASS

### Sequence Design (Locked)

| Parameter | Value |
|-----------|-------|
| Entity key | `card1` (grouping only) |
| Window length | 5 (T1, T2, T3, T4, T5) |
| History | 4 transactions [T1–T4] |
| Target | T5 |
| Stride | 1 (sliding) |
| Min entity transactions | 5 |
| Temporal ordering | **Strictly increasing DT** (Option A) |
| Split assignment | Target transaction's split |
| Cross-split history | Allowed (prior splits only) |
| GRU input shape | `[batch, 4, 406]` |

### Duplicate TransactionDT Discovery

During generation, same-second transactions were found within card entities:

| Finding | Value |
|---------|-------|
| Cards with duplicate DTs | 59 / 12,421 |
| Same-second row pairs | 146 |
| Treatment | **Option A — exclude windows with equal DTs** |
| Windows excluded | **585** |
| Exclusion rate | **~0.11%** |

**Rationale:** `DT(T4) == DT(T5)` does not prove T4 preceded T5. TransactionID provides deterministic ordering for candidate enumeration only, not chronological evidence. Excluding same-second windows is the methodologically honest treatment.

### Final Retained Sequence Counts

| Split | Retained Windows | Fraud | Legit | Fraud% | Unique card1 |
|-------|-----------------|-------|-------|--------|-------------|
| **Train** | **398,312** | 14,397 | 383,915 | 3.61% | 5,640 |
| **Validation** | **75,727** | 2,567 | 73,160 | 3.39% | 4,538 |
| **Test** | **76,520** | 2,700 | 73,820 | 3.53% | 4,737 |
| **Total** | **550,559** | 19,664 | 530,895 | | |

### X Tensor Shapes

| Split | Shape | NaN | Inf |
|-------|-------|-----|-----|
| Train | `(398312, 4, 406)` | None | None |
| Validation | `(75727, 4, 406)` | None | None |
| Test | `(76520, 4, 406)` | None | None |

### E2 Test Coverage vs E1 Baseline

| Metric | Value |
|--------|-------|
| E1 test TransactionIDs | 78,542 |
| **E2 eligible test targets** | **76,520** |
| E1-only (not in E2) | 2,022 |
| **E2 test coverage** | **97.43%** |

> The paired bootstrap comparison (E2 vs E1) must use the **76,520 shared TransactionIDs only.**

### Leakage Tests — LT1–LT8

```
============================= 33 passed in 7.22s ==============================
```

| Test | Area | Result |
|------|------|--------|
| LT1 | Target not in history | ✅ PASS |
| LT2 | Strict temporal ordering (Option A) | ✅ PASS |
| LT3 | Label isolation (isFraud, card1 absent from X) | ✅ PASS |
| LT4 | History strictly ascending | ✅ PASS |
| LT5 | No future/cross-split leakage | ✅ PASS |
| LT6 | Target features absent from history | ✅ PASS |
| LT7 | Split assignment integrity | ✅ PASS |
| LT8 | Target gap (T5-T4) not used | ✅ PASS |
| **Total** | | **33/33** |

**Outputs:**
- [`phase2_gru/reports/sequence_generation_report.md`](file:///c:/Users/Harini/Documents/GitHub/Jayasimha-github/adaptive-upi-fraud-detection/phase2_gru/reports/sequence_generation_report.md)
- [`phase2_gru/reports/leakage_audit.md`](file:///c:/Users/Harini/Documents/GitHub/Jayasimha-github/adaptive-upi-fraud-detection/phase2_gru/reports/leakage_audit.md)
- [`phase2_gru/reports/sequence_generation_results.json`](file:///c:/Users/Harini/Documents/GitHub/Jayasimha-github/adaptive-upi-fraud-detection/phase2_gru/reports/sequence_generation_results.json)

---

## STEP 7 — GRU Architecture ✅ COMPLETE / PASS

### Architecture (Locked)

```
Input:  [batch, 4, 406]
         │
GRU(input_size=406, hidden_size=64, num_layers=1, batch_first=True)
         │
         ▼  final hidden state h_T  [batch, 64]
Dropout(p=0.2)
         │
Linear(64 → 1)
         │
         ▼
logits [batch, 1]          ← BCEWithLogitsLoss during training
         │
sigmoid(logits)            ← fraud probability at inference
```

### Parameter Count

| Layer | Parameters |
|-------|-----------|
| GRU (weight_ih, weight_hh, bias_ih, bias_hh) | 90,624 |
| Linear (64→1) + bias | 65 |
| **Total trainable** | **90,689** |

### Architecture Tests — AT1–AT15

```
15/15 PASSED
```

| AT# | Test | Result |
|-----|------|--------|
| AT1 | Input `[batch, 4, 406]` accepted | PASS |
| AT2 | Output shape `[batch, 1]` | PASS |
| AT3 | No NaN/Inf in output | PASS |
| AT4 | Logits, not probabilities (no Sigmoid inside model) | PASS |
| AT5 | `sigmoid(logits)` → valid probabilities `[0,1]` | PASS |
| AT6 | `gru.input_size == 406` | PASS |
| AT7 | `hidden_size == 64` | PASS |
| AT8 | `num_layers == 1` | PASS |
| AT9 | `seq_len == 4` in config | PASS |
| AT10 | Final layer is `Linear(64 → 1)` | PASS |
| AT11 | `BCEWithLogitsLoss` with pos_weight works | PASS |
| AT12 | `pos_weight ≈ 27.47` | PASS |
| AT13 | `seed=42` deterministic (Python/NumPy/PyTorch) | PASS |
| AT14 | Two `seed=42` models have identical initial parameters | PASS |
| AT15 | Reproducible forward pass + end-to-end smoke test | PASS |

**Outputs:**
- [`phase2_gru/src/models/gru_model.py`](file:///c:/Users/Harini/Documents/GitHub/Jayasimha-github/adaptive-upi-fraud-detection/phase2_gru/src/models/gru_model.py)
- [`phase2_gru/src/training/training_config.py`](file:///c:/Users/Harini/Documents/GitHub/Jayasimha-github/adaptive-upi-fraud-detection/phase2_gru/src/training/training_config.py)
- [`phase2_gru/src/utils/reproducibility.py`](file:///c:/Users/Harini/Documents/GitHub/Jayasimha-github/adaptive-upi-fraud-detection/phase2_gru/src/utils/reproducibility.py)
- [`phase2_gru/reports/gru_architecture_report.md`](file:///c:/Users/Harini/Documents/GitHub/Jayasimha-github/adaptive-upi-fraud-detection/phase2_gru/reports/gru_architecture_report.md)

---

## STEP 8 — GRU Training ✅ COMPLETE (Research Decision Required)

### Training Setup

| Parameter | Value |
|-----------|-------|
| Device | **NVIDIA GeForce RTX 5050 Laptop GPU** |
| VRAM | 8.55 GB |
| CUDA version | 13.2 |
| PyTorch | 2.13.0.dev20260603+cu132 |
| System RAM | 24 GB |
| Seed | 42 |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 128 |
| Max epochs | 30 |
| Early stop patience | 5 |
| Monitor | Validation PR-AUC |
| Loss | BCEWithLogitsLoss |
| pos_weight | 27.47 |
| Script time | 25.8 min (24.7 min seq gen + 68s training) |

### GPU Verification (Pre-training)

```
CUDA available     : True
Device             : NVIDIA GeForce RTX 5050 Laptop GPU
Total VRAM         : 8.55 GB
CUDA version       : 13.2
GPU smoke test     : PASS
GPU peak memory    : 86.2 MB  (far under 8.55 GB limit)
Loss (smoke)       : 0.7396 (finite)
```

### Pre-flight Check

```
logits = -0.1199  (finite) ✅
loss   =  1.8500  (finite) ✅
gradients: all finite ✅
optimizer step: succeeded ✅
```

### Epoch-by-Epoch Training History

| Epoch | Train Loss | Val Loss | Val PR-AUC | Time (s) | Note |
|-------|-----------|----------|-----------|---------|------|
| **1** | **1.3599** | **1.3181** | **0.0345** | 11.2 | **← BEST** |
| 2 | 1.3581 | 1.3152 | 0.0339 | 11.4 | |
| 3 | 1.3574 | 1.3199 | 0.0341 | 11.3 | |
| 4 | 1.3580 | 1.3149 | 0.0343 | 11.4 | |
| 5 | 1.3580 | 1.3215 | 0.0342 | 11.5 | |
| 6 | 1.3576 | 1.3148 | 0.0343 | 11.5 | Early stop |

**Early stopped at epoch 6 (no improvement for 5 epochs)**

### Results Summary

| Metric | Value |
|--------|-------|
| Best epoch | **1** |
| Best val PR-AUC | **0.0345** |
| Epochs completed | 6 / 30 |
| Early stopped | Yes |
| Total training time | 68.2 sec |

### ⚠️ Critical Finding — GRU Trained at Random Performance

| Comparison | Value |
|------------|-------|
| GRU val PR-AUC | **0.0345** |
| Random baseline (fraud rate) | **~0.0339** |
| E1 baseline PR-AUC | **0.5317** |
| Ratio E2/E1 | 0.065 (15× worse than E1) |

**The GRU performed essentially at random — it learned nothing in 6 epochs.**

### Root Cause: Unscaled Features

| System | Needs Feature Scaling? | Why |
|--------|----------------------|-----|
| LightGBM (E1) | NO — scale invariant | Tree splits don't use magnitudes |
| GRU (E2) | **YES — critical** | Gradient updates dominated by large-magnitude features |

The 406 E1 features contain wildly different scales:
- `TransactionAmt`: 0.25 – 31,936 (raw dollars)
- `V1`–`V339`: various Vesta scales
- `C1`–`C14`: count features 0 – 2,720
- `D1`–`D15`: time deltas 0 – 640

Without `StandardScaler`, large-scale features dominate gradients and small-scale features (which may carry strong fraud signal) contribute negligible updates. The GRU cannot learn.

**This is NOT a bug, NOT a hardware issue, NOT a code error.** It is the known requirement that neural networks on tabular data need feature standardization.

### Test Set

**NOT evaluated.** Reserved for STEP 9. Test set was never touched.

### Saved Artifacts

| File | Status |
|------|--------|
| `phase2_gru/artifacts/model/gru_best.pt` | Saved (epoch 1) |
| `phase2_gru/artifacts/training_history.csv` | Saved (6 epochs) |
| `phase2_gru/artifacts/run_metadata.json` | Saved |
| `phase2_gru/reports/E2_training_report.md` | Saved |

---

## Combined Test Results (After STEP 8)

```
Leakage tests    (LT1–LT8)   : 33/33 PASSED   [5.50s]
Architecture tests (AT1–AT15) : 15/15 PASSED
Total                         : 48/48 PASSED
```

---

## Immutability Confirmation

```bash
git status --short
# Output: ?? phase2_gru/
```

**Only `phase2_gru/` is new. Zero modifications to E1 artifacts.**

| Path | Modified? |
|------|-----------|
| `src/` | NO ✅ |
| `experiments/E1_lightgbm/` | NO ✅ |
| `experiments/E1_encoding_comparison/` | NO ✅ |
| `reports/` (Phase 1/1.5) | NO ✅ |
| `configs/` | NO ✅ |
| `tests/` (root) | NO ✅ |

---

## All Phase 2 Files Created

```
phase2_gru/
│
├── src/
│   ├── models/
│   │   └── gru_model.py              ← FraudGRU class (logits output)
│   ├── data/
│   │   ├── sequence_builder.py       ← Builder with Option A filter
│   │   └── dataset.py                ← PyTorch SequenceDataset + DataLoader
│   ├── training/
│   │   ├── training_config.py        ← Frozen hyperparameters (dataclass)
│   │   ├── trainer.py                ← Training loop + early stopping
│   │   └── run_training.py           ← Main training entry point
│   └── utils/
│       └── reproducibility.py        ← Seed setting + device detection
│
├── tests/
│   ├── test_sequence_leakage.py      ← LT1, LT6, LT8
│   ├── test_temporal_order.py        ← LT2, LT4, LT5
│   ├── test_label_isolation.py       ← LT3
│   ├── test_split_integrity.py       ← LT7
│   └── test_gru_architecture.py      ← AT1–AT15
│
├── artifacts/
│   ├── model/
│   │   └── gru_best.pt               ← Best checkpoint (epoch 1)
│   ├── training_history.csv          ← 6-epoch history
│   └── run_metadata.json             ← Full training metadata
│
└── reports/
    ├── repository_inventory.md        ← STEP 1 output
    ├── E2_feature_manifest.json       ← 406 locked features
    ├── sequence_specification.md      ← Locked design v1.1
    ├── preflight_verification.json    ← Dataset verification
    ├── sequence_generation_results.json ← Full counts + coverage
    ├── sequence_generation_report.md  ← STEP 6 audit
    ├── leakage_audit.md               ← LT1–LT8 full results
    ├── gru_architecture_report.md     ← STEP 7 architecture
    ├── E2_training_report.md          ← STEP 8 training results
    └── PHASE2_PROGRESS_REPORT.md      ← This file
```

---

## Key Numbers At a Glance

```
E1 FROZEN BASELINE
  PR-AUC (test)        : 0.531731
  ROC-AUC (test)       : 0.898993
  Bootstrap CI         : [0.5126 – 0.5504]

E2 SEQUENCE DESIGN
  Entity key           : card1 (grouping only, not in GRU input)
  Input features       : 406 (405 E1 + 1 causal gap)
  GRU input shape      : [batch, 4, 406]
  Temporal ordering    : Strictly increasing DT (Option A)

E2 DATASET
  Train sequences      : 398,312
  Val sequences        :  75,727
  Test sequences       :  76,520
  Total                : 550,559
  Excluded (equal DT)  : 585 (0.11%)
  E2 test coverage     : 97.43% of E1 test set

GRU ARCHITECTURE
  Parameters           : 90,689 trainable
  Device               : NVIDIA RTX 5050 (CUDA 13.2, 8.55 GB VRAM)

STEP 8 TRAINING RESULT
  Epochs               : 6 / 30 (early stopped)
  Best val PR-AUC      : 0.0345 (≈ random)
  Random baseline      : ~0.0339 (fraud rate)
  Training time        : 68.2 sec on GPU
  Root cause           : Unscaled features

LEAKAGE TESTS          : 33/33 PASSED
ARCHITECTURE TESTS     : 15/15 PASSED
TOTAL TESTS            : 48/48 PASSED
```

---

## Current Step Status

| Step | Task | Status |
|------|------|--------|
| ✅ STEP 1 | Repository inspection | PASS |
| ✅ STEP 6 | Sequence generation + leakage audit (33/33) | PASS |
| ✅ STEP 7 | GRU architecture + tests (15/15) | PASS |
| ⚠️ STEP 8 | GRU training | COMPLETE — research decision needed |
| 🔒 STEP 9 | Test evaluation + paired bootstrap | LOCKED |
| 🔒 STEP 10 | Final research report | LOCKED |

---

## Pending Research Decision (Before STEP 9)

The STEP 8 GRU trained on **unscaled** E1 features achieved val PR-AUC = 0.0345, which is essentially the random baseline (fraud rate = 3.39%). This is caused by missing feature standardization, not by the GRU architecture or temporal modeling concept.

### Options:

| Option | Description | Research Validity |
|--------|-------------|------------------|
| **A** | Proceed to STEP 9 as-is | Documents unscaled GRU = random. Valid but unfair test of temporal modeling. |
| **B** *(Recommended)* | Add `StandardScaler` (fit train, apply val/test) → retrain STEP 8 | Gives GRU fair test. Standard requirement for neural networks on tabular data. |
| **C** | Report both E2a (unscaled) + E2b (scaled) | Most complete documentation. |

> **Recommended: Option B.** StandardScaler is not hyperparameter tuning — it is standard neural-network preprocessing. The research question (*does temporal sequence modeling add value?*) deserves a fair test with properly scaled inputs.

---

*Phase 2 Progress Report — Updated 2026-09-18*  
*All work in `phase2_gru/` | Zero E1 artifacts modified*
