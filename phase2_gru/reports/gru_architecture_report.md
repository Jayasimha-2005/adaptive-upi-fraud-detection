# Phase 2 — STEP 7 GRU Architecture Report

**Generated:** 2026-09-18  
**Status: STEP 7 PASS — Architecture implemented and verified**  
**No training performed. No E1 files modified.**

---

## 1. Research Question

> *"Does historical transaction behavior provide additional predictive value beyond the tabular LightGBM baseline (E1)?"*

E1 frozen baseline: **PR-AUC = 0.531731** (test set, 78,542 transactions)

---

## 2. Locked Architecture

```
Input: [batch, 4, 406]
         │
         ▼
GRU(input_size=406, hidden_size=64, num_layers=1, batch_first=True)
         │
         ▼ final hidden state h_T  [batch, 64]
         │
Dropout(p=0.2)
         │
         ▼
Linear(64 → 1)
         │
         ▼
logits  [batch, 1]   ← BCEWithLogitsLoss during training
         │
sigmoid(logits)       ← fraud probability during inference
```

**Note on Dropout placement:** PyTorch's GRU `dropout` parameter only applies between stacked layers (num_layers ≥ 2). Since `num_layers=1`, the locked 0.2 dropout is applied as a standalone `nn.Dropout` layer on the final hidden state, before the linear classifier. This is mathematically equivalent and standard practice for single-layer GRUs.

---

## 3. Input Specification

| Property | Value |
|----------|-------|
| Tensor shape | `[batch, 4, 406]` |
| Dtype | `torch.float32` |
| Sequence length | 4 (history: T1, T2, T3, T4) |
| Feature dimension | 406 |
| Features | 405 E1 retained + 1 causal gap |

**Excluded from input (verified):**

| Column | Excluded? |
|--------|-----------|
| `card1` | YES — entity key only |
| `isFraud` | YES — target label |
| `TransactionID` | YES — row identifier |
| `TransactionDT` | YES — ordering only |

---

## 4. Output Specification

| Property | Value |
|----------|-------|
| Training output | Logits `[batch, 1]` — raw, unbounded |
| Inference output | `sigmoid(logits)` → `[0, 1]` |
| Why no Sigmoid in model | BCEWithLogitsLoss applies log-sigmoid internally; numerical stability |

---

## 5. Loss Function

| Property | Value |
|----------|-------|
| Loss | `BCEWithLogitsLoss` |
| `pos_weight` | **27.47** |
| `pos_weight` derivation | `418,924 legit / 15,252 fraud` (pre-Option-A train counts) |
| Post-Option-A alternative | `383,915 / 14,397 ≈ 26.67` (sensitivity analysis option) |
| SMOTE | No |
| Random oversampling | No |

---

## 6. Parameter Count

| Layer | Parameters |
|-------|-----------|
| `gru.weight_ih_l0` | 406 × (3×64) = **77,952** |
| `gru.weight_hh_l0` | 64 × (3×64) = **12,288** |
| `gru.bias_ih_l0` | 3×64 = **192** |
| `gru.bias_hh_l0` | 3×64 = **192** |
| `classifier.weight` | 64×1 = **64** |
| `classifier.bias` | **1** |
| **Total** | **90,689** |

*GRU formula: 3 gates (reset, update, new) × (input + hidden + 2×bias)*  
*~90K parameters is intentionally small — this is a clean baseline experiment, not a capacity study.*

---

## 7. Device

| Property | Value |
|----------|-------|
| PyTorch version | 2.13.0.dev20260603+cu132 |
| CUDA available | Yes (cu132) |
| Default device | CPU (model designed to run on CPU; CUDA optional) |
| Architecture test device | CPU |

---

## 8. Reproducibility Configuration

| Property | Value |
|----------|-------|
| Python seed | 42 |
| NumPy seed | 42 |
| PyTorch seed | 42 |
| PYTHONHASHSEED | 42 |
| CUDA deterministic | `torch.backends.cudnn.deterministic = True` |
| CUDA benchmark | `False` |

---

## 9. Architecture Test Results — AT1–AT15

```
============================================================
STEP 7 — Architecture Tests AT1-AT15
============================================================
  [PASS]  AT1   input torch.Size([4, 4, 406]) accepted
  [PASS]  AT2   output shape torch.Size([4, 1])
  [PASS]  AT3   finite [-0.449, 0.204]
  [PASS]  AT4   logits in [-0.692, 0.563], no Sigmoid layer
  [PASS]  AT5   proba in [0.3896, 0.5509], predict_proba matches
  [PASS]  AT6   gru.input_size == 406
  [PASS]  AT7   hidden_size == 64
  [PASS]  AT8   num_layers == 1
  [PASS]  AT9   seq_len == 4 in config and model
  [PASS]  AT10  Linear(64 -> 1)
  [PASS]  AT11  BCEWithLogitsLoss = 5.8974
  [PASS]  AT12  pos_weight = 27.47
  [PASS]  AT13  seed=42 deterministic across Python/NumPy/PyTorch
  [PASS]  AT14  two seed=42 models identical; seed=99 differs
  [PASS]  AT15  reproducible + smoke: loss=6.5346, proba=[[0.371 0.415 0.45 0.607]]

RESULTS: 15 passed, 0 failed (total 15)
```

| AT# | Area | Result |
|-----|------|--------|
| AT1 | Input shape `[batch, 4, 406]` | PASS |
| AT2 | Output shape `[batch, 1]` | PASS |
| AT3 | Finite output (no NaN/Inf) | PASS |
| AT4 | Logits not probabilities; no Sigmoid in model | PASS |
| AT5 | `sigmoid(logits)` in `[0, 1]`; `predict_proba()` matches | PASS |
| AT6 | `gru.input_size == 406` | PASS |
| AT7 | `hidden_size == 64` | PASS |
| AT8 | `num_layers == 1` | PASS |
| AT9 | `seq_len == 4` in config and model | PASS |
| AT10 | Final layer is `Linear(64 → 1)` | PASS |
| AT11 | `BCEWithLogitsLoss` consumes logits correctly | PASS |
| AT12 | `pos_weight ≈ 27.47` | PASS |
| AT13 | `seed=42` deterministic (Python/NumPy/PyTorch) | PASS |
| AT14 | Two `seed=42` models have identical initial parameters | PASS |
| AT15 | Forward pass reproducible; end-to-end smoke test | PASS |
| **Total** | | **15/15 PASS** |

---

## 10. Smoke Test Results (AT15)

```
Input  : x.shape = [4, 4, 406]   (batch=4, seq=4, features=406)
Labels : [0, 1, 0, 0]            (1 fraud, 3 legit in batch)
Logits : [-0.449, -0.208, -0.449, 0.204]  (raw, unbounded)
Loss   : 6.5346                   (BCEWithLogitsLoss with pos_weight=27.47)
Proba  : [0.371, 0.415, 0.450, 0.607]     (sigmoid applied at inference only)
```

**Pipeline confirmed:** tensor → GRU → logits → BCEWithLogitsLoss → finite loss  
**No model fitting occurred.**

---

## 11. No Training Confirmation

| Action | Status |
|--------|--------|
| Model weights updated | NO — `torch.no_grad()` throughout |
| Optimizer created | NO |
| Data loaded from parquet | NO |
| Training loop executed | NO |
| Checkpoints saved | NO |
| Test set evaluated | NO |
| E2 predictions generated | NO |
| E1 vs E2 comparison | NO |

---

## 12. E1 Immutability Confirmation

```
git status --short
# Output: ?? phase2_gru/
```

Only `phase2_gru/` has new files. Zero modifications to E1 artifacts.

| Path | Modified? |
|------|-----------|
| `src/` | NO |
| `experiments/E1_lightgbm/` | NO |
| `experiments/E1_encoding_comparison/` | NO |
| `root tests/` | NO |
| `requirements.txt` | NO |
| `configs/` | NO |

---

## 13. Files Created This Step

```
phase2_gru/src/models/gru_model.py          ← FraudGRU class
phase2_gru/src/training/training_config.py  ← Frozen hyperparameters
phase2_gru/src/utils/reproducibility.py     ← Seed/device utilities
phase2_gru/tests/test_gru_architecture.py   ← 15 pytest-compatible tests
phase2_gru/reports/gru_architecture_report.md ← This file
```

---

## 14. STEP 7 Status

| Item | Status |
|------|--------|
| GRU model implemented | PASS |
| Training config (frozen) | PASS |
| Reproducibility utilities | PASS |
| All 15 architecture tests | **15/15 PASS** |
| Smoke test | PASS |
| E1 immutability | CONFIRMED |
| Training performed | NO |

## STEP 7: FULLY PASS

**Awaiting approval for STEP 8: GRU training.**

---

*Phase 2 STEP 7 — Completed 2026-09-18*  
*`phase2_gru/` only | Zero E1 artifacts modified*
