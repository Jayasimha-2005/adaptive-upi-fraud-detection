# Phase 2 — E2b Scaled GRU Training Report

**Generated:** 2026-09-18  
**Variant:** E2b — same locked GRU architecture and training configuration as E2a, with train-fitted StandardScaler preprocessing  
**Status: E2b COMPLETE — STOP. Awaiting STEP 9 approval.**  
**Device:** NVIDIA GeForce RTX 5050 Laptop GPU (CUDA 13.2)  
**Test set: NOT evaluated. E1 artifacts: NOT modified. E2a artifacts: NOT modified.**

---

## 1. Research Question

> *"Does historical transaction behavior provide additional predictive value beyond the tabular LightGBM baseline (E1)?"*

**E1 frozen baseline: PR-AUC = 0.531731** (test set, 78,542 transactions)  
**E1 validation PR-AUC: 0.584935** (for reference, not used in final evaluation)

---

## 2. E2a Result and Why It Is Preserved

E2a (unscaled GRU) achieved a best validation PR-AUC of **0.0345**, which is approximately equal to the validation fraud prevalence of 3.39%. This constitutes near-random performance on the validation set.

**E2a is preserved exactly and completely in:**
```
phase2_gru/artifacts/E2a_unscaled/
  model/gru_best.pt
  training_history.csv
  run_metadata.json
phase2_gru/reports/E2_training_report.md
```

E2a remains a valid documented experiment. Its result — near-random validation PR-AUC with unscaled inputs — is an empirical observation. The E2b variant was introduced to test whether train-fitted numerical standardization materially affects GRU learning on these sequences. Only E2b's results can confirm or disconfirm this hypothesis empirically.

---

## 3. Preprocessing Audit

The 406 E2 features (405 retained E1 + 1 log1p gap) were audited using a 500-entity smoke sample from the training sequences.

### Scale Distribution

| Group | Count | Examples |
|-------|-------|---------|
| **std > 100** (large scale) | **66** | `id_02` (std≈67K), `V203` (std≈2,345), `TransactionAmt` (std≈197) |
| std 1–100 (medium scale) | 86 | `V131`, `C1`, `addr1`, `D5` |
| std < 1 (small/ordinal/binary) | 254 | `card4`, `M1`–`M9`, `V1`–`V9` |
| **std ≈ 0** (near zero-variance) | **1** | `V1` (std=0.0 on full train; set to 1.0) |

### NaN / Inf

| Check | Result |
|-------|--------|
| Features with NaN | **0** |
| Features with Inf | **0** |

### Key Observations

- The 406 features span a range of scales from ~0.002 to ~77,694 (post-scaling: all become std≈1)
- 66 features have std > 100, the largest being `id_02` with std ≈ 77,694 on the full training set
- `TransactionAmt` (std ≈ 197) is a particularly well-known high-variance tabular feature
- The log1p gap feature (std ≈ 4.67) is already compressed but benefits from z-scoring
- 1 near-zero-variance feature handled by setting scale to 1.0 (feature centered to ~0)

> [!NOTE]
> **No features were excluded.** All 406 features receive StandardScaler treatment. Zero-variance features are safely centered to ~0 with scale=1.0 (no division-by-zero; sklearn standard behavior). Feature count and ordering are unchanged.

---

## 4. E2b Preprocessing Methodology

### Scaler: `SequenceStandardScaler`

```
Fit  : X_train.reshape(-1, 406)  →  compute mean[406], scale[406] per feature
       Uses all N_train × 4 = 1,593,248 observations
       Fitting uses TRAINING SEQUENCES ONLY
       
Apply: (X - mean) / scale  →  broadcasts over [N, 4, 406]
       Applied identically to each time step (T1, T2, T3, T4)
       Val and test transformed using frozen train statistics
```

### Train-Only Fitting (Critical)

| Dataset | Used to fit scaler? | Transformed? |
|---------|--------------------|----|
| Train (398,312) | **YES** | YES |
| Validation (75,727) | **NO** | YES — using frozen train stats |
| Test (76,520) | **NO** | NO — reserved for STEP 9 |

### Fitted Scaler Statistics (full training set)

| Statistic | Value |
|-----------|-------|
| Samples used for fitting | 1,593,248 (398,312 × 4) |
| Mean range | [−306.83, 134,420.22] |
| Scale range | [0.002, 77,694.42] |
| Features with scale > 100 | 66 |
| Features with scale 1–100 | 86 |
| Features with scale < 1 | 254 |
| Near-zero-variance (scale → 1.0) | 1 |

**Post-scaling NaN/Inf check: PASS** (train and validation)

**Scaler serialized:** `phase2_gru/artifacts/E2b_scaled/preprocessing/standard_scaler.pkl`

---

## 5. E2b Architecture (Unchanged from E2a)

```
FraudGRU(input_size=406, hidden_size=64, num_layers=1, dropout=0.2)
  GRU(406 → 64, batch_first=True)
  Dropout(p=0.2)
  Linear(64 → 1)   ← logits (no Sigmoid)
  sigmoid(logits)   ← fraud probabilities at inference
Total trainable parameters: 90,689
```

No architecture changes between E2a and E2b.

---

## 6. Training Configuration (Locked — identical to E2a)

| Parameter | Value |
|-----------|-------|
| Seed | 42 |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 128 |
| Max epochs | 30 |
| Early stopping patience | 5 |
| Monitor | Validation PR-AUC |
| Loss | BCEWithLogitsLoss |
| pos_weight | 27.47 |
| SMOTE | No |
| Oversampling | No |

**Only change from E2a: input X is scaled using train-fitted StandardScaler.**

---

## 7. E2b Epoch History

| Epoch | Train Loss | Val Loss | Val PR-AUC | Time (s) | Note |
|-------|-----------|----------|------------|---------|------|
| 1 | 1.1528 | 1.1315 | 0.1371 | 11.3 | |
| **2** | **1.1071** | **1.1311** | **0.1377** | **11.1** | **← BEST** |
| 3 | 1.0706 | 1.1396 | 0.1366 | 11.2 | |
| 4 | 1.0337 | 1.1641 | 0.1329 | 11.1 | |
| 5 | 0.9924 | 1.1989 | 0.1351 | 11.1 | |
| 6 | 0.9522 | 1.2198 | 0.1284 | 11.1 | |
| 7 | 0.9111 | 1.2563 | 0.1246 | 11.8 | Early stop |

**Early stopped at epoch 7 (no improvement for 5 epochs after epoch 2)**

### Convergence Pattern

- **Train loss** decreases consistently (1.15 → 0.91) — the model is learning from training data
- **Val loss** decreases slightly epoch 1→2, then increases steadily (1.13 → 1.26) — classic overfitting onset
- **Val PR-AUC** peaks at epoch 2, then degrades — consistent with val loss trend
- The gap between train and val loss grows after epoch 2, indicating the model begins to overfit the training sequences

---

## 8. Best Validation Results

| Metric | Value |
|--------|-------|
| Best epoch | **2** |
| Best validation PR-AUC | **0.1377** |
| Best validation loss | 1.1311 |
| Epochs completed | 7 / 30 |
| Early stopped | Yes |
| Training time | 78.7 sec (1.3 min on GPU) |
| Total script time | 21.7 min (sequence generation + training) |

---

## 9. E2a vs E2b Validation Comparison

| Variant | Preprocessing | Best Epoch | Best Val PR-AUC | Val Loss | Early Stop |
|---------|--------------|-----------|----------------|----------|-----------|
| **E2a** | None (unscaled) | 1 | **0.0345** | 1.3181 | Epoch 6 |
| **E2b** | StandardScaler (train-fitted) | 2 | **0.1377** | 1.1311 | Epoch 7 |
| **E1** *(reference)* | N/A (LightGBM) | N/A | **0.5849** | N/A | N/A |

### Observations

1. **E2b validation PR-AUC (0.1377) is substantially higher than E2a (0.0345).** Scaling materially affected GRU learning on the validation set.

2. **E2a validation PR-AUC (0.0345) was approximately equal to the random baseline (fraud prevalence ~0.0339).** E2b's 0.1377 is well above the random baseline, demonstrating that the GRU did learn discriminative patterns when inputs were numerically conditioned.

3. **E2b shows an early overfitting pattern.** Training loss decreases consistently while validation loss increases after epoch 2. This suggests the model begins memorizing training sequences rather than learning generalizable temporal patterns. The early stopping mechanism correctly identified epoch 2 as the best checkpoint.

4. **E2b validation PR-AUC (0.1377) remains substantially lower than E1 validation PR-AUC (0.5849).** This validates a preliminary observation that the tabular LightGBM baseline significantly outperforms the GRU on the validation set. Final conclusions must wait for test-set evaluation on the 76,520 common TransactionIDs in STEP 9.

> [!IMPORTANT]
> E2a vs E2b comparison is on validation only. Neither model's test set has been evaluated. The definitive E1 vs E2 comparison uses the frozen test set (76,520 common TransactionIDs) in STEP 9 only.

> [!NOTE]
> The empirical observation that E2b substantially outperforms E2a on validation is consistent with the hypothesis that unscaled inputs impaired GRU learning. However, this comparison does not isolate scaling as the sole factor — other aspects of the training dynamic (e.g., loss landscape, gradient flow) may also differ between scaled and unscaled inputs.

---

## 10. GPU Information

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 5050 Laptop GPU |
| VRAM | 8.55 GB |
| CUDA version | 13.2 |
| PyTorch version | 2.13.0.dev20260603+cu132 |
| Peak GPU memory | ~86 MB per batch (same as E2a) |
| Epoch time | ~11.1–11.8 sec/epoch |
| Training time | 78.7 sec (7 epochs) |

---

## 11. Training Duration

| Phase | Time |
|-------|------|
| Sequence generation | ~20.4 min |
| StandardScaler fit + transform | ~12 sec |
| Pre-flight check | < 1 sec |
| Training (7 epochs × ~11s) | 78.7 sec |
| **Total script time** | **21.7 min** |

---

## 12. Test Set

**NOT evaluated.** The test set was extracted from the sequence builder but was never scaled, loaded into a DataLoader, or used in any computation during this script. It remains reserved for STEP 9.

---

## 13. E1 Immutability

`git status --short` shows only `?? phase2_gru/`. Zero E1 files modified.

---

## 14. Artifacts Created This Step

```
phase2_gru/
├── artifacts/
│   ├── E2a_unscaled/                   ← E2a preserved (not modified)
│   │   ├── model/gru_best.pt
│   │   ├── training_history.csv
│   │   └── run_metadata.json
│   └── E2b_scaled/                     ← E2b new artifacts
│       ├── model/gru_best.pt            ← best checkpoint (epoch 2)
│       ├── preprocessing/
│       │   ├── standard_scaler.pkl      ← fitted scaler (train only)
│       │   └── scaler_stats.json        ← human-readable stats
│       ├── training_history.csv         ← 7-epoch history
│       └── run_metadata.json            ← full metadata
├── src/data/
│   └── preprocessing.py                ← SequenceStandardScaler
├── src/training/
│   └── run_training_e2b.py             ← E2b entry point
├── tests/
│   └── test_preprocessing.py           ← PT1–PT8 (22 tests)
└── reports/
    └── E2b_scaled_training_report.md   ← this file
```

---

## 15. Limitations and Methodological Notes

1. **Preprocessing variant only.** E2b changes numerical conditioning, not the model architecture, feature set, sequence construction, or training hyperparameters. It is not a separate model comparison — it is a standardized version of E2a.

2. **Early overfitting.** E2b's best epoch is 2 of 7, with clear val loss degradation thereafter. This suggests the current hidden_size=64 / dropout=0.2 / lr=1e-3 configuration may overfit on these 398K sequences. Regularization investigations are out of scope for the locked experiment.

3. **Validation PR-AUC is not the final result.** The definitive comparison against E1 uses the frozen test set (76,520 common TransactionIDs) in STEP 9 with paired bootstrap.

4. **The gap feature (log1p) is additionally z-scored in E2b.** It was already compressed by log1p in both E2a and E2b; z-scoring in E2b brings it to mean≈0, std≈1. This is a minor additional change vs E2a.

---

## 16. E2b Status

| Item | Status |
|------|--------|
| Preprocessing audit | COMPLETE |
| StandardScaler (train-fitted) | COMPLETE |
| Train-only fitting verified | PASS |
| PT1–PT8 preprocessing tests | **22/22 PASS** |
| E2b training completed | YES |
| Best val PR-AUC | **0.1377** |
| Early stopped | YES (epoch 7) |
| Checkpoint saved | YES (epoch 2) |
| Test evaluated | NO |
| E1 modified | NO |
| E2a modified | NO |

## E2b: COMPLETE — STOP. Awaiting STEP 9 approval.

---

*Phase 2 E2b — Completed 2026-09-18*  
*`phase2_gru/` only | Zero E1 or E2a artifacts modified*
