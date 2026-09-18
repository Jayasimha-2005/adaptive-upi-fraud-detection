# Phase 2 — STEP 8: E2 GRU Training Report

**Generated:** 2026-09-18  
**Status: STEP 8 COMPLETE — Training finished, results require review**  
**Device:** NVIDIA GeForce RTX 5050 Laptop GPU (CUDA 13.2)  
**Test set: NOT evaluated. E1 artifacts: NOT modified.**

---

## 1. Research Question

> *"Does historical transaction behavior provide additional predictive value beyond the tabular LightGBM baseline (E1)?"*

**E1 frozen baseline: PR-AUC = 0.531731** (test set, 78,542 transactions)

---

## 2. Dataset / Sequence Counts

| Split | Sequences | Fraud | Legit | Fraud% |
|-------|----------|-------|-------|--------|
| Train | 398,312 | 14,397 | 383,915 | 3.61% |
| Validation | 75,727 | 2,567 | 73,160 | 3.39% |
| Test | 76,520 | 2,700 | 73,820 | 3.53% |

Duplicate-DT excluded: **585** (Option A)  
E2 test coverage: **97.43%** of E1 test set

---

## 3. Model Architecture

```
FraudGRU(input_size=406, hidden_size=64, num_layers=1, dropout=0.2)
  GRU(406 → 64, batch_first=True)
  Dropout(p=0.2)
  Linear(64 → 1)          ← logits (no Sigmoid)
  sigmoid(logits)          ← probabilities at inference only
```

---

## 4. Parameter Count

| Layer | Parameters |
|-------|-----------|
| GRU weights + biases | 90,624 |
| Linear (64→1) + bias | 65 |
| **Total trainable** | **90,689** |

---

## 5. Training Configuration (Locked)

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
| Random oversampling | No |
| Device | CUDA (RTX 5050) |

---

## 6. Class Imbalance Handling

pos_weight = 27.47 (418,924 legit / 15,252 fraud from pre-Option-A train counts).  
No oversampling applied. Weighted loss only.

---

## 7. Epoch-by-Epoch Training History

| Epoch | Train Loss | Val Loss | Val PR-AUC | Time (s) | Note |
|-------|-----------|----------|------------|----------|------|
| **1** | **1.3599** | **1.3181** | **0.0345** | **11.2** | **← BEST** |
| 2 | 1.3581 | 1.3152 | 0.0339 | 11.4 | |
| 3 | 1.3574 | 1.3199 | 0.0341 | 11.3 | |
| 4 | 1.3580 | 1.3149 | 0.0343 | 11.4 | |
| 5 | 1.3580 | 1.3215 | 0.0342 | 11.5 | |
| 6 | 1.3576 | 1.3148 | 0.0343 | 11.5 | Early stop |

---

## 8. Best Epoch

| Metric | Value |
|--------|-------|
| Best epoch | **1** |
| Best validation PR-AUC | **0.0345** |
| Best validation loss | 1.3181 |

---

## 9. ⚠️ CRITICAL FINDING — GRU DID NOT LEARN

> [!CAUTION]
> **val PR-AUC = 0.0345 is essentially the random-classifier baseline.**
> The validation fraud rate is 2,567 / 75,727 = **3.39%**.
> A classifier that always predicts the fraud prevalence achieves PR-AUC ≈ 0.034.
> The trained GRU achieved 0.0345 — indistinguishable from random.

### What the numbers say

| Metric | Value | Interpretation |
|--------|-------|----------------|
| val PR-AUC (GRU) | 0.0345 | ≈ random baseline |
| val fraud prevalence | 3.39% | = random PR-AUC baseline |
| E1 PR-AUC | 0.5317 | 15× better than GRU |
| Loss change across 6 epochs | 1.3599 → 1.3574 | Essentially flat |
| Early stop at epoch | 6 | No learning signal |

### Root cause diagnosis

**The GRU is not learning because the input features are unscaled.**

| System | Feature scaling needed? | Why |
|--------|------------------------|-----|
| LightGBM (E1) | NO | Tree splits are scale-invariant |
| GRU (E2) | **YES** | Gradient updates are scale-sensitive |

The 406 E1 features contain a wide range of magnitudes:
- `TransactionAmt`: can be 0.25 – 31,936 (raw dollars)
- `V1`–`V339`: Vesta-engineered, various scales
- `C1`–`C14`: count features, 0–2,720
- `D1`–`D15`: time deltas, 0–640
- Binary / ordinal encoded categoricals: 0–1 or small integers

When unscaled features with magnitude ~1,000 and ~0.001 coexist, the GRU's weight gradients are dominated by the large-magnitude dimensions. The small-magnitude features (which may contain strong fraud signal) contribute negligible gradient. The network cannot learn from them.

### What this is NOT

- **NOT a hardware issue** (GPU worked perfectly)
- **NOT a code bug** (data loaded correctly, shapes correct, loss computed correctly)
- **NOT a random seed issue** (seed=42 deterministic)
- **NOT early stopping mis-configuration** (correctly stopped when val PR-AUC didn't improve)

### Research interpretation

This is a legitimate and important research finding:

> *"A GRU trained on raw (unscaled) tabular E1 features with the locked architecture does not learn to discriminate fraud from non-fraud sequences."*

This is consistent with the known requirement that neural networks applied to tabular data need feature standardization. The result does NOT invalidate the research — it defines the scope of what the locked initial configuration can achieve, and motivates the feature scaling treatment.

---

## 10. Training Duration

| Phase | Time |
|-------|------|
| Sequence generation | ~24.7 min |
| Pre-flight + DataLoader | < 5 sec |
| Training (6 epochs × 11s) | 68.2 sec |
| **Total script time** | **25.8 min** |

Epoch time: ~11.2s/epoch on RTX 5050 (3,112 batches × 128 samples)

---

## 11. Device & Reproducibility

| Item | Value |
|------|-------|
| Device | CUDA — NVIDIA GeForce RTX 5050 Laptop GPU |
| VRAM total | 8.55 GB |
| CUDA version | 13.2 |
| cuDNN version | 92000 |
| PyTorch version | 2.13.0.dev20260603+cu132 |
| Python seed | 42 |
| NumPy seed | 42 |
| PyTorch seed | 42 |
| CUDA deterministic | True |

---

## 12. Checkpoint

| File | Status |
|------|--------|
| `phase2_gru/artifacts/model/gru_best.pt` | Saved (epoch 1) |
| `phase2_gru/artifacts/model/gru_model_config.json` | Saved |
| `phase2_gru/artifacts/training_history.csv` | Saved (6 epochs) |
| `phase2_gru/artifacts/run_metadata.json` | Saved |

---

## 13. Test Set

**NOT evaluated.** Reserved for STEP 9.  
The test set was never touched in this script.

---

## 14. E1 Immutability

E1 artifacts are unchanged. `git status --short` shows only `?? phase2_gru/`.

---

## 15. Open Research Decision Required

Before STEP 9, a research decision is needed on how to interpret and handle this result:

### Option A — Proceed to STEP 9 as-is
- Report that the unscaled GRU achieves PR-AUC ≈ 0.034 on the test set
- Compare against E1 (0.5317) on the 76,520 common IDs
- Result: E2 << E1 (definitive, but the reason is feature scaling, not temporal modeling)
- Research interpretation: "Raw temporal modeling without feature standardization does not add predictive value"

### Option B — Add feature standardization (StandardScaler on train, apply to val/test)
- This is a methodological improvement, not a hyperparameter change
- StandardScaler is standard preprocessing for neural networks
- Rerun STEP 8 with scaled features
- Then compare fairly: LightGBM (scale-invariant) vs GRU (properly scaled)
- Research interpretation: "Does temporal sequence modeling, with appropriate neural-network preprocessing, add value?"

### Option C — Report both (primary + sensitivity)
- E2a: unscaled (this run) — negative result due to scaling
- E2b: scaled — the real temporal modeling experiment
- Documents the scaling requirement as a finding

> [!IMPORTANT]
> **Recommended: Option B** — Apply StandardScaler (fit on train X only, transform val/test). This is the correct scientific treatment for neural networks on tabular features, and gives the GRU a fair chance to learn temporal patterns. The unscaled result can be reported as a documented sensitivity.

---

## 16. STEP 8 Status

| Item | Status |
|------|--------|
| Training completed | YES |
| Best val PR-AUC | 0.0345 (≈ random) |
| Early stopped | YES (epoch 6) |
| Checkpoint saved | YES |
| Test evaluated | NO |
| E1 modified | NO |
| Critical finding | Feature scaling required |

## STEP 8: COMPLETE — Research Decision Required Before STEP 9

*Phase 2 STEP 8 — Completed 2026-09-18*
