# Phase 2 — E1 vs E2b Paired Comparison Report

**Generated:** 2026-09-18  
**Population:** 76,520 common test TransactionIDs  
**Bootstrap:** 2,000 paired resamples, seed=42

---

## 1. Experimental Design

| Property | E1 | E2b |
|---------|-----|------|
| Model | LightGBM (tabular) | GRU (sequence) |
| Features | 406 (full E1 set) | 406 (405 E1 + 1 log1p gap) |
| Temporal modeling | None | 4-step history window |
| Feature scaling | Not required (tree-based) | StandardScaler (train-fitted) |
| Training population | 434,176 transactions | 398,312 sequences |
| Parameters | ~3,700 leaves × splits | 90,689 |
| Threshold | 0.616521 (frozen) | 0.810221 (val F1-max) |
| Status | **Frozen baseline** | Experimental variant |

---

## 2. Test Population Coverage

```
E1  full test                :  78,542 TransactionIDs
E2b eligible test sequences  :  76,520 TransactionIDs
─────────────────────────────────────────────────────
Common paired population     :  76,520  ← primary comparison
E1-only (no E2b sequence)    :   2,022  (3.08% of E1 test set, excluded from primary)
E2b coverage                 :  97.43%
```

The 2,022 E1-only transactions are excluded from the primary paired comparison because they cannot form valid 4-step sequences. For completeness, E1 on the full 78,542 is reported separately.

---

## 3. Primary Test Metrics — Common 76,520 Population

| Metric | E1 (common) | E2b (common) | Direction |
|--------|------------|-------------|-----------|
| **PR-AUC** | **0.5267** | 0.1683 | E1 > E2b |
| **ROC-AUC** | **0.8981** | 0.7422 | E1 > E2b |
| F1 | **0.5308** | 0.2432 | E1 > E2b |
| Precision | **0.5790** | 0.1913 | E1 > E2b |
| Recall | **0.4900** | 0.3337 | E1 > E2b |
| MCC | **0.5170** | 0.2166 | E1 > E2b |
| Balanced Accuracy | **0.7385** | 0.6411 | E1 > E2b |
| Brier Score | **0.0324** | 0.2377 | E1 better (lower) |
| ECE | **0.0523** | 0.4247 | E1 better (lower) |
| FPR | **0.0130** | 0.0516 | E1 better (lower) |
| FNR | **0.5100** | 0.6663 | E1 better (lower) |

---

## 4. Operational Comparison — Common 76,520 Population

| Metric | E1 | E2b |
|--------|-----|------|
| Recall @ 0.1% FPR | **0.2081** | 0.0215 |
| Recall @ 0.5% FPR | **0.3770** | 0.0985 |
| Recall @ 1.0% FPR | **0.4578** | 0.1441 |
| Recall @ 2.0% FPR | **0.5259** | 0.2119 |
| Precision @ Top-100 | **0.9800** | 0.4800 |
| Precision @ Top-500 | **0.9040** | 0.4280 |
| Precision @ Top-1,000 | **0.8310** | 0.3650 |
| Precision @ Top-5,000 | **0.3362** | 0.1850 |

At every operational threshold, E1 achieves substantially higher fraud detection rates at lower false positive rates. The operational gap is consistent with the PR-AUC gap.

---

## 5. Paired Bootstrap — Primary Metric (Δ PR-AUC)

**Convention: Δ = PR-AUC(E2b) − PR-AUC(E1)**

| Quantity | Value |
|---------|-------|
| Observed E1 PR-AUC | 0.5267 |
| Observed E2b PR-AUC | 0.1683 |
| **Observed Δ PR-AUC** | **−0.3584** |
| 95% CI lower | −0.3756 |
| 95% CI upper | −0.3395 |
| **CI excludes zero** | **Yes** |
| Valid resamples | 2,000 / 2,000 |
| Seed | 42 |

**Paired bootstrap ROC-AUC (secondary):**

| Quantity | Value |
|---------|-------|
| Observed E1 ROC-AUC | 0.8981 |
| Observed E2b ROC-AUC | 0.7422 |
| Observed Δ ROC-AUC | −0.1559 |
| 95% CI | [−0.16715940175849875, −0.14454939181321735] |
| CI excludes zero | Yes |

> [!IMPORTANT]
> The 95% paired bootstrap CI for Δ PR-AUC lies entirely below zero: [−0.3756, −0.3395]. Across all 2,000 resamples of the same 76,520 test transactions, the paired analysis consistently finds E1 superior to E2b on PR-AUC. The CI does not contain zero, indicating that the observed gap is not attributable to sampling variability alone in this evaluation.

---

## 6. Reference: E1 on Full Test Population

For documentation purposes, the original E1 baseline metrics on the full 78,542-transaction test set (as reported in Phase 1.5):

| Metric | E1 (full 78,542) | E1 (common 76,520) | Difference |
|--------|-------------------|-------------------|-----------|
| PR-AUC | 0.531731 | 0.5267 | −0.0050 |
| ROC-AUC | 0.898993 | 0.8981 | −0.0009 |
| F1 | 0.534035 | 0.5308 | −0.0032 |

The small reduction in E1 metrics on the common population (vs. full population) is expected: the 2,022 excluded transactions are those without sufficient card history, and such transactions may be systematically easier to classify.

---

## 7. Research Interpretation

**Research question:** *"Does historical transaction behavior provide additional predictive value beyond the tabular LightGBM baseline?"*

**Empirical finding:**

Under the locked experimental conditions — a single-layer GRU (hidden=64, dropout=0.2) trained on 4-step sequences of 406 standardized features with Adam optimizer (lr=1e-3, pos_weight=27.47, early stopping on val PR-AUC) — the E2b GRU does **not** provide additional predictive value over E1 on the common test population.

The 95% paired bootstrap CI for Δ PR-AUC = [−0.3756, −0.3395] excludes zero. The direction is negative throughout.

**Possible interpretations consistent with this finding:**

1. **The tabular representation (406 E1 features) already captures most fraud-predictive information available from the raw transaction record.** A GRU processing the same features in temporal sequence does not extract additional signal beyond what LightGBM finds in the per-transaction feature space.

2. **The L=4 history window may be insufficient.** Longer sequences, or sequences grouped by different entity definitions, might reveal temporal patterns not detectable in 4-step windows.

3. **The GRU architecture (90,689 parameters, single layer) may underfit the temporal complexity of the sequences.** The early overfitting pattern (best epoch=2) suggests the model may benefit from regularization or architecture adjustments — but these are outside the locked experimental scope.

4. **Feature engineering for the GRU may be incomplete.** E1 features were engineered for tabular LightGBM (including many aggregation and frequency features that may already encode temporal information). A GRU processing these features in sequence may be redundantly summarizing information the features already contain.

> [!NOTE]
> This report does not declare a "winner." The experimental finding is that E1 PR-AUC (0.5267) substantially exceeds E2b PR-AUC (0.1683) on the 76,520 common test transactions, with a paired bootstrap CI that excludes zero. The scientific conclusion is that the specific E2b configuration does not improve over E1 under these locked conditions. Future work may investigate different architectures, window lengths, entity definitions, or feature representations.

---

## 8. Experiment Summary Table

| Property | E1 | E2a | E2b |
|---------|-----|------|------|
| Type | LightGBM (tabular) | GRU, unscaled | GRU, StandardScaler |
| Val PR-AUC | 0.5849 | 0.0345 | 0.1377 |
| **Test PR-AUC** | **0.5267** | *(not evaluated)* | **0.1683** |
| Test ROC-AUC | 0.8981 | — | 0.7422 |
| Best epoch | — | 1 | 2 |
| Early stopped | — | Epoch 6 | Epoch 7 |
| Training time | — | 68 sec (GPU) | 79 sec (GPU) |
| Scaler | None | None | StandardScaler (train) |
| Paired Δ PR-AUC | — | — | −0.3584 |
| Bootstrap CI | — | — | [−0.3756, −0.3395] |

---

*E1 vs E2b Paired Comparison — Phase 2 STEP 9 — 2026-09-18*  
*Common population: 76,520 TransactionIDs | 2,000 paired bootstrap resamples | seed=42*
