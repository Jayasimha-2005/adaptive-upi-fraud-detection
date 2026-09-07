# Phase 1 — LightGBM Baseline Report (E1)

**Dataset:** IEEE-CIS Fraud Detection (Vesta Corporation / Kaggle Competition)
**Type:** Real-world-derived, anonymized, historical e-commerce transactions

> This is a baseline experiment. No GRU, Transformer, Kafka, or adaptive retraining.
> The primary metric is **PR-AUC** (Average Precision). Accuracy is not reported.

## Dataset

- Training rows: 434,176
- Validation rows: 77,822
- Test rows: 78,542
- Training fraud: 15,252 (3.513%)
- Features: 406

## Temporal Split

| Split | Days | Rows | Fraud % |
|-------|------|------|---------|
| Train | 1–127 | 434,176 | 3.513% |
| Validation | 127–155 | 77,822 | 3.389% |
| Test | 155–183 | 78,542 | 3.532% |

## Preprocessing

- JOIN: LEFT JOIN train_transaction + train_identity on TransactionID
- Temporal features: hour_sin, hour_cos, day_index (from TransactionDT)
- Missingness indicators: binary flags for key high-missing columns
- Numerical imputation: median (fit on train only)
- Categorical encoding: LabelEncoder (fit on train only)
- Dropped: D6,D7,D8,D9,D12,D13,D14 (>80% missing), dist2 (93.6% missing)
- Dropped: V-columns with >80% missing in training data
- Excluded: TransactionID, isFraud, TransactionDT (raw)

## Model Configuration

- Model: LightGBM 4.7.0
- Objective: binary
- Metric: average_precision (PR-AUC)
- is_unbalance: True (class imbalance handling)
- Best iteration: 1000
- Training time: 50.66s

## Validation Results (threshold frozen here)

| Metric | Value |
|--------|-------|
| **PR-AUC** | **0.584935** |
| ROC-AUC    | 0.923115 |
| Precision  | 0.628479 |
| Recall     | 0.513841 |
| F1         | 0.565408 |
| MCC        | 0.554723 |
| Balanced Accuracy | 0.751594 |
| FPR        | 0.010654 |
| FNR        | 0.486159 |
| Decision threshold | 0.616521 |

## Final Test Results (evaluated ONCE with frozen threshold)

| Metric | Value |
|--------|-------|
| **PR-AUC** | **0.531731** |
| ROC-AUC    | 0.898993 |
| Precision  | 0.581810 |
| Recall     | 0.493511 |
| F1         | 0.534035 |
| MCC        | 0.520301 |
| Balanced Accuracy | 0.740262 |
| FPR        | 0.012987 |
| FNR        | 0.506489 |
| TP=1369 FP=984 TN=74784 FN=1405 | |

## Top Features (by Gain)

| Rank | Feature | Gain |
|------|---------|------|
| 1 | `V258` | 792886.35 |
| 2 | `C13` | 454330.72 |
| 3 | `V294` | 393786.40 |
| 4 | `card1` | 363974.53 |
| 5 | `TransactionAmt` | 345534.19 |
| 6 | `day_index` | 314009.21 |
| 7 | `card2` | 292409.57 |
| 8 | `addr1` | 245972.37 |
| 9 | `C8` | 194804.98 |
| 10 | `D15` | 174443.87 |
| 11 | `C1` | 173555.04 |
| 12 | `M4` | 172864.70 |
| 13 | `D2` | 171575.03 |
| 14 | `card6` | 145283.47 |
| 15 | `V70` | 144995.73 |

## Limitations

1. This is a baseline — no hyperparameter tuning (Optuna) performed.
2. No GRU temporal sequence modeling (Phase 2).
3. No concept drift adaptation (Phase 3).
4. No streaming evaluation (Phase 4).
5. SHAP computed on validation sample only.
6. Dataset is anonymized — V-column interpretations are not available.
7. IEEE-CIS is e-commerce fraud, not UPI transaction fraud.

## Reproducibility

- Random seed: 42
- Python: 3.13.13
- LightGBM: 4.7.0
- pandas: 3.0.5
- numpy: 2.4.6

**Reproduction command:**
```bash
python experiments/run_phase1.py --config configs/phase1_lightgbm.yaml
```

## Next Recommended Phase

Phase 2: Temporal sequence modeling using GRU on card1 entity histories.
Requires: Phase 1 model as tabular baseline for comparison.