# Phase 2 — E2 GRU Sequence Specification
## LOCKED — v1.1 (2026-09-17)

> This document is the authoritative specification for Phase 2 GRU sequence construction.  
> **Any deviation from this specification must be documented and reviewed before implementation.**

---

## 1. Research Question

> **"Does historical transaction behavior provide additional predictive value beyond the tabular LightGBM baseline (E1)?"**

All outcomes — GRU better, equal, or worse — are valid research results. This specification is designed to give the GRU a fair and clean test of temporal behavioral information.

---

## 2. Locked Design Decisions

### Q1 — card1 Treatment: **Option A (Excluded from GRU input)**

| Property | Value |
|----------|-------|
| card1 role | Entity grouping key **only** |
| card1 in GRU input X? | **NO** |
| Justification | Prevents entity memorization; ensures GRU learns general behavioral patterns, not card identity. Strengthens methodological argument that temporal sequences — not card fingerprints — provide predictive value. |

### Q2 — Temporal Gap Feature: **YES**

| Property | Value |
|----------|-------|
| Feature name | `time_since_previous_transaction` |
| Unit | Seconds |
| Dtype | float32 |
| Included? | **YES** |
| Justification | Transaction timing carries fraud signal (rapid bursts vs. normal spacing). Pure sequential ordering without time gaps loses this information. |

---

## 3. Final E2 Feature Specification

```
E1 original features        : 406
  − card1 (entity key)      :  −1
                            ─────
E1 retained transaction feats: 405
  + time_since_prev_trans   :  +1
                            ─────
E2 final input features     : 406
```

**Explicitly excluded from GRU input:**

| Field | Reason |
|-------|--------|
| `card1` | Entity grouping key — NOT a behavioral feature for GRU |
| `isFraud` | Target label — NEVER in input |
| `TransactionID` | Row identifier — not predictive |
| `TransactionDT` | Raw temporal counter — ordering only; temporal info captured by hour_sin, hour_cos, day_index (already in feature set) and the gap feature |

The exact 406 E2 features are documented in:  
→ `phase2_gru/reports/E2_feature_manifest.json`

---

## 4. Entity Definition

| Parameter | Value |
|-----------|-------|
| Entity key | `card1` |
| Source column | `card1` in processed parquets |
| Interpretation | Anonymized card-level identifier — not a real customer ID |
| Usage | Group transactions by card1 to form sequences; sort within each group |
| Minimum entity size | 5 transactions (to support L=5 window) |

---

## 5. Sequence Window Definition

### 5.1 Window Parameters

| Parameter | Value |
|-----------|-------|
| Window length (L) | 5 |
| History steps | 4 (T1, T2, T3, T4) |
| Target position | 5th (T5) |
| Stride | 1 (sliding window) |
| Minimum card transactions | ≥ 5 |

### 5.2 Window Construction

For a card entity with sorted transactions `[T_a, T_b, T_c, ..., T_n]`:

```
Window 1:  history=[T_a, T_b, T_c, T_d]  target=T_e
Window 2:  history=[T_b, T_c, T_d, T_e]  target=T_f
Window 3:  history=[T_c, T_d, T_e, T_f]  target=T_g
...
```

### 5.3 GRU Input/Output Shape

```
X shape : [batch_size, 4, 406]
           └────────┘  └─┘  └───┘
           batch       seq  features
y shape : [batch_size, 1]   (binary: 0=legit, 1=fraud)
```

---

## 6. Temporal Ordering Rule

For every window and every card entity:

1. Sort all transactions by `TransactionDT` **ascending** before any window construction
2. Never randomly reorder transactions
3. For every window, strict ordering must hold:

```
TransactionDT(T1) < TransactionDT(T2) < TransactionDT(T3) < TransactionDT(T4) < TransactionDT(T5)
```

This is validated by **Leakage Test 4** (temporal ordering).

---

## 7. Causal Temporal Gap Feature

### 7.1 Definition

For a history window `[T1, T2, T3, T4]`:

| Position | Gap Computation | Value |
|----------|----------------|-------|
| T1 | No prior within window | `0.0` (initialization) |
| T2 | `TransactionDT(T2) − TransactionDT(T1)` | seconds |
| T3 | `TransactionDT(T3) − TransactionDT(T2)` | seconds |
| T4 | `TransactionDT(T4) − TransactionDT(T3)` | seconds |

### 7.2 Critical Rule — Target Gap Not Used

**The gap between T4 and T5 (target) is NOT computed and NOT included.**  
The model predicts T5 using only information from T1–T4.

```
ALLOWED:  gap(T2-T1), gap(T3-T2), gap(T4-T3)
FORBIDDEN: gap(T5-T4)   ← uses target timestamp → leakage risk
```

### 7.3 Gap Feature Properties

| Property | Value |
|----------|-------|
| Unit | Seconds |
| Always non-negative? | Yes (transactions sorted ascending) |
| Scale | May require normalization (log1p or StandardScaler from train only) |
| Missing (T1) | 0.0 |
| Causal? | Yes — uses only past timestamps |

### 7.4 Gap Normalization

Gap values can be very large (days to seconds range). Apply:
```
gap_normalized = log1p(gap_seconds)
```
Normalization statistics (if using StandardScaler) must be computed **on train sequences only**.

---

## 8. Split Assignment Rule

**The target transaction's split determines the window's split assignment.**

| Window's split | Determined by |
|---------------|--------------|
| TRAIN | target T5 is in TRAIN (TransactionDT ≤ 10,972,800) |
| VALIDATION | target T5 is in VALIDATION |
| TEST | target T5 is in TEST |

### 8.1 Causal Cross-Split History Rule (LOCKED)

History transactions may come from prior splits, as in real deployment:

```
VALIDATION window example:
  history [T1, T2, T3, T4] may be from TRAIN period  ✅
  target  [T5]             must be in VALIDATION period

TEST window example:
  history [T1, T2, T3, T4] may be from TRAIN + VALIDATION periods  ✅
  target  [T5]             must be in TEST period
```

**Forbidden:**

```
Using T5 or later transactions in the history  ❌
Using target TransactionDT to compute any history feature  ❌
```

---

## 9. Leakage Tests (All 8 Required)

All tests must **PASS** before any training begins.

| Test | Description | Test File |
|------|-------------|-----------|
| **LT1** | Target transaction not in history | `test_sequence_leakage.py` |
| **LT2** | All history TransactionDTs < target TransactionDT | `test_temporal_order.py` |
| **LT3** | `isFraud` absent from input features X | `test_label_isolation.py` |
| **LT4** | History sorted ascending by TransactionDT | `test_temporal_order.py` |
| **LT5** | No future transaction in history | `test_temporal_order.py` |
| **LT6** | Target transaction features not in history vector | `test_sequence_leakage.py` |
| **LT7** | Split assignment: target's split = window's label split | `test_split_integrity.py` |
| **LT8** | Gap feature uses only T1–T4 timestamps; T5 gap not included | `test_sequence_leakage.py` |

---

## 10. Class Imbalance Handling

| Parameter | Value |
|-----------|-------|
| Train fraud count | 15,252 |
| Train legit count | 418,924 |
| Positive weight | `418,924 / 15,252 ≈ 27.47` |
| Strategy | `BCEWithLogitsLoss(pos_weight=tensor([27.47]))` |
| SMOTE? | **No** — breaks temporal ordering |
| Random oversampling? | **No** — not justified for sequences |

---

## 11. Threshold Selection Protocol

| Rule | Value |
|------|-------|
| Threshold selection dataset | **Validation only** |
| Selection metric | F1-maximizing (same as E1) |
| Threshold frozen before test? | **Yes** |
| Test set evaluated | **Once only** |

---

## 12. GRU Architecture (Initial)

| Parameter | Value |
|-----------|-------|
| Input size | 406 |
| Hidden size | 64 (initial) |
| Num layers | 1 (initial) |
| Dropout | 0.2 |
| Output | Linear(64 → 1) + Sigmoid |
| Framework | PyTorch |

---

## 13. Training Configuration (Initial)

| Parameter | Value |
|-----------|-------|
| Random seed | 42 |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 128 |
| Max epochs | 30 |
| Early stopping patience | 5 (on validation PR-AUC) |
| Loss | BCEWithLogitsLoss(pos_weight=27.47) |

---

## 14. Reproducibility Requirements

Must save:
- Model weights (`artifacts/model/gru_weights.pt`)
- Model config JSON
- E2 feature list (`E2_feature_manifest.json`)
- Sequence config
- Training history CSV
- Run metadata JSON
- Random seed

---

## 15. Comparison Protocol

**Paired bootstrap** on the same test TransactionIDs.

Metric: Δ PR-AUC = PR-AUC(E2) − PR-AUC(E1)

Report 95% CI for Δ. If CI contains 0: no statistically significant difference. No winner declared without evidence.

---

*Locked by: Phase 2 design review 2026-09-17*  
*Q1: card1 = entity key only (Option A)*  
*Q2: time_since_previous_transaction included*  
*E2 input features: 406 = 405 E1 retained + 1 gap*  
*Next step: STEP 6 — sequence_builder.py implementation + 8 leakage tests*
