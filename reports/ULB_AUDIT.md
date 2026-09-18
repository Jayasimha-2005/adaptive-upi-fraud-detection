# ULB Credit Card Fraud Dataset — Detailed Audit
**Generated:** 2026-09-18 | **Audit only. No training.**

---

## 1. File Verification

| Property | Expected | Actual |
|---------|---------|--------|
| Path | `Datasets/ulb_creditcard/creditcard.csv` | ✅ Present |
| Rows | ~284K | **284,807** |
| Columns | 31 | **31** |
| Fraud count | ~492 | **492** |
| Fraud rate | ~0.172% | **0.1727%** |
| Time column | Yes | ✅ `Time` |
| Amount column | Yes | ✅ `Amount` |
| V features | V1–V28 | ✅ 28 features |
| Target column | `Class` | ✅ `Class` (0/1) |

---

## 2. Column Names (Actual)

```
Time, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13,
V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25,
V26, V27, V28, Amount, Class
```

---

## 3. Class Distribution

| Class | Count | Percentage |
|-------|-------|-----------|
| 0 (Legitimate) | 284,315 | 99.827% |
| 1 (Fraud) | **492** | **0.1727%** |
| Total | 284,807 | 100% |

**Extremely imbalanced** — 578:1 ratio. Much more severe than IEEE-CIS (~28:1).

---

## 4. Temporal Analysis

| Property | Value |
|---------|-------|
| Time column type | float64 (seconds elapsed) |
| Time minimum | **0.0 sec** |
| Time maximum | **172,792.0 sec** |
| Duration | **48.0 hours exactly** |
| Unique Time values | 124,592 |
| Duplicate Time values | 160,215 (many transactions share the same second) |
| Monotone increasing? | **Yes** (non-decreasing) |
| Strictly increasing? | **No** (ties exist) |

**Critical limitation:** 48 hours is insufficient for concept drift research, which requires weeks to months of temporal coverage to observe meaningful distributional shift.

---

## 5. Duplicate Row Analysis

| Category | Count |
|---------|-------|
| Exact duplicate rows (all 31 columns) | **1,081** |
| — Of which fraud (Class=1) | **32** |
| — Of which legit (Class=0) | **1,822** |
| Unique rows | 283,726 |
| Duplicate (Time+Amount only) | 4,863 |

The 1,081 full-row duplicates suggest either repeated logging of the same transaction or a data quality issue. Any experiment should decide on a deduplication strategy before training. The presence of 32 fraudulent duplicate rows means naive deduplication could slightly affect fraud recall metrics.

---

## 6. V Feature Analysis

| Feature | Min | Max | Std | Notes |
|--------|-----|-----|-----|-------|
| V1 | −56.41 | 2.45 | 1.96 | PCA component |
| V2 | −72.72 | 22.06 | 1.65 | PCA component |
| V3 | −48.33 | 9.38 | 1.52 | PCA component |
| V4 | −5.68 | 16.88 | 1.42 | PCA component |
| V5 | −113.74 | 34.80 | 1.38 | PCA component |
| ... | | | | |
| V28 | | | | PCA component |

**All V1–V28 are zero-mean principal components of undisclosed original features.** They are not interpretable on their own and cannot be compared to IEEE-CIS V features.

> [!IMPORTANT]
> **IEEE-CIS V1–V339 ≠ ULB V1–V28.**
>
> IEEE-CIS V features: Vesta-proprietary engineered transaction features (count, frequency, aggregation-derived). NOT PCA.
>
> ULB V features: PCA-transformed principal components of withheld original European credit card transaction features.
>
> These are dataset-specific representations. They must NEVER be combined, compared as equivalent, or processed with shared preprocessing objects.

---

## 7. Entity Identifier Analysis

**ULB has NO entity identifier.** There is no card ID, account ID, or customer ID column. The original cardholders cannot be identified or tracked across transactions.

**Consequence for temporal modeling:**
- The locked Phase 2 sequence specification (group by `card1`, 4-step windows) **cannot be applied to ULB**
- Time-based sequencing without entity grouping would produce sequences from random cardholders mixed together — methodologically invalid
- Any temporal GRU experiment on ULB would require a completely different sequence construction methodology

---

## 8. Amount Distribution

| Statistic | Value |
|---------|-------|
| Min | 0.00 |
| Max | 25,691.16 |
| Mean | 88.35 |
| Std | 250.12 |
| Fraud mean Amount | (to be computed if needed) |

---

## 9. Missing Values

| Metric | Value |
|--------|-------|
| Total missing values | **0** |
| Features with any NaN | 0 |

ULB is clean — all 31 columns fully populated.

---

## 10. Temporal Coverage Suitability

| Research Use | Suitable? | Reason |
|-------------|---------|--------|
| Tabular baseline model | ✅ Yes | Clean, well-known benchmark |
| External validation | ✅ Yes | Good for cross-dataset robustness check |
| Concept drift research | ❌ No | 48 hours insufficient for meaningful drift observation |
| Adaptive retraining | ❌ No | 48 hours insufficient — no concept drift windows |
| Temporal sequence GRU | ❌ No | No entity identifier for sequence grouping |
| Streaming demonstration | ⚠️ Marginal | Can simulate streaming by replaying Time column |

---

## 11. Recommended Preprocessing for ULB (If Used)

| Component | Treatment |
|----------|----------|
| Missing values | None needed |
| Duplicate rows | Decision needed: drop duplicates OR keep all |
| Time | Use as sort key; compute transaction hour-of-day if needed |
| V1–V28 | StandardScaler (already approximately zero-mean PCA) |
| Amount | Log1p + StandardScaler |
| Class | Target — isolate strictly |
| Entity grouping | Not possible |
| Temporal split | Chronological by Time (first ~80% train, ~10% val, ~10% test) |

**CRITICAL:** Any ULB preprocessing objects (scaler, imputer) must be fitted on ULB training data only. They must not be shared with IEEE-CIS preprocessing.

---

## 12. ULB Research Role Assignment

**Recommended role: EXTREME CLASS IMBALANCE BENCHMARK + EXTERNAL VALIDATION**

ULB is a well-established public benchmark (cited in thousands of papers). Its value for this project is:
1. Testing whether the tabular LightGBM approach generalizes to a different fraud domain (European credit cards vs. US e-commerce)
2. Demonstrating handling of extreme class imbalance (0.17% vs. IEEE-CIS 3.5%)
3. Providing an independent external validation for tabular modeling methodology

**Not recommended for:**
- Concept drift (48h duration)
- Temporal GRU (no entity identifier)
- Adaptive retraining (no drift windows)

---

*ULB Audit — 2026-09-18 | Read-only. No training. No preprocessing changes.*
