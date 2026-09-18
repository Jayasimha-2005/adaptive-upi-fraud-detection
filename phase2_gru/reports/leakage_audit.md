# Phase 2 — Leakage Audit Report
## LT1–LT8 Complete Results

**Generated:** 2026-09-17  
**Test run:** `python -m pytest phase2_gru/tests/ -v`  
**Result: ✅ 33/33 PASSED (7.22s)**

---

## Summary Table

| ID | Leakage Area | Tests | Result | Key Assertion |
|----|-------------|-------|--------|---------------|
| **LT1** | Target not in history | 3 | ✅ PASS | `target_tid` not in history TIDs; `target_dt` > all hist DTs |
| **LT2** | Strict temporal ordering | 3 | ✅ PASS | `DT(T1) < DT(T2) < DT(T3) < DT(T4) < DT(T5)` |
| **LT3** | Label isolation | 8 | ✅ PASS | `isFraud`, `card1`, `TransactionID`, `TransactionDT` all absent from X |
| **LT4** | History strictly ascending | 3 | ✅ PASS | `diff(hist_DTs) > 0` strictly (not just ≥ 0) |
| **LT5** | No future transactions | 2 | ✅ PASS | Cross-split history causal; no post-target DTs |
| **LT6** | Target features absent | 1 | ✅ PASS | Target row features not present in history matrix |
| **LT7** | Split assignment integrity | 6 | ✅ PASS | Window split = target transaction's split |
| **LT8** | Target gap not used | 6 | ✅ PASS | Only T1–T4 gaps computed; `DT(T5)-DT(T4)` never included |
| **Total** | | **33** | ✅ **33/33** | |

---

## LT1 — Target Transaction Not in History

**File:** `tests/test_sequence_leakage.py::TestLT1_TargetNotInHistory`

| Test | Result |
|------|--------|
| `test_target_tid_not_in_history_tids` | ✅ PASS |
| `test_history_rows_strictly_before_target` | ✅ PASS |
| `test_window_count_correct` | ✅ PASS |

**Verification:** For a window `[T1, T2, T3, T4] → T5`:
- T5's `TransactionID` does not appear among T1–T4
- T5's `TransactionDT` > all of T1–T4's DTs
- For n transactions: exactly `n - L + 1` windows generated ✅

---

## LT2 — Strict Temporal Ordering

**File:** `tests/test_temporal_order.py::TestLT2_HistoryBeforeTarget`

| Test | Result |
|------|--------|
| `test_all_history_dts_before_target` | ✅ PASS |
| `test_equal_dt_window_excluded_option_a` | ✅ PASS |
| `test_future_dt_not_in_any_retained_window` | ✅ PASS |

**Critical — Option A verification:**
- Window with `DT(T4) == DT(T5)` was **excluded** ✅
- All retained windows satisfy strict `<` (not `<=`) ✅
- `hist_dt > target_dt` correctly raises `AssertionError` ✅

---

## LT3 — Label Isolation

**File:** `tests/test_label_isolation.py::TestLT3_LabelIsolation`

| Test | Result |
|------|--------|
| `test_isfraud_not_in_e2_feature_list` | ✅ PASS |
| `test_transactionid_not_in_e2_features` | ✅ PASS |
| `test_transactiondt_not_in_e2_features` | ✅ PASS |
| `test_card1_not_in_e2_features` | ✅ PASS |
| `test_isfraud_not_passed_as_feature_to_builder` | ✅ PASS |
| `test_x_values_not_equal_to_isfraud_column` | ✅ PASS |
| `test_feature_manifest_json_confirms_exclusion` | ✅ PASS |
| `test_e2_feature_count_is_406` | ✅ PASS |

**E2_feature_manifest.json confirmation:**
- `isFraud` listed in `excluded_from_gru_input` ✅
- Final feature count = 406 ✅

---

## LT4 — History Strictly Ascending

**File:** `tests/test_temporal_order.py::TestLT4_HistorySortedAscending`

| Test | Result |
|------|--------|
| `test_ascending_gaps_are_strictly_positive` | ✅ PASS |
| `test_equal_dt_gap_is_zero_and_excluded` | ✅ PASS |
| `test_sort_order_before_window_construction` | ✅ PASS |
| `test_unsorted_entity_correctly_sorted_inside_builder` | ✅ PASS |

**Option A mechanism confirmed:** Equal-DT sequences produce `diff = 0` → excluded by `np.diff(all_window_dts) > 0` check ✅

---

## LT5 — No Future Transactions in History

**File:** `tests/test_temporal_order.py::TestLT5_NoFutureInHistory`

| Test | Result |
|------|--------|
| `test_cross_split_history_respects_causal_rule` | ✅ PASS |
| `test_no_test_row_in_train_history` | ✅ PASS |

**Cross-split causal behavior confirmed:**
- TRAIN history → VALIDATION target: ✅ allowed and correctly assigned `split='validation'`
- TRAIN+VAL history → TEST target: ✅ allowed and correctly assigned `split='test'`
- All history DTs strictly before target DT in cross-split cases ✅

---

## LT6 — Target Features Not in History

**File:** `tests/test_sequence_leakage.py::TestLT6_TargetFeaturesNotInHistory`

| Test | Result |
|------|--------|
| `test_target_features_absent_from_history_matrix` | ✅ PASS |

**Structural guarantee:** History rows are indices `[start : start+4]`, target is index `start+4`. Different DataFrame rows → distinct feature values. Verified with deterministic toy data where each row has unique values ✅.

---

## LT7 — Split Assignment Integrity

**File:** `tests/test_split_integrity.py::TestLT7_SplitIntegrity`

| Test | Result |
|------|--------|
| `test_train_target_yields_train_window` | ✅ PASS |
| `test_validation_target_yields_validation_window` | ✅ PASS |
| `test_test_target_yields_test_window` | ✅ PASS |
| `test_assign_split_boundaries` | ✅ PASS |
| `test_no_split_contamination` | ✅ PASS |
| `test_window_counts_with_real_data_smoke` | ✅ PASS |

**Boundary conditions verified:**

| DT value | `assign_split()` | Expected |
|----------|-----------------|----------|
| 10,972,800 | `'train'` | ✅ |
| 10,972,801 | `'validation'` | ✅ |
| 13,392,000 | `'validation'` | ✅ |
| 13,392,001 | `'test'` | ✅ |

**Real-data smoke test:** 100 entities loaded, all target DTs verified to match assigned split ✅

---

## LT8 — Target Gap (T5-T4) Not Used

**File:** `tests/test_sequence_leakage.py::TestLT8_TargetGapNotUsed`

| Test | Result |
|------|--------|
| `test_gap_feature_is_last_column` | ✅ PASS |
| `test_only_four_gaps_computed` | ✅ PASS |
| `test_t1_gap_is_zero` | ✅ PASS |
| `test_gaps_are_non_negative` | ✅ PASS |
| `test_gap_uses_log1p` | ✅ PASS |
| `test_x_shape_has_gap_column` | ✅ PASS |

**Gap computation verified:**

| Position | Formula | Confirmed |
|----------|---------|-----------|
| T1 | `0.0` (no prior in window) | ✅ |
| T2 | `log1p(DT2 - DT1)` | ✅ |
| T3 | `log1p(DT3 - DT2)` | ✅ |
| T4 | `log1p(DT4 - DT3)` | ✅ |
| T5 | **NOT COMPUTED** | ✅ |

`compute_gaps()` returns exactly 4 values (HISTORY_LEN). X.shape confirmed `(4, 406)` with gap as last column ✅.

---

## Duplicate TransactionDT Audit

| Metric | Value |
|--------|-------|
| Cards with duplicate DTs | 59 / 12,421 (train) |
| Same-second row pairs | 146 |
| Treatment | Option A — exclude windows |
| Windows excluded | **585** |
| Exclusion rate | **~0.11%** |
| LT2 test `test_equal_dt_window_excluded_option_a` | ✅ PASS |
| LT4 test `test_equal_dt_gap_is_zero_and_excluded` | ✅ PASS |

---

## Final Verdict

```
============================= 33 passed in 7.22s ==============================
```

**LT1–LT8: ALL PASS ✅**  
**Option A duplicate-DT treatment: VERIFIED ✅**  
**No leakage pathways detected.**

---

*Generated by: Phase 2 STEP 6 | sequence_builder.py v1.1 (Option A)*  
*Next step: STEP 7 — GRU model architecture (awaiting approval)*
