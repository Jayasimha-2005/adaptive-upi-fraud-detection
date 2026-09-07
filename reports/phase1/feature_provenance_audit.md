# IEEE-CIS Feature Provenance Audit
## Phase 1.5C — Temporal Leakage Risk Assessment

**Experiment:** E1_lightgbm | **Dataset:** IEEE-CIS Fraud Detection (Vesta/Kaggle 2019)
**Status:** Additive audit — E1 model is IMMUTABLE

## Provenance Tiers

| Tier | Icon | Meaning |
|------|------|---------|
| KNOWN | ✅ | Temporal scope confirmed from competition documentation or Vesta statements |
| STRONGLY_INFERRED | 🟡 | Strong indirect evidence but not formally documented |
| UNKNOWN_OPAQUE | 🟠 | No public documentation; cannot rule out future-information embedding |
| POTENTIAL_LEAKAGE | 🔴 | Concrete reason to suspect forward-looking computation |

## Tier Summary

| Tier | Feature Groups |
|------|--------------|
| ✅ KNOWN | 6 |
| 🟡 STRONGLY_INFERRED | 8 |
| 🟠 UNKNOWN_OPAQUE | 2 |
| 🔴 POTENTIAL_LEAKAGE | 0 |

> [!IMPORTANT]
> V1–V339 and some id-columns are classified as UNKNOWN_OPAQUE. Any paper or report
> using these features must explicitly state that their temporal scope is unverified
> and that temporal leakage cannot be formally ruled out without Vesta's internal documentation.

---

## Detailed Provenance by Feature Group

### 1. TransactionDT
**Tier:** ✅ `KNOWN`  
**Temporal Scope:** Point-in-time — timestamp of current transaction  
**Evidence:** Documented in competition description as a timedelta from a fixed reference epoch, measured in seconds. No future information possible.  
**Action:** KEEP — used for temporal features (hour_sin, hour_cos); raw DT excluded from features

### 2. TransactionAmt
**Tier:** ✅ `KNOWN`  
**Temporal Scope:** Point-in-time — amount of current transaction  
**Evidence:** Transaction amount is observed at the moment of transaction.  
**Action:** KEEP

### 3. ProductCD
**Tier:** ✅ `KNOWN`  
**Temporal Scope:** Point-in-time — product category  
**Evidence:** Five-category product code; available at transaction time.  
**Action:** KEEP

### 4. card1–card6
**Tier:** ✅ `KNOWN`  
**Temporal Scope:** Point-in-time — card static attributes  
**Evidence:** Card number prefix, card type, issuing country/bank, and card category. These are static properties of the card, not computed aggregates. No temporal dependency.  
**Action:** KEEP — primary entity key (card1) for Phase 2 sequences

### 5. addr1, addr2
**Tier:** 🟡 `STRONGLY_INFERRED`  
**Temporal Scope:** Point-in-time — billing address region (current transaction)  
**Evidence:** Billing and shipping region codes. Addresses are reported per transaction and are not aggregated. However, the encoding (numeric region code) may reflect a global mapping computed offline — minor risk.  
**Action:** KEEP — low risk

### 6. dist1, dist2
**Tier:** 🟡 `STRONGLY_INFERRED`  
**Temporal Scope:** Point-in-time — distance between addresses  
**Evidence:** Inferred to be distance between billing and shipping addresses for the current transaction. dist2 dropped (93.6% missing). No future dependency expected, but not formally documented.  
**Action:** dist1 KEPT, dist2 DROPPED (>80% missing)

### 7. P_emaildomain, R_emaildomain
**Tier:** 🟡 `STRONGLY_INFERRED`  
**Temporal Scope:** Point-in-time — email domain of purchaser and recipient  
**Evidence:** Email domains are per-transaction attributes. No aggregation expected. Some forum posts confirm these are raw field values.  
**Action:** KEEP

### 8. C1–C14 (counting aggregates)
**Tier:** 🟡 `STRONGLY_INFERRED`  
**Temporal Scope:** Inferred backward-looking: cumulative count of distinct entities (addresses, emails, cards) linked to the card up to the current transaction  
**Evidence:** Competition host (Vesta) described C-columns as 'counting' features. Competition forum posts suggest C1 = # addresses associated with card, C13 = # transactions on card, etc. If counts are cumulative up to (but NOT including) the current transaction, they are point-in-time safe. However, if they include the current transaction or use future data, they carry POTENTIAL_LEAKAGE. Exact window is NOT formally documented.  
**Action:** KEEP for E1 baseline. Flag for Phase 1.5C sensitivity test: measure model performance with C-columns removed to assess dependency. Report in paper as 'inferred backward-looking, not formally verified'.

### 9. D1–D5, D10, D11 (time-delta, kept)
**Tier:** 🟡 `STRONGLY_INFERRED`  
**Temporal Scope:** Inferred: days since a historical event (first card seen, last transaction, etc.) measured at the current transaction time  
**Evidence:** Competition host described D-columns as 'timedelta' features. D1 in particular is widely interpreted as 'days since card was first seen'. If measured relative to current transaction's TransactionDT, they are definitionally backward-looking. NOT formally documented by Vesta.  
**Action:** KEEP — but document as inferred, not verified

### 10. D6, D7, D8, D9, D12, D13, D14 (dropped)
**Tier:** 🟡 `STRONGLY_INFERRED`  
**Temporal Scope:** Inferred time-delta (same as above)  
**Evidence:** Same reasoning as D1–D5 group, but these have >80% missingness.  
**Action:** DROPPED due to >80% missingness — provenance irrelevant

### 11. M1–M9 (match flags)
**Tier:** 🟡 `STRONGLY_INFERRED`  
**Temporal Scope:** Point-in-time — binary match between fields of current transaction  
**Evidence:** Described as boolean match flags (e.g., 'does billing name match card name?', 'does billing address match shipping address?'). These compare fields within the same transaction, so no future information is involved. T/F encoded.  
**Action:** KEEP — treat as categorical after encoding

### 12. V1–V339 (Vesta proprietary, fully anonymized)
**Tier:** 🟠 `UNKNOWN_OPAQUE`  
**Temporal Scope:** UNKNOWN — Vesta has not publicly documented V-column construction. Some V-columns may be velocity checks (transactions per hour/day), device fingerprint risk scores, or historical pattern signals. Some may be point-in-time; others may use short look-back windows.  
**Evidence:** No public documentation available. Competition forums note that V-columns are 'Vesta-engineered features'. The fact that many have >80% missingness and appear in discrete bands suggests they are complex risk signals, not simple measurements. High-missingness subset (>80%) already dropped.  
**Action:** KEEP for E1 baseline (as Kaggle convention and prior literature does). MUST be reported as 'Unknown provenance — temporal leakage cannot be formally ruled out' in any paper. Consider V-column ablation experiment before claiming final results. Tier: UNKNOWN_OPAQUE.

### 13. id_01–id_38 (anonymized device/network features)
**Tier:** 🟠 `UNKNOWN_OPAQUE`  
**Temporal Scope:** Mostly point-in-time (device/browser fingerprint at transaction time). Some id-columns may be device-level aggregates.  
**Evidence:** Competition documentation says identity table provides 'digital signature' information. Device type, browser, screen resolution, etc. are point-in-time. Some id-columns (id_31–id_38) appear to be numeric features with unknown construction — possibly aggregated behavioral signals.  
**Action:** KEEP — but id_31–id_38 flagged as UNKNOWN_OPAQUE for paper documentation

### 14. DeviceType, DeviceInfo
**Tier:** 🟡 `STRONGLY_INFERRED`  
**Temporal Scope:** Point-in-time — device type and model at transaction time  
**Evidence:** DeviceType (mobile/desktop) and DeviceInfo (device model string) are clearly per-transaction device attributes. No temporal dependency.  
**Action:** KEEP

### 15. hour_sin, hour_cos (engineered)
**Tier:** ✅ `KNOWN`  
**Temporal Scope:** Point-in-time — cyclical encoding of transaction hour  
**Evidence:** Computed from TransactionDT by our pipeline. hour = (TransactionDT // 3600) % 24. Strictly point-in-time; no future information.  
**Action:** KEEP

### 16. is_missing_* (engineered missingness flags)
**Tier:** ✅ `KNOWN`  
**Temporal Scope:** Point-in-time — binary flag indicating field is missing  
**Evidence:** Computed from per-transaction missingness pattern. Whether a field is missing is determined at transaction time. Missingness indicators fit only on training data (fit on column-level presence/absence — no statistics required). No leakage.  
**Action:** KEEP

---

## Recommendations for Paper Writing

1. **V1–V339:** State explicitly: *'Vesta proprietary signals with unknown construction. Temporal leakage cannot be formally ruled out. Used as per Kaggle competition convention and prior literature.'*

2. **C1–C14:** State: *'Inferred backward-looking counting aggregates. Exact temporal scope is not formally documented. A V-column and C-column ablation experiment is planned to assess sensitivity.'*

3. **D1:** State: *'Inferred time-delta feature. Lower D1 is strongly correlated with fraudulent transactions (fraud mean: 38.7 days vs. legitimate mean: 95.6 days). Causal interpretation is a hypothesis, not verified ground truth.'*

4. **id_31–id_38:** State: *'Anonymized features from identity table with unknown construction. Classified as UNKNOWN_OPAQUE.'*

*Generated by: src/audit/feature_provenance.py | Phase 1.5C*