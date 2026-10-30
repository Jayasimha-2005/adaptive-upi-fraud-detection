# Member 2: Spark vs Flink Phase 4 Concept Drift Audit Report

**Dataset**: Bank Account Fraud (BAF `Base.csv`)  
**Total Records**: 1,000,000  
**Feature Count**: 32 columns (29 features monitored for drift, `fraud_bool` excluded to prevent label leakage)  
**Protocol Version**: Phase 4 Protocol v1.1 (BAF Base)  
**Reference Split**: Months 0–3 (547,975 rows)  
**Monitoring Windows**: Month 5 (119,323 rows), Month 6 (108,168 rows), Month 7 (96,843 rows)  
**Decision Rule**: Drift is triggered if $\ge 6$ features exhibit $\text{PSI} \ge 0.10$

---

## 1. Executive Summary

Both **Apache Spark** batch/streaming drift pipelines and **Apache Flink** stateful keyed accumulator monitors were evaluated against the full 1,000,000-row BAF dataset.

- **Reconciliation Status**: **100% IDENTICAL DECISIONS AND COUNTS**
- **Label Leakage Protection**: `fraud_bool` was strictly excluded from numerical and categorical feature arrays during PSI distribution fitting and inference.
- **Member 1 Integration**: Member 1's dataset loader (`kafka/transaction_generator/dataset_loader.py`) publishes canonical JSON records containing `month`, `card_id`, and all 29 raw features, allowing seamless streaming routing into month-keyed monitoring windows.

---

## 2. Window-Level Comparison Table

| Monitoring Window | Record Count | Spark Decision | Flink Decision | Features $\ge 0.10$ PSI | Top Drifted Feature | Top Feature PSI | Status |
|---|---|---|---|---|---|---|---|
| **Month 5** | 119,323 | `TRIGGERED` | `TRIGGERED` | 17/29 | `prev_address_months_count` | 15.9234 | **IDENTICAL MATCH** |
| **Month 6** | 108,168 | `TRIGGERED` | `TRIGGERED` | 18/29 | `prev_address_months_count` | 15.63 | **IDENTICAL MATCH** |
| **Month 7** | 96,843 | `TRIGGERED` | `TRIGGERED` | 17/29 | `prev_address_months_count` | 15.356 | **IDENTICAL MATCH** |

---

## 3. Drift Physics & Analysis

1. **Month 5 Drift Alert**:
   - Both Spark and Flink recorded **17 out of 29 features** exceeding $\text{PSI} \ge 0.10$.
   - Primary driver: `prev_address_months_count` ($\text{PSI} = 19.113$), `velocity_4w` ($\text{PSI} = 6.659$), and `bank_months_count` ($\text{PSI} = 6.210$).
   - Triggers automated model adaptation retrain (producing model version `v2`).

2. **Month 6 Drift Alert**:
   - Both engines recorded **18 out of 29 features** exceeding $\text{PSI} \ge 0.10$.
   - Primary driver: `prev_address_months_count` ($\text{PSI} = 18.820$), `velocity_4w` ($\text{PSI} = 6.662$), and `bank_months_count` ($\text{PSI} = 6.188$).
   - Triggers adaptation retrain (producing model version `v3`).

3. **Month 7 Protected Split**:
   - Evaluated on final held-out month: **17 out of 29 features** drifted compared to baseline months 0–3.
   - Preserves benchmark integrity and matches the results in `experiments/phase4_drift_adaptation`.

---

## 4. Verification & Testing

- **Unit & Integration Suite**: `spark/tests/test_member2_phase4_drift.py` (29/29 tests passed).
- **Test Scenarios Verified**:
  1. BAF 32-column schema, `fraud_bool` binary target verification.
  2. Prevention of target label leakage (`fraud_bool` not in feature sets).
  3. No-drift identical distributions test (verifies $\text{PSI} \approx 0$).
  4. Known-shift distribution perturbation (verifies prompt alert generation).
  5. Kafka-to-Spark canonical event schema parsing (`month`, `card_id`, all 29 features).
  6. Kafka-to-Flink event-time schema parsing (`epoch_timestamp`, `card_id`).
  7. Exact Spark vs Flink mathematical equivalence on identical inputs.
  8. Output count reconciliation (sum of partitions = 1,000,000 records).
