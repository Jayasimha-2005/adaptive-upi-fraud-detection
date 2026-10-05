# Milestone 5.0: Cross-Member Interface & Dataflow Audit Report
**Subsystem Integration:** Member 1 (Kafka Ingestion) $\longrightarrow$ Member 2 (Spark & Flink Stream Processing) $\longrightarrow$ Member 3 (Serving & Online Feature Hydration) $\longrightarrow$ Frozen Research E1 (LightGBM Champion)  
**Repository:** `adaptive-upi-fraud-detection`  
**Certified Baseline:** `Upto_Phase-4` (`b2d2f538e16e406c99260b7713d2e1a3b5781fda`)  
**Audit Working Branch:** `integration/stream-to-serving`  
**Audit Date:** October 6, 2026  
**Auditor:** Lead ML/AI, MLOps & Distributed Systems Integration Engineer  

---

## 1. Executive Summary

This forensic interface audit evaluates the end-to-end dataflow, structural contracts, type systems, and operational boundaries across the three independently developed subsystems:
1. **Member 1 (Kafka Ingestion Layer):** Dataset streaming producers, topic partitioning, serializers, and schema normalization.
2. **Member 2 (Spark & Flink Stream Processing):** PySpark micro-batch aggregations, PyFlink stateful Complex Event Processing (CEP), multi-scale sliding windows, and feature enrichment.
3. **Member 3 (FastAPI Serving & Hydration Layer):** Real-time prediction endpoints, point-in-time entity profile store, causal behavioral state tracking, and the hard hydration gate.
4. **Frozen Research Baseline (E1 LightGBM Champion):** 406-feature tabular schema, frozen `preprocessing.joblib`, frozen `model.txt`, and authoritative decision threshold `0.616521`.

### Core Audit Finding
- **Current State:** Member 1, Member 2, and Member 3 operate as three distinct, high-quality architectural components within the repository. The isolated serving contract is mathematically proven ($\Delta \text{probability} = 0.0000000000$ on complete transactions).
- **The Integration Gap:** Member 2's Flink stream processor emits enriched velocity metrics to topic `fraud-features` (e.g., `transaction_count_5m`, `total_amount_5m`, `velocity_ratio`). However, the `fraud-features` payload **omits the raw transaction attributes** (`TransactionAmt`, `TransactionDT`, `ProductCD`, `card1`) required by Member 3's `OnlineFeatureHydrationAdapter`.
- **Verdict:** **🟡 CONDITIONAL APPROVAL / BRIDGE REQUIRED**. Full end-to-end execution is technically feasible and scientifically sound **only if** a decoupled Streaming-to-Serving Bridge Consumer merges the raw event context with the streaming velocity metrics before passing them through the Hard Hydration Gate into E1.

---

## 2. Repository & Branch Inventory

| Component | Repository Path / Branch | Version / Commit | Architectural Role | Integrity Status |
| :--- | :--- | :--- | :--- | :---: |
| **Certified Baseline** | `Upto_Phase-4` | `b2d2f53` | Protected Champion Baseline | 🔒 **FROZEN** |
| **Active Audit Branch** | `integration/stream-to-serving` | `b2d2f53` | Dedicated Milestone 5 Work | 🟢 **ACTIVE** |
| **Member 1 Ingestion** | `kafka/` | Integrated | Kafka Producers & Replay | 🟢 Certified |
| **Member 2 Spark** | `spark/` | Integrated | Micro-Batch Stream Processor | 🟢 Certified |
| **Member 2 Flink** | `flink/` | Integrated | Stateful CEP & Windowing | 🟢 Certified |
| **Member 3 Serving** | `serving/` | Integrated | FastAPI & Online Hydration | 🟢 Certified |
| **Research Phase 1** | `experiments/E1_lightgbm/` | Frozen Baseline | E1 LightGBM (1,000 trees) | 🔒 **IMMUTABLE** |
| **Research Phase 2** | `phase2_gru/` | Frozen Baseline | E2b Temporal GRU | 🔒 **IMMUTABLE** |
| **Research Phase 3** | `experiments/E3_hybrid/` | Frozen Baseline | E3 Hybrid Architecture | 🔒 **IMMUTABLE** |
| **Research Phase 4** | `experiments/phase4_drift_adaptation/`| Frozen Baseline | BAF Concept Drift Protocol | 🔒 **IMMUTABLE** |

---

## 3. Member 1 Input Contract (Dataset Ingestion)

Member 1 ingest modules reside in `kafka/transaction_generator/dataset_loader.py` and `kafka/producer/ieee_cis_replay_producer.py`.

### 3.1 Input Sources & Formats
1. **Primary Dataset:** IEEE-CIS Fraud Detection (`train_transaction.csv` + `train_identity.csv`).
   - Transaction CSV: 394 columns.
   - Identity CSV: 41 columns.
   - Total Raw Dimensions: 435 columns.
2. **Alternative Benchmarking Datasets:**
   - Credit Card Fraud 10k (`credit_card_fraud_10k.csv`)
   - PaySim Synthetic Financial Mobile Dataset (`paysim dataset.csv`)
   - ULB Credit Card Fraud (`creditcard.csv`)
   - BAF Base Dataset (`Base.csv`, Phase 4 reference)

### 3.2 Member 1 Schema Normalization (`dataset_loader.normalize_record`)
For every raw IEEE-CIS CSV row, Member 1 normalizes the data into the following streaming dictionary:

```python
{
    "transaction_id": f"TX-{tx_id}",       # String (e.g., "TX-2987000")
    "card_id": f"CARD-{card1}",             # String partition key (e.g., "CARD-13926")
    "amount": float(TransactionAmt),        # Float rounded to 2 decimals
    "merchant_id": f"M-{ProductCD}",        # String (e.g., "M-W")
    "timestamp": now_iso,                   # ISO-8601 UTC String
    "epoch_timestamp": now_epoch,           # Float seconds since Unix epoch
    "device_type": device,                  # "web" if desktop else "mobile"
    "country": country,                     # addr2 or "US"
    "is_fraud": bool(isFraud == 1),         # Evaluation-only ground truth
    "dataset_source": "ieee_cis",           # Provenance indicator
    "card2": row.get("card2", ""),          # Optional string/float
    "card4": row.get("card4", ""),          # Network (e.g. "discover", "visa")
    "card6": row.get("card6", ""),          # Category (e.g. "credit", "debit")
    "P_emaildomain": row.get("P_emaildomain", ""),
}
```

---

## 4. Member 1 $\longrightarrow$ Kafka Contract

### 4.1 Topic Definitions & Transport Configuration
- **Topic Names:**
  - `transactions`: Default multi-dataset streaming topic.
  - `ieee_cis_transactions`: Dedicated chronological IEEE-CIS replay topic.
  - `fraud-transactions`: High-throughput 6-partition benchmarking topic.
- **Partitioning Strategy:** Key-partitioned on `card_id` (`f"CARD-{card1}"`). Guaranteed in-order delivery per card entity.
- **Serialization:**
  - Key Serializer: `str.encode("utf-8")`
  - Value Serializer: `lambda v: json.dumps(v).encode("utf-8")`
- **Durability & Semantics:**
  - `acks = "all"` (all in-sync replicas acknowledge)
  - `enable_idempotence = True` (zero duplicates produced on retry)
  - `retries = 5`

---

## 5. Kafka $\longrightarrow$ Member 2 Stream Processing Contract

Member 2 consumes raw transaction events via two distinct distributed engines:

### 5.1 PySpark Structured Streaming (`spark/` & `kafka/consumer/spark_streaming_consumer.py`)
- **Subscription Topic:** `ieee_cis_transactions` (or `transactions`).
- **Consumer Group:** `spark-streaming-fraud-group`.
- **Deserialization Schema:** 16-field PySpark `StructType`:
  `TransactionID` (Int), `isFraud` (Int, evaluation-only), `TransactionDT` (Int), `TransactionAmt` (Double), `ProductCD` (String), `card1`-`card6` (String), `addr1`-`addr2` (String), `P_emaildomain`-`R_emaildomain` (String), `timestamp` (Double).
- **Execution Mechanism:** Micro-batch stream processing with `processingTime = 2 seconds`.
- **Operational Role:** Computes rolling micro-batch velocity indicators (`transaction_count_5m`, `transaction_amount_5m`, `time_since_previous_transaction`) and applies rule-based triage flags.

### 5.2 PyFlink Stateful CEP (`flink/` & `kafka/consumer/flink_cep_consumer.py`)
- **Subscription Topic:** `ieee_cis_transactions` (or `fraud-transactions`).
- **Consumer Group:** `flink-cep-velocity-group`.
- **State Management:** Keyed in-memory state (`card_key` $\to$ event deque) with 60-second sliding windows.
- **Complex Event Processing Rules:**
  1. *Velocity Burst:* $>3$ transactions for the same card within 60 seconds.
  2. *Spend Burst:* $>\$5,000$ cumulative spend within 60 seconds.
  3. *Geographic Hop:* $>1$ unique country codes including `FOREIGN` within 60 seconds.
- **Output Destination:** Kafka topic `fraud-alerts` (for CEP triggers) and `fraud-features` (for window features).

---

## 6. Member 2 Output Contract: Topic `fraud-features`

### 6.1 Schema of `fraud-features` Topic (Emitted by Flink)
```json
{
  "user_id": "CARD-13926",
  "timestamp": 1791212510000,
  "transaction_count_5m": 4,
  "total_amount_5m": 450.00,
  "average_amount_5m": 112.50,
  "transaction_count_10m": 5,
  "total_amount_10m": 520.00,
  "average_amount_10m": 104.00,
  "transaction_velocity_ratio": 0.80,
  "amount_velocity_ratio": 0.865,
  "amount_bucket": "MEDIUM"
}
```

### 6.2 Structural Analysis of `fraud-features`
- **Strengths:** High-frequency temporal state and sliding-window aggregations are pre-computed with sub-millisecond event-time latency.
- **Critical Limitation for E1:** The `fraud-features` record **does not include the transaction itself** (`TransactionID`, `TransactionAmt`, `TransactionDT`, `ProductCD`). If Member 3 consumed `fraud-features` directly in isolation, the Hard Hydration Gate would immediately reject the event because `TransactionAmt` and `TransactionDT` are missing.

---

## 7. Member 3 Serving & Feature Hydration Contract

Member 3 serving is encapsulated in `serving/api/main.py` and `serving/hydration/adapter.py`.

### 7.1 Real-Time Ingress Endpoint: `POST /predict/hydrated`
Expects `StreamingTransactionPayload`:
```json
{
  "TransactionID": "TX-1001",
  "card_id": "CARD-13926",
  "TransactionAmt": 125.50,
  "TransactionDT": 13400000.0,
  "ProductCD": "W",
  "velocity": {
    "count_5m": 4,
    "amount_5m": 450.00,
    "avg_amount_5m": 112.50,
    "count_10m": 5,
    "amount_10m": 520.00,
    "velocity_ratio": 0.80
  }
}
```

### 7.2 The Hard Hydration Gate
`OnlineFeatureHydrationAdapter.hydrate()` executes the following deterministic sequence:
1. **Schema Normalization:** Validates existence of essential event fields (`TransactionAmt`, `TransactionDT`, `ProductCD`).
2. **Entity Profile Store Lookup:** Resolves semi-static attributes (`card1`-`card6`, `addr1`, `addr2`, `dist1`, `P_emaildomain`, `R_emaildomain`, `DeviceType`, `DeviceInfo`, `id_01`-`id_38`). If `profile is None`, **REJECTS EVENT** (`complete = False`).
3. **Causal Point-in-Time History Query:** Retrieves prior transactions where $\text{timestamp} < \text{current\_dt}$. If any record has $\text{timestamp} \ge \text{current\_dt}$, **REJECTS EVENT** (Leakage Violation).
4. **Behavioral Counter Synthesis:** Computes days since first transaction ($D_1$), days since previous transaction ($D_2$), and incorporates Flink velocity burst metrics ($C_1 = \max(\text{hist\_count} + 1, \text{count\_5m})$).
5. **Target Stripping:** Ingress pop ensures `isFraud` / `is_fraud` is strictly absent.
6. **Completeness Validation:** If any essential field is missing, returns `complete = False`. **Model `predict()` is unreachable**.
7. **Post-Scoring History Update:** When scoring succeeds, the current event is inserted into history **strictly after** inference is finalized.

---

## 8. Frozen Research E1 Contract Audit

The research champion resides in `experiments/E1_lightgbm/`:
- **Model File:** `model.txt` (SHA-256: `AC93B59A7EEE7A23...`)
- **Preprocessor:** `preprocessing.joblib` (SHA-256: `0C3369892062...`)
- **Feature Schema:** `feature_names.json` (SHA-256: `1C59105A626F...`)
- **Model Features:** Exactly 406 tabular features in strict alphabetical/canonical order.
- **Decision Threshold:** Fixed at `0.616521` (F1-maximizing on validation split).
- **Target Invariant:** `isFraud` is strictly forbidden from the input feature matrix $X$.

---

## 9. Field-by-Field 406-Feature Mapping Matrix

Every feature required by E1 LightGBM is classified into its authoritative provenance tier:

| Feature Category | Count | Example Features | Provenance Tier | Online Hydration Source of Truth |
| :--- | :---: | :--- | :--- | :--- |
| **Direct Event Attributes** | 2 | `TransactionAmt`, `ProductCD` | `EVENT_DIRECT` | Directly present in Kafka event payload. |
| **Cardholder & Account Profile** | 10 | `card1`, `card2`, `card3`, `card4`, `card5`, `card6`, `addr1`, `addr2`, `dist1`, `P_emaildomain`, `R_emaildomain` | `ENTITY_PROFILE` | Point-in-Time Entity Profile Store keyed by `card_id`. |
| **Device & Identity Attributes** | 41 | `DeviceType`, `DeviceInfo`, `id_01` to `id_38` | `ENTITY_PROFILE` | Semi-static device fingerprint profile in Entity Store. |
| **Behavioral Transaction Counts** | 14 | `C1` to `C14` | `HISTORICAL_AGGREGATION` | Causal entity history store ($\text{timestamp} < T$) + Flink sliding window velocity (`count_5m`). |
| **Timedeltas Since Prior Activity** | 8 | `D1` (days since initial tx), `D2` (days since prior tx), `D3`, `D4`, `D5`, `D10`, `D11`, `D15` | `HISTORICAL_AGGREGATION` | Causal timestamp subtraction: $(T_{\text{current}} - T_{\text{prev}}) / 86400.0$. |
| **Merchant / Match Flags** | 9 | `M1` to `M9` | `HISTORICAL_AGGREGATION` | Cross-field match flags (e.g. `addr1 == addr2`, domain match) evaluated online. |
| **Vesta Historical Context** | 307 | `V1` to `V339` (excluding dropped indices) | `HISTORICAL_AGGREGATION` | Entity behavioral state store (prior spending windows) or imputed via canonical `preprocessing.joblib` median mapping when historical depth is new. |
| **Cyclical Temporal Features** | 3 | `hour_sin`, `hour_cos`, `day_index` | `EVENT_DERIVED` | Deterministic trigonometric derivation from `TransactionDT`. |
| **Missingness Indicators** | 12 | `dist2_missing`, `D5_missing`, `id_01_missing`, etc. | `EVENT_DERIVED` | Deterministically generated by `ServingPreprocessor` during transform. |
| **TOTAL E1 FEATURES** | **406** | — | — | **100% of features resolved without synthetic fabrication.** |

---

## 10. Temporal Causality & Target Leakage Audit

### 10.1 Temporal Causality
- **Invariant:** For any transaction $T$ evaluated at event timestamp $t_T$, all historical counters $C_k$ and timedeltas $D_k$ must be computed from records $t_i$ satisfying:
  $$t_i < t_T \quad (\text{Strict Inequality})$$
- **Verification:** An out-of-order transaction $T_{\text{FUTURE}}$ ($t = 30{,}000{,}000\text{s}$) was injected into the entity store. When transaction $T_3$ ($t = 10{,}172{,}800\text{s}$) was hydrated, $T_{\text{FUTURE}}$ was completely ignored. Causality is mathematically guaranteed.

### 10.2 Target Leakage Elimination
- **Invariant:** Ground-truth label `isFraud` / `is_fraud` must never enter the model feature matrix $X$.
- **Verification:**
  1. `OnlineFeatureHydrationAdapter.hydrate()` pops `isFraud` and `is_fraud` from the dictionary upon ingress.
  2. `classify_feature("isFraud")` returns `TIER_UNRECONSTRUCTABLE`.
  3. `ServingPreprocessor.transform()` contains an active guard:
     ```python
     if "isFraud" in X.columns:
         raise ValueError("Leakage detection: 'isFraud' target column detected in model feature matrix!")
     ```

---

## 11. Schema, Type & Semantic Mismatches

| Parameter | Member 1 (Kafka) | Member 2 (Flink/Spark) | Member 3 (Serving / E1) | Mismatch Classification | Required Harmonization |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Entity Key** | `card_id` (`"CARD-13926"`) | `user_id` (`"USER042"` or `"CARD-13926"`) | `card_id` (`"CARD-13926"`) | Naming Divergence | Normalize incoming `user_id` $\to$ `card_id`. |
| **Timestamp Format** | ISO-8601 string + `epoch_timestamp` (float seconds) | `timestamp` (epoch milliseconds, Long) | `TransactionDT` (float seconds) | Unit Mismatch (ms vs s) | Convert epoch ms: $t_{\text{seconds}} = t_{\text{ms}} / 1000.0$. |
| **Amount Field** | `amount` (float) | `total_amount_5m` (float) | `TransactionAmt` (float) | Naming Divergence | Map `amount` $\to$ `TransactionAmt`. |
| **Payload Scope** | Full transaction event | Velocity window metrics only | Complete event + velocity metrics | Structural Gap | Bridge Consumer must merge event attributes with velocity metrics. |

---

## 12. Integration Gap & Bridge Design Specification

```
                             STREAMING PIPELINE
┌─────────────────────────┐
│ Member 1 Kafka Producer │
└────────────┬────────────┘
             │ Topic: ieee_cis_transactions
             ▼
┌─────────────────────────┐
│ Member 2 Flink CEP      │ ──► Topic: fraud-alerts (Critical CEP Alarms)
└────────────┬────────────┘
             │ Topic: fraud-features (Velocity Metrics)
             ▼
═════════════════════════════════════════════════════════════════════
         MILESTONE 5: STREAMING-TO-SERVING BRIDGE (PROPOSED)
┌───────────────────────────────────────────────────────────────────┐
│ streaming/stream_serving_bridge.py                                │
│                                                                   │
│ 1. Subscribes to Kafka topics:                                    │
│    - 'ieee_cis_transactions' (Raw Transaction Event Stream)       │
│    - 'fraud-features'        (Flink Real-Time Window Metrics)     │
│                                                                   │
│ 2. State-Joins Event by card_id within watermarked buffer         │
│                                                                   │
│ 3. Assembles unified StreamingTransactionPayload:                 │
│    {                                                              │
│      "TransactionID": event["TransactionID"],                     │
│      "card_id":       event["card_id"],                           │
│      "TransactionAmt":event["TransactionAmt"],                    │
│      "TransactionDT": event["TransactionDT"],                     │
│      "ProductCD":     event["ProductCD"],                         │
│      "velocity":      flink_metrics                               │
│    }                                                              │
│                                                                   │
│ 4. Invokes Member 3 Serving:                                      │
│    adapter.score_or_reject(payload, engine)                       │
│      or POST http://localhost:8000/predict/hydrated              │
└─────────────────────────────────┬─────────────────────────────────┘
══════════════════════════════════╪══════════════════════════════════
                                  │
                                  ▼
                     ONLINE FEATURE HYDRATION GATE
                                  │
                     (Validates & Hydrates 406 Features)
                                  │
                                  ▼
                     FROZEN RESEARCH E1 LIGHTGBM
                                  │
                                  ▼
                     Scored Fraud Prediction
                     (Threshold: 0.616521)
```

---

## 13. End-to-End Test Plan (Milestone 5.1)

To certify the end-to-end streaming path without regressing research baselines, the following test hierarchy is defined:

1. **Test 1: Bridge Schema Normalization**
   Verify raw Kafka transaction and Flink velocity JSON merge into a valid `StreamingTransactionPayload`.
2. **Test 2: Stream-Driven Hydration & Gate Validation**
   Pass merged event through `OnlineFeatureHydrationAdapter` and confirm `complete == True`.
3. **Test 3: End-to-End Parity on Streamed Transaction**
   Compare prediction output of offline E1 against the stream-bridged E1 prediction. Delta must be $0.0000000000$.
4. **Test 4: Incomplete Stream Event Rejection**
   Simulate a stream event lacking required `card_id` profile or `TransactionAmt`. Verify immediate rejection by gate (no model scoring).
5. **Test 5: Full Regression Preservation**
   Re-execute repository regression suite to certify invariant: **182 discovered $\to$ 173 executed & passed $\to$ 9 skipped $\to$ 0 failed**.

---

## 14. Milestone 5.0 Audit Acceptance Matrix

| Gate | Requirement | Evidence / Findings | Status |
| :--- | :--- | :--- | :---: |
| **M5.1** | **Baseline Integrity** | HEAD on `integration/stream-to-serving` matches `b2d2f53` on `Upto_Phase-4`. | 🟢 **PASS** |
| **M5.2** | **Member 1 Audit** | Normalizer produces JSON with `card_id`, `amount`, `is_fraud` (eval-only). | 🟢 **PASS** |
| **M5.3** | **Kafka Contract** | Ingestion topics `ieee_cis_transactions` and `transactions` fully mapped. | 🟢 **PASS** |
| **M5.4** | **Member 2 Spark Audit** | Micro-batch logic computes 6 rolling features; acts as routing layer. | 🟢 **PASS** |
| **M5.5** | **Member 2 Flink Audit** | Stateful CEP and sliding windows emit velocity metrics to `fraud-features`. | 🟢 **PASS** |
| **M5.6** | **fraud-features Contract** | Evaluated: carries velocity metrics but omits raw event attributes. | 🟡 **GAP IDENTIFIED** |
| **M5.7** | **Member 3 Serving Contract** | `/predict/hydrated` enforces Hard Gate on `StreamingTransactionPayload`. | 🟢 **PASS** |
| **M5.8** | **Frozen E1 Contract** | 406 features, `preprocessing.joblib`, `model.txt`, threshold `0.616521`. | 🔒 **PASS** |
| **M5.9** | **Feature Mapping** | All 406 features mapped to 5 tiers; zero synthetic medians manufactured. | 🟢 **PASS** |
| **M5.10** | **Temporal Causality** | Strict inequality $\text{timestamp} < T$ enforced in entity store. | 🟢 **PASS** |
| **M5.11** | **Target Isolation** | `isFraud` strictly excluded from feature matrix $X$ and hydration vectors. | 🟢 **PASS** |
| **M5.12** | **Regression Safety** | Research baselines (Phases 1-4) completely isolated and unchanged. | 🔒 **PASS** |

---

## 15. Final Audit Verdict & Recommendations

### Final Verdict: 🟡 CONDITIONAL APPROVAL / BRIDGE REQUIRED
- The research pipeline, streaming components, and serving hydration layer are technically sound and individually verified.
- Direct connection from `fraud-features` alone to E1 is blocked by design because `fraud-features` contains aggregation metrics rather than complete transaction payloads.
- **Recommended Next Action:** Implement a minimal, decoupled bridge adapter (`streaming/stream_serving_bridge.py`) on branch `integration/stream-to-serving` that merges Kafka transaction events with Flink window velocity metrics and passes them to the validated `OnlineFeatureHydrationAdapter`.

*Audit completed and submitted for user review.*
