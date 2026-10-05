# Architectural Specification: Online Feature Hydration Adapter for E1 Serving

**Component:** Online Feature Hydration Adapter  
**Layer:** Integration between Streaming Infrastructure (M1/M2) and ML Serving (M3)  
**Status:** Architectural Design Specification  
**Host Repository:** `adaptive-upi-fraud-detection`  
**Target Branch:** `integration/member3-serving`  
**Auditor:** Senior ML Research, MLOps & Systems Integration Engineer  

---

## 1. Problem Statement & Motivation

The Stage-A audit of Member 3 identified a critical architectural gap:
- **Member 1 (Kafka)** produces compact transaction events containing 6 to 10 operational fields (e.g. `transaction_id`, `card_id`, `amount`, `event_time`, `merchant_id`, `device_type`).
- **Member 2 (Flink)** computes real-time streaming velocity features over 5m/10m sliding windows (e.g. `transaction_count_5m`, `total_amount_5m`, `transaction_velocity_ratio`).
- **The Champion E1 LightGBM Model** strictly requires an input vector of **exactly 406 engineered tabular features** derived from the raw IEEE-CIS dataset schema (containing cardholder profile details, identity attributes, 14 count variables $C$, 15 timedelta variables $D$, 9 match flags $M$, and 339 anonymized V-features).

If a compact streaming event is fed directly into Member 3's existing serving wrapper without an intermediate hydration layer:
$$\text{Compact Kafka Event (6 fields)} \xrightarrow{\text{ServingPreprocessor}} \text{Reindex with NaN (388 fields)} \xrightarrow{\text{Imputation}} \text{Population Medians}$$

This constitutes **fallback imputation / feature completion**, not **true feature hydration**. Scoring transactions where 95% of features are imputed medians fundamentally destroys the predictive validity of E1 and yields scientifically untrustworthy fraud probabilities.

To bridge this gap while preserving research integrity, this document specifies the **Online Feature Hydration Adapter**.

---

## 2. System Architecture & Information Flow

```text
  ┌─────────────────────────────────────────────────────────────┐
  │                    MEMBER 1: KAFKA LAYER                    │
  │  Topic: fraud-transactions                                  │
  │  Payload: { transaction_id, card_id, amount, event_time,    │
  │             merchant_id, device_type, country }             │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                    MEMBER 2: FLINK LAYER                    │
  │  Stateful Sliding Windows (5m / 10m)                        │
  │  Output: { count_5m, amount_5m, velocity_ratio, ... }       │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │            ONLINE FEATURE HYDRATION ADAPTER (NEW)           │
  │                                                             │
  │  ┌───────────────────────────────────────────────────────┐  │
  │  │ 1. Entity Profile Store (Cardholder Identity Cache)   │  │
  │  │    Lookup: card1..card6, addr1..addr2, P_emaildomain, │  │
  │  │            R_emaildomain, DeviceInfo, id_01..id_38    │  │
  │  └───────────────────────────────────────────────────────┘  │
  │  ┌───────────────────────────────────────────────────────┐  │
  │  │ 2. Point-in-Time Behavioral Aggregator               │  │
  │  │    Reconstruct: C1..C14 (counts), D1..D15 (deltas),   │  │
  │  │                 M1..M9 (match flags), V1..V339        │  │
  │  └───────────────────────────────────────────────────────┘  │
  │  ┌───────────────────────────────────────────────────────┐  │
  │  │ 3. Streaming Metric Synthesis                         │  │
  │  │    Map Flink velocity ratio to dynamic risk signals   │  │
  │  └───────────────────────────────────────────────────────┘  │
  └──────────────────────────────┬──────────────────────────────┘
                                 │ Hydrated IEEE-CIS Transaction (394 raw cols)
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                MEMBER 3: FASTAPI SERVING ENGINE             │
  │                                                             │
  │  1. ServingPreprocessor.transform()                         │
  │     - Generate cyclic temporal features (hour_sin/cos)      │
  │     - Apply fitted IEEECISPreprocessor                      │
  │     - Strictly validate 406 columns matching schema         │
  │     - Assert zero target leakage (isFraud NOT in X)         │
  │  2. LightGBM Booster (experiments/E1_lightgbm/model.txt)    │
  │  3. Decision Rule: prob >= 0.616521 ? "FRAUD" : "LEGIT"     │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
                     FINAL VERDICT & METRICS
```

---

## 3. Data Entities & State Stores

The Hydration Adapter relies on two logical feature stores:

### 3.1 Entity Profile Store (Cardholder & Device Cache)
Stores semi-static identity attributes associated with the primary entity partition key (`card_id` or `user_id`):
- **Card Attributes:** `card1` (issuer bank), `card2` (card number series), `card3` (country), `card4` (card network: visa, mastercard), `card5` (category), `card6` (type: debit, credit).
- **Address & Geo Attributes:** `addr1` (billing region), `addr2` (billing country), `dist1` (distance between billing and transaction).
- **Digital Identity:** `P_emaildomain`, `R_emaildomain`, `DeviceType`, `DeviceInfo`, `id_01` through `id_38`.

### 3.2 Point-in-Time Behavioral State Store
Maintains running temporal and counter states per entity key up to time $t - \epsilon$:
- **Counter Variables ($C_1 \dots C_{14}$):** Number of transactions associated with the cardholder across various historical time horizons and merchants.
- **Timedelta Variables ($D_1 \dots D_{15}$):** Elapsed days/seconds between the current transaction and previous activity (e.g. days since first transaction, days since previous card transaction).
- **Match Flags ($M_1 \dots M_9$):** Name and address match verifications.
- **V-Features ($V_1 \dots V_{339}$):** Vesta payment network aggregated behavioral indicators.

---

## 4. Point-in-Time Correctness & Leakage Safeguards

### Absolute Rules on Temporal & Target Integrity:
1. **Zero Future Lookahead:** When transaction $e_t$ arrives at timestamp $T$, the behavioral counters ($C$) and timedeltas ($D$) must be computed strictly from transactions where $t_{\text{prev}} < T$. Future transactions must never contaminate the state.
2. **State Mutation Post-Scoring:** The entity state in the behavioral store is updated **only after** the feature vector for $e_t$ has been generated and scored, or within an atomic read-then-update transaction.
3. **Zero Target Leakage:** The ground-truth label `isFraud` (or `is_fraud`) is **strictly forbidden** from the Hydration Adapter's input, state store, and output feature frame. If present in a testing harness, it is passed strictly out-of-band as evaluation metadata.

---

## 5. Dual Operational Modes

To satisfy both scientific benchmarking and real-time streaming integration, the Adapter supports two operational modes:

### Mode A: Historical Replay / Benchmark Mode
Used for evaluating offline vs online serving consistency against historical test splits (e.g. Phase 4 consistency on IEEE-CIS test split):
- The adapter ingests full historical transaction records from the test set (`train_transaction.csv` / `train_identity.csv`).
- It extracts the compact subset to simulate the streaming event, queries the historical store to hydrate back the true point-in-time features, and verifies that the output matches the offline model prediction with 0.000000 probability difference.

### Mode B: Real-Time Streaming Ingestion Mode
Used when connected to active Kafka and Flink streaming pipelines:
- Incoming Kafka events provide the live transaction data (`card_id`, `amount`, `event_time`).
- The adapter queries the `EntityProfileStore` for known cards. If a card is unknown (cold-start / new user), it assigns safe domain-default profile markers and logs a cold-start event.
- Real-time Flink metrics (`transaction_count_5m`, `transaction_velocity_ratio`) directly update the short-term behavioral state ($C_1$, $C_2$, $D_1$).
- A fully populated raw IEEE-CIS record is synthesized and forwarded to Member 3's `POST /predict` endpoint.

---

## 6. Interface & Wire Specification

### Input to Hydration Adapter (from Kafka/Flink):
```json
{
  "transaction_id": "TX-3544193",
  "card_id": "CARD-13926",
  "amount": 100.00,
  "event_time": 1791212210000,
  "merchant_id": "M-W",
  "device_type": "mobile",
  "country": "US",
  "velocity_features": {
    "count_5m": 3,
    "amount_5m": 250.00,
    "velocity_ratio": 1.25
  }
}
```

### Output from Hydration Adapter (to ServingPreprocessor):
A single-row `pandas.DataFrame` or JSON payload containing all expected raw IEEE-CIS columns:
- `TransactionID`: `"TX-3544193"`
- `TransactionDT`: `13392100.0`
- `TransactionAmt`: `100.00`
- `ProductCD`: `"W"`
- `card1` .. `card6`: Hydrated cardholder attributes
- `addr1`, `addr2`, `dist1`: Hydrated geographical attributes
- `P_emaildomain`, `R_emaildomain`: Hydrated email domains
- `C1` .. `C14`: Reconstructed count features
- `D1` .. `D15`: Reconstructed timedelta features
- `M1` .. `M9`: Hydrated match flags
- `V1` .. `V339`: Hydrated payment network indicators
- `id_01` .. `id_38`: Hydrated identity attributes
- `DeviceType`, `DeviceInfo`: Hydrated device attributes

This hydrated DataFrame is passed directly to `ServingPreprocessor.transform()`, which produces the canonical, validated $1 \times 406$ matrix for LightGBM inference.

---

## 7. Implementation Checklist for Stage B

- [ ] Implement `serving/hydration/entity_store.py` (in-memory cardholder profile registry).
- [ ] Implement `serving/hydration/adapter.py` (`FeatureHydrationAdapter` class).
- [ ] Add unit tests in `serving/tests/test_feature_hydration.py` verifying:
  - 100% preservation of raw features when present.
  - Correct point-in-time calculation of $D_1$ (time delta).
  - Clean error handling for unknown entity keys.
  - Zero target leakage (`isFraud` never injected into model frame).
- [ ] Connect adapter to Member 3 `api/main.py` via a new `/predict/hydrated` endpoint or configurable middleware.
