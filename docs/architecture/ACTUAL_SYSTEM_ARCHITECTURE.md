# ACTUAL SYSTEM ARCHITECTURE CLASSIFICATION

**Repository:** `adaptive-upi-fraud-detection`  
**Current Branch:** `Upto_Phase-4` (`c78494e`)  
**Audit Classification:** Category D — **Research Pipeline + Independent Streaming Infrastructure**

---

## 1. Architectural Classification Verdict

Based exclusively on verified code implementations, the repository is classified as:

### 🏷️ Category D: Research Pipeline + Independent Streaming Infrastructure

### Rationale:
1. **The Research Track (Phases 1–4) is Fully Operational Offline:**
   - E1 LightGBM baseline operates on 406 tabular features with frozen weights and calibrated threshold `0.616521`.
   - E2b Temporal GRU operates on 5-step transaction history sequences.
   - E3 explores offline hybrid fusion architectures.
   - Phase 4 implements concept drift monitoring (PSI) and adaptive retraining on BAF.
2. **The Real-Time Infrastructure Track (Members 1 & 2) is Fully Operational for Event Ingestion & Processing:**
   - Member 1 ingests, throttles, and partitions transaction streams into Kafka topics across a 3-node KRaft cluster.
   - Member 2 Spark runs Parquet conversions (7.25x scan acceleration) and Structured Streaming micro-batch aggregations.
   - Member 2 Flink runs event-time sliding windows (5m/10m) and sub-second CEP anomaly alerting.
3. **The Two Tracks Coexist in Decoupled Parallelism:**
   - Member 2 Spark and Flink do **NOT** invoke the E1 model weights (`model.txt`) or preprocessor (`preprocessing.joblib`).
   - The streaming infrastructure does **NOT** generate the 406 tabular features required by E1.
   - The system is therefore **NOT** "Fully integrated real-time ML fraud detection" (Category A) nor "Partially integrated real-time ML pipeline" (Category C).
   - It is a **dual-track system** where data engineering streaming infrastructure and machine learning research models are independently complete and co-located within the repository. The architecture leaves a future extension point for an isolated online inference adapter; such an adapter is outside the current Member 1/Member 2 scope.

---

## 2. Dual-Track Component Topology

```text
===========================================================================================
RESEARCH TRACK (Offline / Frozen Baseline / Complete)
===========================================================================================
  Datasets/IEEE-CIS (train_transaction.csv)
          │
          ▼
  src/features/pipeline.py (406 Tabular Features)
          │
          ├──────────────────────────┐
          ▼                          ▼
  experiments/E1_lightgbm    experiments/phase4_drift_adaptation
  - model.txt (Booster)      - BAF Base (Months 0–7)
  - preprocessing.joblib     - PSI Monitoring (threshold 0.10)
  - PR-AUC: 0.5267           - Adaptive retraining (v1 -> v2 -> v3)
  - Optimal Thresh: 0.616521 - Bootstrap validation
          │
          ▼
  phase2_gru/ (E2b GRU)
  - gru_best.pt (L=5 sequences)
  - standard_scaler.pkl

===========================================================================================
REAL-TIME INFRASTRUCTURE TRACK (Streaming / Event Processing / Complete)
===========================================================================================
  Transaction Replay / Generators
          │
          ▼
  Member 1: Apache Kafka 3.8 (cluster/ & kafka/)
  - KRaft 3-node cluster configs (server-[1-3].properties)
  - Partition key: card_id (6 partitions)
  - Semantics: acks=all, enable.idempotence=True
          │
          ▼
  Kafka Topic: 'fraud-transactions' (JSON UTF-8)
          │
     ┌────┴───────────────────────────┐
     │                                │
     ▼                                ▼
  Member 2: Apache Spark 3.5.9   Member 2: Apache Flink 2.2
  (spark/)                       (flink/)
  - 7.25x Parquet optimizer      - PyFlink DataStream CEP
  - 5-min sliding window         - 5m/10m sliding event windows
  - amt_to_mean_ratio            - Velocity ratio (VR = Amt / avg_10m)
  - Sink: Parquet Lakehouse      - Sink: Kafka 'fraud-alerts'

===========================================================================================
BRIDGE LAYER (Planned for Member 3 / Future Scope)
===========================================================================================
  [Online Inference Adapter: Kafka 'fraud-features' -> 406 Features -> E1 Model -> Score]
```
