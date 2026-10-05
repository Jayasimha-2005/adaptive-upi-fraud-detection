# ACTUAL DATA FLOW ARCHITECTURE

**Repository:** `adaptive-upi-fraud-detection`  
**Current Branch:** `Upto_Phase-4` (`c78494e`)  
**Audit Status:** [VERIFIED] from active codebase  

---

## 1. Verified Runtime Topology

Based on direct source inspection of `kafka/producer/`, `kafka/consumer/`, `spark/streaming/`, `flink/streaming/`, and `run_kafka_spark_flink_pipeline.py`, the actual runtime data flow is:

```text
                                TRANSACTION REPLAY / SIMULATION
                                 (IEEE-CIS / PaySim / Synthetic)
                                               │
                                               ▼
                              ┌─────────────────────────────────┐
                              │  Member 1: Kafka Producer       │ [VERIFIED]
                              │  - Key: card_id / user_id       │
                              │  - acks=all, idempotence=True   │
                              └────────────────┬────────────────┘
                                               │
                                               ▼
                              ┌─────────────────────────────────┐
                              │  Kafka Broker Ingestion Topic   │ [VERIFIED]
                              │  'fraud-transactions'           │
                              │  (or 'ieee_cis_transactions')   │
                              │  6 Partitions                   │
                              └────────┬───────────────┬────────┘
                                       │               │
                      ┌────────────────┴───┐       ┌───┴────────────────┐
                      │                    │       │                    │
                      ▼                    │       │                    ▼
    ┌───────────────────────────────────┐  │       │  ┌───────────────────────────────────┐
    │ Member 2: Apache Flink 2.2 CEP    │  │       │  │ Member 2: Apache Spark 3.5.9      │ [VERIFIED]
    ├───────────────────────────────────┤  │       │  ├───────────────────────────────────┤
    │ • Keyed Stream by card_id         │  │       │  │ • Structured Streaming            │
    │ • 5m & 10m Sliding Event Windows  │  │       │  │ • 5-minute sliding windows        │
    │ • Velocity Ratio: Amt / avg_10m   │  │       │  │ • amt_to_mean_ratio               │
    │ • Sub-second alert generation     │  │       │  │ • 7.25x Parquet Lakehouse Sink    │
    └─────────────────┬─────────────────┘  │       │  └─────────────────┬─────────────────┘
                      │                    │       │                    │
                      ▼                    │       │                    ▼
    ┌───────────────────────────────────┐  │       │  ┌───────────────────────────────────┐
    │ Kafka: 'fraud-alerts'             │  │       │  │ Parquet Storage:                  │ [VERIFIED]
    │ Kafka: 'fraud-features'           │  │       │  │ 'spark/output/streaming/features' │
    └───────────────────────────────────┘  │       │  └───────────────────────────────────┘
                                           │       │
                                           ▼       ▼
===========================================================================================
[VERIFIED DISCONNECTION]: DOWNSTREAM MODEL INFERENCE BOUNDARY
===========================================================================================
                                           │
                                           │  ❌ [NOT CONNECTED IN CODE]
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │ Frozen Phase 1 E1 LightGBM Model        │
                      │ - experiments/E1_lightgbm/model.txt     │ [VERIFIED FROZEN]
                      │ - Input: 406 Canonical Features         │
                      │ - Threshold: 0.616521                   │
                      └─────────────────────────────────────────┘
```

---

## 2. Evidence Classification

| Connection Link | Classification | Status & Code Evidence |
| :--- | :--- | :--- |
| **Replay $\to$ Kafka Producer** | **[VERIFIED]** | Proven by `kafka/producer/ieee_cis_replay_producer.py` reading dataset and producing JSON. |
| **Producer $\to$ Broker Topic** | **[VERIFIED]** | Proven by `kafka/producer/producer.py` using `broker="localhost:9092"`. |
| **Kafka $\to$ Spark Streaming** | **[VERIFIED]** | Proven by `spark/streaming/stream_processor.py` consuming from `fraud-transactions`. |
| **Kafka $\to$ Flink CEP** | **[VERIFIED]** | Proven by `flink/streaming/stream_processor.py` consuming from `fraud-transactions`. |
| **Spark $\to$ Flink** | **[NOT CONNECTED]** | Code proves Spark and Flink are parallel independent consumers; neither feeds the other. |
| **Flink $\to$ Spark** | **[NOT CONNECTED]** | Flink outputs to `fraud-features` / `fraud-alerts`; Spark does not consume Flink topics. |
| **Spark $\to$ E1 Model** | **[NOT CONNECTED]** | Zero calls to `model.txt` or `preprocessing.joblib` in `spark/`. |
| **Flink $\to$ E1 Model** | **[NOT CONNECTED]** | Zero calls to `model.txt` or `preprocessing.joblib` in `flink/`. |
| **Kafka $\to$ E1 Model** | **[NOT CONNECTED]** | Heuristic amount-tier scoring in `spark_streaming_consumer.py` is explicitly rule-based, not E1 ML inference. |

---

## 3. Summary of Actual vs Intended Architecture

- **Intended Architecture:** High-level project architecture diagrams intended for Kafka/Spark/Flink to serve as an event streaming backbone feeding an online model scoring engine.
- **Actual Implemented Code:** The real-time infrastructure layer successfully processes streaming events into aggregations and CEP alerts. However, the connection between this stream processing layer and the offline 406-feature LightGBM E1 model has **NOT yet been implemented in code**. E1 exists strictly as a frozen, validated offline research model.
