# Final Integrated Architecture: Dual-Track Research & Real-Time System

**Repository:** `adaptive-upi-fraud-detection`  
**Integrated Tracks:** Research Track (Phase 1–4) & Real-Time Infrastructure Track (Kafka, Spark, Flink)  
**Status:** Architecture Blueprint  
**Auditor / Architect:** Senior Software & Research Integration Engineer

---

## 1. System Philosophy & Dual-Track Separation

The repository unifies two distinct, complementary tracks of software and machine learning engineering:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                ADAPTIVE-UPI-FRAUD-DETECTION                            │
├───────────────────────────────────────────┬────────────────────────────────────────────┤
│           RESEARCH TRACK (IMMUTABLE)      │         REAL-TIME INFRASTRUCTURE TRACK     │
│   • Phase 1: LightGBM Baseline (E1)       │   • Member 1: Apache Kafka Streaming       │
│   • Phase 1.5: Robustness & Calibration   │   • Member 2: Apache Spark Batch/Streaming │
│   • Phase 2: Temporal GRU Modeling (E2)   │   • Member 2: Apache Flink Real-Time CEP   │
│   • Phase 3: Tabular-Temporal Hybrids (E3)│   • Integration: Multi-Process Pipeline    │
│   • Phase 4: Drift Adaptation on BAF      │   • Downstream Serving: Model Ingestion    │
└───────────────────────────────────────────┴────────────────────────────────────────────┘
```

### Architectural Guardrails:
1. **No Retroactive Rewriting:** Real-time components do NOT alter, retrain, or invalidate the scientific results of Phases 1–4.
2. **Distinct Feature Sets:** Real-time streaming features (e.g. 5m/10m velocity ratios) are **processing-only / future features** and are NOT retroactively injected into frozen E1 (which requires its canonical 406 IEEE-CIS features).
3. **No Target Leakage:** Evaluation labels (`isFraud`, `is_fraud`) are strictly evaluation-only metadata and never model input features.

---

## 2. End-to-End Component Architecture Diagram

```text
==========================================================================================
                                REAL-TIME SYSTEM TRACK
==========================================================================================

   [ Historical CSV Datasets / Replay ]        [ Synthetic Stream Generator ]
        (IEEE-CIS / PaySim / BAF)              (Rate: 100 - 5,000+ tx/sec)
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        │
                                        ▼
                      ┌───────────────────────────────────┐
                      │    Kafka Producer Ingestion       │
                      │  - Partition Key: card_id / user  │
                      │  - Durability: acks=all, retries=5│
                      │  - Codecs: LZ4 / Snappy / None    │
                      └─────────────────┬─────────────────┘
                                        │
                                        ▼
                      ┌───────────────────────────────────┐
                      │    Apache Kafka Cluster (KRaft)   │
                      │  Topic: fraud-transactions        │
                      │  Partitions: 3 / 6 / 12           │
                      └─────────┬───────────────────┬─────┘
                                │                   │
             ┌──────────────────┴──┐             ┌──┴───────────────────┐
             │                     │             │                      │
             ▼                     │             │                      ▼
┌─────────────────────────┐        │             │        ┌─────────────────────────┐
│    Apache Flink 2.2     │        │             │        │    Apache Spark 3.5.9   │
│  (Real-Time Streaming)  │        │             │        │   (Batch & Micro-Batch) │
├─────────────────────────┤        │             │        ├─────────────────────────┤
│ • Event-time watermark  │        │             │        │ • High-throughput batch │
│ • key_by(user_id)       │        │             │        │ • Parquet conversion    │
│ • 5m/10m Sliding Window │        │             │        │   (7.25x speedup)       │
│ • Velocity Ratios       │        │             │        │ • 927k records/sec      │
│ • Sub-second alert CEP  │        │             │        │ • Micro-batch streaming │
└────────────┬────────────┘        │             │        └────────────┬────────────┘
             │                     │             │                     │
             ▼                     │             │                     ▼
┌─────────────────────────┐        │             │        ┌─────────────────────────┐
│ Kafka: fraud-features   │        │             │        │ Parquet Data Lake Sink  │
│ (Real-time risk stream) │        │             │        │ (Batch feature store)   │
└────────────┬────────────┘        │             │        └────────────┬────────────┘
             │                     │             │                     │
             └──────────────────┐  │  ┌──────────┘                     │
                                │  │  │                                │
                                ▼  ▼  ▼                                │
=======================================================================│==================
                                MODEL SCORING & RESEARCH BOUNDARY      │
=======================================================================│==================
                                                                       │
                        ┌─────────────────────────┐                    │
                        │   Model Serving Layer   │                    │
                        │ (LightGBM E1 Classifier)│                    │
                        │ - Preprocessing: joblib │                    │
                        │ - Model: model.txt      │                    │
                        │ - Threshold: 0.616521   │                    │
                        └────────────┬────────────┘                    │
                                     │                                 │
                                     ▼                                 ▼
                        ┌─────────────────────────┐       ┌─────────────────────────┐
                        │   Transaction Verdict   │       │  Phase 4 Drift Monitor  │
                        │ • APPROVED (p < 0.6165) │       │ • PSI Population Drift  │
                        │ • BLOCKED (p >= 0.6165) │       │ • Retraining Trigger    │
                        └─────────────────────────┘       └─────────────────────────┘
```

---

## 3. Integrated Directory Organization

The target repository structure following the integration of Member 1 and Member 2:

```text
adaptive-upi-fraud-detection/
├── configs/                             # Canonical research configurations
│   └── phase1_lightgbm.yaml
├── src/                                 # Canonical research source code (immutable)
│   ├── audit/
│   ├── data/
│   ├── evaluation/
│   ├── features/
│   ├── models/
│   └── utils/
├── experiments/                         # Research experiment artifacts & runs (immutable)
│   ├── E1_lightgbm/                     # Frozen E1 LightGBM model, scaler, predictions
│   ├── E3_hybrid/                       # Phase 3 hybrid models
│   ├── phase4_drift_adaptation/         # Phase 4 concept drift & adaptation on BAF
│   ├── run_phase1.py
│   └── run_phase15_extend.py
├── phase2_gru/                          # Phase 2 temporal GRU modeling (immutable)
│   ├── artifacts/
│   ├── reports/
│   ├── src/
│   └── tests/
├── reports/                             # Research reports & sprint deliverables
│   ├── phase1/
│   ├── phase2/
│   └── DATASET_FORENSICS.json
│
├── cluster/                             # Kafka multi-broker cluster configs (Member 1)
│   ├── server-1.properties
│   ├── server-2.properties
│   └── server-3.properties
│
├── kafka/                               # Apache Kafka Event Ingestion Layer (Member 1)
│   ├── config/                          # Kafka connection parameters & defaults
│   │   └── config.py
│   ├── transaction_generator/           # Normalizer & synthetic stream generator
│   │   ├── dataset_loader.py
│   │   └── transaction_generator.py
│   ├── producer/                        # High-throughput & benchmark producers
│   │   ├── ieee_cis_replay_producer.py
│   │   ├── high_throughput_producer_benchmark.py
│   │   ├── batching_compression_benchmark.py
│   │   ├── acks_test_producer.py
│   │   └── idempotence_test_producer.py
│   ├── consumer/                        # Consumer groups, scaling, and fraud evaluators
│   │   ├── fraud_detection_consumer.py
│   │   ├── flink_cep_consumer.py
│   │   ├── spark_streaming_consumer.py
│   │   ├── parallel_consumer.py
│   │   └── at_least_once_consumer.py
│   ├── reports/                         # Member 1 research reports & benchmark CSVs
│   │   ├── MEMBER1_RESEARCH_REPORT.md
│   │   ├── MEMBER1_FINAL_DELIVERABLES.md
│   │   └── benchmarks/
│   │       ├── member1_dataset_experiment_results.csv
│   │       └── member1_results_summary.py
│   ├── tests/                           # Kafka unit & contract tests
│   │   ├── test_kafka_connection.py
│   │   └── test_spark_flink_integration.py
│   └── requirements.txt                 # Kafka-specific Python dependencies
│
├── spark/                               # Apache Spark Batch & Streaming Layer (Member 2)
│   ├── batch/                           # Historical data cleaning & windowing
│   │   └── historical_pipeline.py
│   ├── streaming/                       # PySpark Structured Streaming consumer
│   │   ├── stream_processor.py
│   │   └── state_manager.py
│   ├── features/                        # Streaming window feature definitions
│   │   ├── realtime_features.py
│   │   └── feature_definitions.py
│   ├── kafka_connector/                 # Spark-Kafka source and sink adapters
│   │   └── consumer_sink.py
│   ├── benchmarks/                      # Parquet conversion & aggregation benchmarks
│   │   ├── convert_to_parquet.py
│   │   └── run_benchmarks.py
│   ├── configs/                         # Spark configuration
│   │   └── spark_config.yaml
│   ├── reports/                         # Spark benchmark & research reports
│   │   └── SPARK_RESEARCH_REPORT.md
│   ├── tests/                           # Spark unit tests
│   │   ├── test_spark_batch.py
│   │   └── test_spark_streaming.py
│   └── requirements.txt                 # PySpark dependencies
│
├── flink/                               # Apache Flink Real-Time CEP Layer (Member 2)
│   ├── streaming/                       # PyFlink event-time stream processor
│   │   ├── stream_processor.py
│   │   ├── kafka_source.py
│   │   ├── watermark.py
│   │   └── transaction_schema.py
│   ├── features/                        # Stateful velocity and transaction features
│   │   ├── velocity_features.py
│   │   ├── aggregation_features.py
│   │   └── transaction_features.py
│   ├── batch/                           # Flink historical batch preparation
│   │   └── historical_pipeline.py
│   ├── jars/                            # Bundled Kafka connector JARs
│   │   ├── flink-connector-kafka-3.3.0-1.20.jar
│   │   └── kafka-clients-3.8.1.jar
│   ├── configs/                         # Flink configuration
│   │   ├── flink_config.yaml
│   │   └── kafka_config.yaml
│   ├── reports/                         # Flink research reports
│   │   ├── FLINK_RESEARCH_REPORT.md
│   │   └── FLINK_ARCHITECTURE.md
│   ├── tests/                           # Flink unit tests
│   │   ├── test_features.py
│   │   └── test_window_features.py
│   └── requirements.txt                 # PyFlink dependencies
│
├── tests/                               # Test Suites
│   ├── test_phase1.py                   # Canonical Phase 1 tests
│   └── integration/                     # End-to-end integration tests
│       ├── run_full_e2e_integration_test.py # 14-step integration test suite
│       └── test_scale_benchmark_optimizations.py
│
├── docs/                                # Project & Integration Documentation
│   └── integration/                     # Integration reports and specifications
│       ├── TARGET_REPOSITORY_AUDIT.md
│       ├── MEMBER1_MEMBER2_SOURCE_AUDIT.md
│       ├── MEMBER1_MEMBER2_MERGE_MAP.md
│       ├── CONFLICT_ANALYSIS.md
│       ├── FINAL_INTEGRATED_ARCHITECTURE.md
│       ├── DATA_CONTRACT.md
│       └── CODEBASE_INPUT_OUTPUT.md
│
├── run_kafka_spark_flink_pipeline.py    # Top-level multi-process pipeline runner
├── requirements.txt                     # Core research dependencies
└── README.md                            # Comprehensive unified guide
```

---

## 4. Pipeline Data Flow & Engine Responsibilities

### Ingestion Flow (Member 1: Apache Kafka)
1. **Source:** Historical CSV transactions (IEEE-CIS, PaySim, BAF) or real-time synthetic generator.
2. **Normalization:** `dataset_loader.py` maps raw records into normalized JSON events.
3. **Partitioning:** Message key is set to `card_id` / `user_id`. Kafka uses Murmur2 hashing to guarantee all events for the same entity arrive on the **exact same partition in strict temporal order**.
4. **Wire Delivery:** Topic `fraud-transactions` with `acks=all`, `enable.idempotence=true`, and LZ4 compression.

### Stream Processing Flow (Member 2: Apache Flink)
1. **Source:** Ingests from topic `fraud-transactions`.
2. **Event-Time Processing:** Bounded-out-of-orderness watermarks handle network jitter up to 30 seconds.
3. **Keyed State:** Streams partitioned by `user_id`.
4. **Sliding Windows:** Evaluates 5-minute burst windows and 10-minute baseline windows.
5. **Feature Calculation:** Calculates transaction count velocity ratio and monetary velocity ratio.
6. **Output Sink:** Dispatches rich feature vectors to topic `fraud-features` for sub-second serving.

### Batch Processing Flow (Member 2: Apache Spark)
1. **Source:** Reads raw transaction and identity CSV datasets.
2. **Columnar Storage:** Converts flat CSVs into Snappy-compressed Parquet.
3. **Performance Gain:** Column pruning and predicate pushdown yield a **7.25x speedup** (0.637s vs 4.615s for 590k records).
4. **Feature Store:** Generates historical card aggregations (`card_transaction_count_before`, `card_amount_sum_before`, `time_since_previous_transaction`) for offline feature tables and drift monitoring.

### Model Serving & Research Integration
1. **Inference Input:** Ingests canonical 406 features prepared from transactions.
2. **Model Scoring:** Evaluated using frozen LightGBM booster (`experiments/E1_lightgbm/model.txt`).
3. **Decision Policy:** If $p \ge 0.616521$ $\to$ `BLOCKED / SENT TO FRAUD AUDIT`, otherwise `APPROVED`.
4. **Drift Monitoring:** As batches accumulate in the Parquet Data Lake, Phase 4 PSI monitoring detects feature drift and evaluates adaptation needs.

---

## 5. Ownership & Integration Boundaries

| Domain | Responsible Role | Primary Languages & Frameworks | Modification Policy |
| :--- | :--- | :--- | :--- |
| **Research Baseline (Phases 1–4)** | Lead Author | Python, LightGBM, PyTorch, Scikit-Learn | **FROZEN & IMMUTABLE** |
| **Event Ingestion & Messaging** | Member 1 | Python, Apache Kafka, KRaft | Owned by Member 1 |
| **Batch & Micro-Batch Processing** | Member 2 | PySpark, Java 17, Parquet | Owned by Member 2 |
| **Real-Time CEP & Velocity** | Member 2 | PyFlink, Java 17, DataStream API | Owned by Member 2 |
| **Integration Layer & Glue** | Integration Engineer | Multiprocessing, Pytest, Markdown | Co-developed & verified |
| **FastAPI Serving & Docker** | Member 3 | FastAPI, Docker, Uvicorn | **STRICTLY EXCLUDED (Future)** |

---

## 6. Comprehensive Validation Strategy

Validation is executed in strict progressive stages:
1. **Research Integrity Suite:** Verify 162 existing tests pass with identical hashes and metrics.
2. **Modular Unit Tests:** Execute `kafka/tests/`, `spark/tests/`, and `flink/tests/` independently.
3. **14-Step Integration Suite:** Execute `tests/integration/run_full_e2e_integration_test.py` to verify broker connectivity, serialization, partition distribution, zero message loss, and schema integrity.
4. **Multi-Process Pipeline Smoke Test:** Execute `run_kafka_spark_flink_pipeline.py --records 100 --rate 100` to verify live end-to-end event flow.
