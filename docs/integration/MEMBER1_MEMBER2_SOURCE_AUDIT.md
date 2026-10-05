# Source Repository Audit: Member 1 & Member 2 (`HarikaReddy440/Kafka-spark-flink`)

**Source Repository:** `https://github.com/HarikaReddy440/Kafka-spark-flink`  
**Inspected Commit HEAD:** `f74042f` ("Convert spark directory to regular tracked files")  
**Audit Date:** October 2026  
**Auditor:** Senior Software & Research Integration Engineer

---

## 1. Executive Summary & Repository Organization

The source repository represents the completed engineering work of **Member 1 (Apache Kafka Ingestion Layer)** and **Member 2 (Apache Spark Batch/Streaming & Apache Flink Real-Time CEP Layer)**.

A forensic inspection of the Git tree reveals an important structural layout decision made by Member 2 during development:
1. **Member 1 Code:** Resides cleanly at the root level under `kafka/`, `cluster/`, and top-level integration scripts (`run_kafka_spark_flink_pipeline.py`, `run_full_e2e_integration_test.py`, `test_scale_benchmark_optimizations.py`).
2. **Member 2 Code:** Member 2 cloned the target research repository into a subdirectory named `spark/`, within which they developed:
   - `spark/spark/`: The actual Apache Spark implementation (batch, streaming, features, kafka_connector, benchmarks, configs, tests).
   - `spark/flink/`: The actual Apache Flink implementation (batch, streaming, features, jars, configs, tests).
   - `spark/SPARK_AND_FLINK_IMPLEMENTATION_AND_RESULTS.md`, `spark/COMPREHENSIVE_SYSTEM_AND_BENCHMARK_REPORT.md`, `spark/INPUT_OUTPUT.md`: Synthesis reports.
   - **Duplicated Research Trees:** Inside `spark/`, Member 2 carried old copies of `experiments/`, `dataset_analysis/`, `analysis_scripts/`, `phase2_gru/`, `reports/`, and `src/`. **These duplicated research trees must be excluded during merge** to protect the canonical research artifacts in the target repository.

---

## 2. Member 1 Component Audit (Apache Kafka Ingestion Layer)

### 2.1 Component Overview
- **Lead Role:** Member 1 — Event Ingestion and Streaming Layer.
- **Research Question:** *"How can financial transactions be reliably ingested, distributed, and consumed in a scalable real-time fraud detection system?"*
- **Primary Subsystems:**
  - `kafka/transaction_generator/`: Universal CSV loader and synthetic stream generator.
  - `kafka/producer/`: High-throughput producers, replay producers, delivery semantics validators (`acks`, idempotence, retries).
  - `kafka/consumer/`: Multi-process consumer group scaling, offset commit recovery, rule-based fraud consumer, Spark/Flink consumer adapters.
  - `cluster/`: Multi-node KRaft broker configuration profiles (`server-1.properties` to `server-3.properties`).
  - `kafka/tests/`: Connection and schema integration verification.

### 2.2 Deep Component Technical Inventory

| Subsystem / Script | Input | Output | Runtime & Dependencies | External Services / Ports | Dataset Assumptions |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `kafka/transaction_generator/dataset_loader.py` | CSV row from IEEE-CIS, CC 10k, PaySim, BAF, or ULB | Normalized Python dict (`transaction_id`, `card_id`, `amount`, `is_fraud`, etc.) | Python 3.10+ / `csv`, `datetime`, `pathlib` | None | Looks for CSVs in `Datasets/` candidate paths |
| `kafka/transaction_generator/transaction_generator.py` | CLI params (rate, fraud ratio, card pool) | Infinite stream of synthetic JSON events | Python 3.10+ / `random`, `uuid` | None | Synthetic generator |
| `kafka/producer/ieee_cis_replay_producer.py` | Raw IEEE-CIS `train_transaction.csv` | Serialized JSON messages to topic `ieee_cis_transactions` or `fraud-transactions` | `kafka-python`, `json`, `time` | Kafka broker `localhost:9092` | IEEE-CIS train CSV |
| `kafka/producer/high_throughput_producer_benchmark.py` | In-memory cached records | Benchmark metrics (tx/s, latency percentiles) | Multiprocessing, `kafka-python`, `lz4`/`snappy` | Kafka broker `localhost:9092` | Pre-cached batch |
| `kafka/consumer/fraud_detection_consumer.py` | Kafka topic `transactions` | Console alerts: APPROVED vs BLOCKED / SENT TO FRAUD AUDIT | `kafka-python`, `json` | Kafka broker `localhost:9092` | Consumes normalized schema |
| `kafka/consumer/spark_streaming_consumer.py` | Kafka topic `ieee_cis_transactions` | Micro-batch risk scoring console output | `kafka-python` (standalone) or `pyspark` | Kafka broker `localhost:9092` | Normalized IEEE-CIS stream |
| `kafka/consumer/flink_cep_consumer.py` | Kafka topic `ieee_cis_transactions` | Sliding-window velocity alerts | `kafka-python`, `deque`, `time` | Kafka broker `localhost:9092` | Keyed by `card1` |
| `kafka/consumer/multiprocess_parallel_consumer_benchmark.py` | Partitioned Kafka topics (`transactions_p[1,3,6,12]`) | `member1_parallel_consumer_results.csv` | Multiprocessing, `kafka-python` | Kafka broker `localhost:9092` | Consumer group scaling |
| `kafka/consumer/at_least_once_consumer.py` | Kafka topic `transactions` with injected worker kills | Offset recovery & duplicate counts | `kafka-python` | Kafka broker `localhost:9092` | At-least-once verification |
| `cluster/server-[1-3].properties` | Configuration files for KRaft cluster | 3-node Kafka cluster (node.id 1, 2, 3) | Apache Kafka 3.x+ (JVM) | Ports 9092/9093, 9094/9095, 9096/9097 | KRaft metadata quorum |

### 2.3 Member 1 Benchmark Artifacts & Findings
The root of the source repo contains 4 validated benchmark CSV results:
1. `member1_dataset_experiment_results.csv`: Demonstrates sustained zero-loss ingestion across 100, 500, 1000, and 5000 tx/s.
2. `member1_parallel_consumer_results.csv`: Evaluates consumer group scaling efficiency across 1, 3, 6, and 12 partitions (peaking at 30,769 tx/s with 12 consumers).
3. `member1_clean_experiment_results.csv`: Measures durability overhead across `acks=0`, `acks=1`, and `acks=all`.
4. `member1_experiment_results.csv`: Summary of failure recovery times and consumer lag.

---

## 3. Member 2 Component Audit (Apache Spark Layer)

### 3.1 Component Overview
- **Lead Role:** Member 2 — Apache Spark Processing Layer.
- **Research Question:** *"How can Apache Spark provide a unified framework for large-scale historical fraud analysis and real-time transaction stream processing?"*
- **Source Location in Harika Repo:** `spark/spark/`
- **Primary Subsystems:**
  - `batch/historical_pipeline.py`: Large-scale IEEE-CIS CSV ingestion, cleaning, entity linkage with identity data, and windowed card history feature engineering.
  - `benchmarks/run_benchmarks.py` & `convert_to_parquet.py`: Conversion of 590,540 raw CSV rows to Snappy Parquet and 3-trial performance comparison.
  - `features/realtime_features.py` & `feature_definitions.py`: Specification and pure-Python reference implementation of 5m/10m window aggregations and transaction velocity.
  - `streaming/stream_processor.py`: PySpark Structured Streaming consumer with stateful processing and checkpointing.
  - `configs/spark_config.yaml`: Centralized configuration for batch and streaming pipelines.
  - `tests/`: Unit tests for batch cleaning, streaming features, and event-time ordering.

### 3.2 Deep Component Technical Inventory

| Subsystem / Script | Input | Output | Runtime & Dependencies | External Services / Ports | Dataset Assumptions |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `spark/batch/historical_pipeline.py` | `train_transaction.csv` & `train_identity.csv` | Cleaned DataFrame with historical features (`card_transaction_count_before`, `card_amount_sum_before`, `time_since_previous_transaction`) | PySpark 3.5+, Java 17 | Local Spark JVM (`local[*]`) | IEEE-CIS 590k records |
| `spark/benchmarks/convert_to_parquet.py` | Raw IEEE-CIS CSV | Columnar Snappy-compressed Parquet | PySpark 3.5+ | Local Spark JVM | `Datasets/raw/` or `Datasets/IEEE CIS...` |
| `spark/benchmarks/run_benchmarks.py` | CSV vs Parquet directory | Throughput & latency comparison across 3 trials | PySpark 3.5+ | Local Spark JVM | 590,540 rows |
| `spark/features/realtime_features.py` | Stream of `Event(card_id, merchant_id, amount, event_seconds)` | Dict of 5m/10m counts, amounts, unique merchants, time delta | Pure Python (no PySpark needed for unit testing) | None | In-memory event sequence |
| `spark/streaming/stream_processor.py` | Kafka topic `fraud-transactions` | Streaming DataFrame with window features written to Parquet / console | PySpark Structured Streaming, `spark-sql-kafka-0-10_2.12:3.5.9` | Kafka broker `localhost:9092` | JSON event schema |
| `spark/configs/spark_config.yaml` | YAML configuration | Config dict for master, shuffle partitions, windows, Kafka topics | `pyyaml` | None | System paths |

### 3.3 Member 2 Spark Benchmark Results
- **Parquet Speedup:** 7.25x speedup (86.20% processing time reduction: 4.615s down to 0.637s).
- **Batch Throughput:** 927,010 records/sec on columnar Parquet.
- **Streaming Micro-Batch:** 85.40 records/sec with ~2.9s micro-batch trigger latency.

---

## 4. Member 2 Component Audit (Apache Flink Layer)

### 4.1 Component Overview
- **Lead Role:** Member 2 — Apache Flink Real-Time CEP Layer.
- **Research Question:** *"How can Apache Flink provide a unified processing layer for large-scale historical fraud analysis and real-time transaction stream processing?"*
- **Source Location in Harika Repo:** `spark/flink/`
- **Primary Subsystems:**
  - `streaming/stream_processor_flink22_test.py`: PyFlink streaming pipeline with bounded watermarking, `key_by(user_id)`, and sliding window aggregation.
  - `features/velocity_features.py` & `aggregation_features.py`: Stateful `UserVelocityState` deque and multi-window aggregations.
  - `batch/historical_pipeline.py`: Chunked Pandas/Flink batch loader for historical dataset joins.
  - `jars/`: Bundled Kafka connectors:
    - `flink-connector-kafka-3.3.0-1.20.jar` (464 KB)
    - `kafka-clients-3.8.1.jar` (8.4 MB)
  - `benchmarks/streaming_benchmark_producer.py`: High-precision rate-controlled streaming producer for throughput benchmarks.
  - `configs/flink_config.yaml` & `kafka_config.yaml`: Flink job and Kafka topic configuration.

### 4.2 Deep Component Technical Inventory

| Subsystem / Script | Input | Output | Runtime & Dependencies | External Services / Ports | Dataset Assumptions |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `flink/streaming/stream_processor_flink22_test.py` | Kafka topic `fraud-transactions` | Multi-window features published to Kafka topic `fraud-features` | PyFlink 2.2 / 2.3, Java 17 | Kafka broker `localhost:9092` | JSON schema with `user_id`, `amount`, `event_time` |
| `flink/features/velocity_features.py` | Timestamps & amounts per entity | `count`, `total_amount`, pruned history | Pure Python (`collections.deque`) | None | In-memory sliding window |
| `flink/features/transaction_features.py` | Monetary amount | Amount tier (`LOW`, `MEDIUM`, `HIGH`, `VERY_HIGH`) | Pure Python | None | Numeric amount |
| `flink/benchmarks/streaming_benchmark_producer.py` | In-memory or dataset records | Rate-controlled Kafka message burst | `kafka-python` | Kafka broker `localhost:9092` | Sustained 500 tx/s |
| `flink/batch/historical_pipeline.py` | Raw IEEE-CIS CSV files | Processed batch feature parquet files | Pandas / PyArrow chunked reader | File system | Looks for IEEE-CIS CSVs |

### 4.3 Member 2 Flink Benchmark Results
- **Streaming Throughput:** 499.96 records/s (exact target match at 500 tx/s).
- **Latency:** Sub-second (millisecond-level) per-event state updates.
- **Zero Loss:** 5,000 / 5,000 messages acknowledged with 100% success rate.

---

## 5. End-to-End Multi-Process Integration Scripts Audit

The source repository contains top-level scripts designed to orchestrate Member 1 and Member 2 simultaneously:

1. `run_kafka_spark_flink_pipeline.py`:
   - Uses `multiprocessing` to spawn:
     - Worker 1: Producer (`ieee_cis_replay_producer.py` at 500 tx/s).
     - Worker 2: Spark Structured Streaming consumer (`spark_streaming_consumer.py`).
     - Worker 3: Flink CEP consumer (`flink_cep_consumer.py`).
   - Supports `--mode all|spark|flink|producer`.

2. `run_full_e2e_integration_test.py`:
   - Executes a comprehensive 14-step integration test:
     1. Broker connectivity.
     2. Topic verification (`fraud-transactions` with 6 partitions).
     3. Topic verification (`fraud-features` with 6 partitions).
     4. JSON serialization.
     5. Canonical 100-record generation.
     6. Producer dispatch.
     7. Key partition affinity distribution.
     8. Offset tracking and commits.
     9. Flink consumer window verification.
     10. Spark consumer micro-batch verification.
     11. End-to-end latency measurement.
     12. Zero message loss guarantee.
     13. Schema compatibility verification.
     14. Summary report generation.

3. `test_scale_benchmark_optimizations.py`:
   - Evaluates scaling performance under 1,000, 5,000, and 10,000 transaction batches.

---

## 6. Discrepancies Between Code and Documentation

1. **Physical Directory Nesting vs Documentation:**
   - **Documentation states:** Top-level `kafka/`, `spark/`, and `flink/`.
   - **Actual Source Code:** Member 2's Spark code is inside `spark/spark/` and Flink code is inside `spark/flink/`. Furthermore, `spark/` contains an entire duplicate clone of the research repository (`spark/experiments/`, `spark/phase2_gru/`, etc.).
   - **Resolution:** In the target repository, unpack `spark/spark/` into `spark/` and `spark/flink/` into `flink/`. Completely exclude the duplicate research directories.

2. **Hardcoded Machine Paths in Member 2 Batch Scripts:**
   - **Code (`spark/flink/batch/historical_pipeline.py`):** Uses `/mnt/c/Users/HADASSAH KIRAN/Downloads/Datasets/...`.
   - **Documentation:** Mentions repo-relative `Datasets/` directory.
   - **Resolution:** Refactor paths to use repository-relative resolution (`ROOT / "Datasets" / ...`).

3. **Ground Truth Label in Spark Heuristic Scorer:**
   - **Code (`spark_streaming_consumer.py`):** Calculates `risk_score` using `when((col("TransactionAmt") > 1000.0) & (col("isFraud") == 1), 0.88)`.
   - **Documentation:** Claims "LightGBM ML Inference Scorer".
   - **Resolution:** Document this as a heuristic simulation rule. Explicitly state that `isFraud` is an evaluation-only ground-truth label and not an input to the real research model.

4. **Temporary / Backup Files in Source:**
   - The source contains backup files such as `spark/flink/streaming/stream_processor.py.backup`, `stream_processor.py.before_kafka_sink`, and compiled bytecode `__pycache__` / `.pyc`.
   - **Resolution:** Exclude all backup and bytecode files during merge.
