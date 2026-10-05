# MEMBER 1 & MEMBER 2 INPUT / OUTPUT SPECIFICATION

**Repository:** `adaptive-upi-fraud-detection`  
**Current Branch:** `Upto_Phase-4` (`c78494e`)  
**Audit Scope:** Apache Kafka (Member 1), Apache Spark (Member 2), Apache Flink (Member 2)  

---

## 1. Member 1: Apache Kafka Layer (`kafka/`, `cluster/`)

### Input Specification
- **Supported Sources:**
  1. `train_transaction.csv` (IEEE-CIS dataset, 590,540 rows, 434 raw columns) via `kafka/producer/ieee_cis_replay_producer.py`.
  2. Synthetic high-velocity transactions generated via `kafka/transaction_generator/transaction_generator.py`.
  3. Credit Card 10k dataset via `kafka/transaction_generator/dataset_loader.py`.
  4. PaySim mobile money transfer CSVs via multi-dataset loader.
- **Input Formats:** CSV rows, normalized Python dictionaries.

### Processing & Transformation
- **Normalization:** Converts raw string values to typed floats/integers, computes epoch timestamps, assigns `card_id` as composite partition key.
- **Throttling:** Token bucket rate limiter supporting rates from 20 tx/s up to unthrottled (~10,000+ tx/s).
- **Delivery Semantics:** Reliable producer delivery configuration using `acks=all`, `enable.idempotence=True`, `retries=5` to prevent message loss and duplicate writes to broker logs.

### Output Specification
- **Primary Topic:** `fraud-transactions` (or `ieee_cis_transactions`).
- **Partition Count:** 6 Partitions (configured in cluster configs `cluster/server-[1-3].properties`).
- **Partitioning Key:** `card_id` (string UTF-8, ensures partition-level sequence affinity per cardholder).
- **Message Value:** Serialized JSON string (UTF-8) containing:
  ```json
  {
    "TransactionID": 3000001,
    "card_id": "CARD_12345",
    "TransactionAmt": 150.00,
    "TransactionDT": 86400,
    "ProductCD": "W",
    "card1": 10001.0,
    "timestamp": 1728153000.123
  }
  ```

---

## 2. Member 2: Apache Spark Layer (`spark/`)

### Input Specification
- **Streaming Input:** Kafka topic `fraud-transactions` consumed via Structured Streaming reader (`spark.readStream.format("kafka")`).
- **Batch Input:** Raw IEEE-CIS CSV files from `Datasets/IEEE-CIS-Fraud-Detection/train_transaction.csv`.

### Processing & Transformation
- **Parquet Conversion (`spark/benchmarks/convert_to_parquet.py`):**
  - Converts CSV $\to$ Snappy-compressed Parquet.
  - Scan rate acceleration: **7.25x** (927k rec/s vs 128k rec/s).
  - Storage reduction: **77%** (683 MB $\to$ 157 MB).
- **Micro-Batch Streaming (`spark/streaming/stream_processor.py`):**
  - Event-time windowing: 5-minute sliding windows with 10-second watermark tolerance.
  - Aggregations: `tx_count_5m`, `amt_sum_5m`, `amt_avg_5m`, `amt_to_mean_ratio`.
  - Stateful tracking: `mapGroupsWithState` in `spark/streaming/state_manager.py` tracks previous transaction amount and inter-transaction time delta $\Delta t$.

### Output Specification
- **Lakehouse Parquet Sink:** Written to `spark/output/streaming/features` (Parquet schema containing card window statistics).
- **Stateful Output Sink:** Written to `spark/output/streaming/previous_transaction`.
- **Console Sink:** Micro-batch tabular metrics displayed during benchmarking.

---

## 3. Member 2: Apache Flink Layer (`flink/`)

### Input Specification
- **Stream Ingestion:** Kafka topic `fraud-transactions` via PyFlink DataStream API (`KafkaSourceBuilder` with JAR `flink-connector-kafka-3.3.0-1.20.jar`).
- **Batch Ingestion:** `flink/batch/historical_pipeline.py` consumes raw transaction tables.

### Processing & Transformation
- **Event-Time Watermarking:** Assigns timestamps with 10-second bounded out-of-orderness (`WatermarkStrategy.for_bounded_out_of_orderness`).
- **Keyed Stream:** `key_by(lambda tx: tx.card_id)` ensures per-card state isolation.
- **Complex Event Processing (CEP):**
  - 5-minute and 10-minute sliding event-time windows.
  - Velocity ratio calculation:
    $$V_R = \frac{\text{TransactionAmt}}{\text{avg\_amt\_10m} + 1.0}$$
  - Spike detection: Alerts triggered when $V_R > 3.0$ or when $\ge 3$ transactions occur within 5 minutes.

### Output Specification
- **Alert Stream:** Emitted to Kafka topic `fraud-alerts` with payload:
  ```json
  {
    "alert_type": "VELOCITY_SPIKE",
    "card_id": "CARD_12345",
    "velocity_ratio": 4.25,
    "transaction_amt": 5000.0,
    "avg_10m": 1176.47,
    "timestamp": 1728153005.0
  }
  ```
- **Feature Stream:** Emitted to Kafka topic `fraud-features` for downstream consumers.
