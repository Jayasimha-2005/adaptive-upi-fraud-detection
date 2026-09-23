# MEMBER 2 — APACHE SPARK UNIFIED PROCESSING LAYER

## Sprint 1 — Batch and Real-Time Processing

**Research Project:** Adaptive Financial Fraud Detection
**Module:** Apache Spark Processing Layer

---

# 1. Executive Summary

Member 2 was responsible for developing the Apache Spark processing layer for the adaptive financial fraud detection system.

The implementation provides two processing paths:

1. **Historical batch processing** of IEEE-CIS transaction and identity data.
2. **Real-time transaction processing** using Kafka and Spark Structured Streaming.

The completed implementation includes:

* Spark DataFrame-based batch processing
* IEEE-CIS transaction and identity integration
* Historical card-level feature generation
* Previous-transaction temporal features
* Kafka integration
* Spark Structured Streaming
* JSON transaction parsing and validation
* Event-time processing
* Watermarking
* Sliding 5-minute and 10-minute windows
* Real-time transaction features
* Checkpoint recovery
* Event-time and out-of-order transaction experiments
* Batch and streaming performance benchmarks

The implementation was developed and tested in a Windows local Spark environment.

---

# 2. Research Question

> **How can Apache Spark provide a unified framework for large-scale historical fraud analysis and real-time transaction stream processing?**

The implementation investigates whether the same Spark-based processing layer can support both historical transaction processing and continuous transaction-stream feature generation.

---

# 3. Member 2 Scope

| Component                        | Status                |
| -------------------------------- | --------------------- |
| Spark Batch Processing           | ✅ Complete            |
| IEEE-CIS Transaction Processing  | ✅ Complete            |
| Transaction + Identity Join      | ✅ Complete            |
| Historical Feature Engineering   | ✅ Complete            |
| Previous Transaction — Batch     | ✅ Complete            |
| Kafka Integration                | ✅ Complete            |
| Structured Streaming             | ✅ Complete            |
| JSON Parsing and Validation      | ✅ Complete            |
| Event-Time Processing            | ✅ Complete            |
| Watermarking                     | ✅ Complete            |
| 5-Minute Features                | ✅ Complete            |
| 10-Minute Features               | ✅ Complete            |
| Checkpoint Recovery              | ✅ Tested              |
| Out-of-Order Event Experiment    | ✅ Complete            |
| Batch Benchmark                  | ✅ Complete            |
| Streaming Benchmark              | ✅ Complete            |
| Previous Transaction — Streaming | ⚠️ Limited on Windows |

---

# 4. System Architecture

## 4.1 Historical Processing

```text
IEEE-CIS Transaction Data
          +
IEEE-CIS Identity Data
          |
          v
     Spark Batch
          |
          v
 Transaction + Identity Join
          |
          v
 Data Processing
          |
          v
 Historical Feature Engineering
          |
          v
      Parquet
          |
          v
 Downstream ML Pipeline
```

## 4.2 Real-Time Processing

```text
Live Transactions
       |
       v
     Kafka
       |
       v
Spark Structured Streaming
       |
       v
 JSON Parsing
       |
       v
 Validation
       |
       v
 Event Time
       |
       v
 Watermark
       |
       v
5m / 10m Windows
       |
       v
Real-Time Features
       |
       v
Downstream ML Pipeline
```

---

# 5. Implementation Environment

The implementation was developed and tested using:

| Component        | Version / Configuration |
| ---------------- | ----------------------- |
| Operating System | Windows                 |
| Python           | 3.13.15                 |
| Java             | 17                      |
| PySpark          | 3.5.9                   |
| Kafka            | 4.1.0                   |
| Kafka Broker     | `localhost:9092`        |
| Spark Mode       | `local[*]`              |
| Kafka Topic      | `fraud-transactions`    |

A local Hadoop/Windows helper configuration was also used for Spark execution.

Spark installation was verified using a basic Spark DataFrame execution test.

---

# 6. Historical Batch Processing

The batch pipeline processes the IEEE-CIS transaction and identity datasets.

Input files:

```text
Datasets/IEEE CIS/train_transaction.csv
Datasets/IEEE CIS/train_identity.csv
```

The processing flow is:

```text
train_transaction.csv
        +
train_identity.csv
        |
        v
Spark DataFrames
        |
        v
Transaction + Identity Join
        |
        v
Data Processing
        |
        v
Temporal / Card Features
        |
        v
Aggregation
        |
        v
Parquet Output
```

The raw datasets are not committed to Git because of their large size.

---

# 7. Historical Feature Engineering

The batch implementation generates card-level historical features.

Implemented features include:

```text
card_transaction_count_before
card_amount_sum_before
card_amount_mean_before
card_amount_max_before
card_fraud_count_before
```

These features use the transaction history available before the current transaction.

---

# 8. Previous Transaction Feature

A temporal feature was implemented to determine the previous transaction for each card.

The implementation generates:

```text
previous_transaction_time
time_since_previous_transaction
```

Example:

```text
CARD123

Transaction 1 → 10:00
Transaction 2 → 10:02
Transaction 3 → 10:05
```

Result:

```text
Transaction 1 → previous = NULL
Transaction 2 → previous = 10:00
Transaction 3 → previous = 10:02
```

The feature was verified successfully using the batch pipeline.

---

# 9. Batch Processing Verification

The batch pipeline was tested using a smaller dataset subset before larger processing.

A verified output contained:

```text
card_id   TransactionDT   previous_transaction_time   time_since_previous_transaction
10023     100169          NULL                         NULL
10023     179206          100169                      79037.0
10023     231410          179206                      52204.0
10023     245681          231410                      14271.0
```

The resulting data was successfully written and read from Parquet.

---

# 10. Kafka Integration

Kafka was configured as the real-time transaction source.

Configuration:

```text
Bootstrap Server: localhost:9092
Topic: fraud-transactions
Partitions: 1
Replication Factor: 1
```

Kafka producer testing successfully sent JSON transactions to the topic.

Example transaction structure:

```json
{
  "transaction_id": "TX001",
  "card_id": "CARD123",
  "merchant_id": "M001",
  "amount": 500.0,
  "event_time": "2026-09-24T01:00:00"
}
```

---

# 11. Spark Structured Streaming

The streaming implementation connects Spark Structured Streaming to Kafka.

Processing flow:

```text
Kafka
  |
  v
Read Kafka Messages
  |
  v
Parse JSON
  |
  v
Validate Transaction
  |
  v
Convert event_time
  |
  v
Apply Watermark
  |
  v
Create Sliding Windows
  |
  v
Calculate Features
  |
  v
Write Output
```

The streaming implementation uses the Spark Kafka connector:

```text
org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.9
```

---

# 12. Real-Time Features

The final integrated streaming pipeline generates:

### 5-minute features

```text
transaction_count_5m
transaction_amount_5m
```

### 10-minute features

```text
transaction_count_10m
transaction_amount_10m
unique_merchants_10m
```

---

# 13. Real-Time Feature Verification

The following transactions were used for verification:

```text
10:00 → ₹500
10:02 → ₹800
10:03 → ₹12,000
10:04 → ₹2,000
```

The resulting 5-minute window correctly produced:

```text
transaction_count_5m  = 4
transaction_amount_5m = 15,300
```

The corresponding 10-minute window produced:

```text
transaction_count_10m  = 4
transaction_amount_10m = 15,300
unique_merchants_10m   = 4
```

The output demonstrated the expected sliding-window behaviour.

---

# 14. Event-Time Processing

The streaming implementation uses the transaction's `event_time`.

An out-of-order sequence was tested:

```text
10:01
10:04
10:02
10:03
```

The transactions were intentionally not delivered in chronological order.

The experiment demonstrated that Spark's event-time windows could correctly incorporate the transactions according to their event timestamps while they remained within the configured lateness boundary.

---

# 15. Watermark Configuration

The final streaming pipeline uses:

```text
Watermark = 10 minutes
```

The watermark was tested using later event-time transactions to advance the streaming event-time progress.

The experiment confirmed that the configured event-time windows produced the expected results after watermark progression.

---

# 16. Sliding Window Implementation

The streaming implementation uses a 10-minute event-time window with a 1-minute slide.

This produces overlapping windows.

Example:

```text
00:00 → 00:10
00:01 → 00:11
00:02 → 00:12
00:03 → 00:13
```

The final verification produced multiple overlapping windows with changing transaction counts and amounts as transactions entered and left the 5-minute and 10-minute windows.

---

# 17. Checkpoint Recovery Experiment

Checkpoint recovery was explicitly tested.

Experiment flow:

```text
Start Streaming Query
        |
        v
Process Transaction
        |
        v
Checkpoint Created
        |
        v
Stop Query
        |
        v
Restart Using Same Checkpoint
        |
        v
Send New Transactions
        |
        v
Continue Processing
```

The experiment successfully demonstrated that the streaming query could restart using the existing checkpoint and continue processing subsequent Kafka transactions.

The result supports the use of checkpointing for recovery in the implemented streaming pipeline.

---

# 18. Previous-Transaction Streaming Experiment

Spark's stateful processing API was independently tested for previous-transaction tracking.

The test successfully produced:

```text
PREV001 → previous = NULL
PREV002 → previous = 03:00:00
PREV003 → previous = 03:02:00
PREV004 → previous = 03:05:00
```

The corresponding time differences were also calculated successfully.

However, integrating this stateful operation into the main Kafka → Spark → Parquet pipeline caused Python worker connection/time-out problems in the Windows local environment.

Therefore:

```text
Batch previous-transaction feature:
        ENABLED

Main streaming previous-transaction feature:
        DISABLED
```

The feature was not presented as part of the final integrated streaming pipeline.

---

# 19. Batch vs Streaming Benchmark

## 19.1 Batch Benchmark

Measured result:

```text
Records              = 47,203
Processing Time       = 33.67 seconds
Throughput            = 1,401.89 records/second
```

## 19.2 Streaming Benchmark

Measured result:

```text
Records               = 20
Processing Time       = 5.44 seconds
Throughput            = 3.68 records/second
Micro-batches         = 2
Average records/batch = 10
```

The streaming benchmark also showed that the local environment could fall behind a 5-second processing trigger when Spark required more time to process a micro-batch.

These measurements represent the local Windows development environment and are not general performance limits of Spark.

---

# 20. Testing and Verification

The following implementation areas were tested:

| Test                                                | Result     |
| --------------------------------------------------- | ---------- |
| Spark startup                                       | ✅ Passed   |
| Batch pipeline syntax                               | ✅ Passed   |
| Batch Parquet output                                | ✅ Passed   |
| Historical feature generation                       | ✅ Passed   |
| Previous transaction — batch                        | ✅ Passed   |
| Kafka topic                                         | ✅ Passed   |
| Kafka producer                                      | ✅ Passed   |
| Streaming startup                                   | ✅ Passed   |
| JSON parsing                                        | ✅ Passed   |
| Transaction validation                              | ✅ Passed   |
| 5-minute windows                                    | ✅ Passed   |
| 10-minute windows                                   | ✅ Passed   |
| Event-time processing                               | ✅ Passed   |
| Out-of-order events                                 | ✅ Passed   |
| Watermark processing                                | ✅ Passed   |
| Checkpoint recovery                                 | ✅ Passed   |
| Stateful previous transaction — isolated test       | ✅ Passed   |
| Stateful previous transaction — integrated pipeline | ⚠️ Limited |
| Batch benchmark                                     | ✅ Passed   |
| Streaming benchmark                                 | ✅ Passed   |

---

# 21. Complete Codebase Structure

```text
spark/
│
├── __init__.py
│
├── batch/
│   ├── __init__.py
│   └── historical_pipeline.py
│
├── streaming/
│   ├── __init__.py
│   ├── stream_processor.py
│   └── state_manager.py
│
├── features/
│   ├── __init__.py
│   ├── feature_definitions.py
│   └── realtime_features.py
│
├── kafka_connector/
│   ├── __init__.py
│   ├── consumer_sink.py
│   └── producer_simulator.py
│
├── experiments/
│   ├── __init__.py
│   ├── event_time_experiment.py
│   └── batch_vs_streaming.py
│
├── benchmarks/
│   ├── __init__.py
│   └── run_benchmarks.py
│
├── configs/
│   └── spark_config.yaml
│
├── reports/
│   └── SPARK_RESEARCH_REPORT.md
│
├── tests/
│   ├── test_event_time.py
│   ├── test_previous_transaction_stateful.py
│   ├── test_spark_batch.py
│   └── test_spark_streaming.py
│
└── requirements.txt
```

---

# 22. Reproducibility

## Batch

```text
python -m spark.batch.historical_pipeline
```

Input:

```text
Datasets/IEEE CIS/train_transaction.csv
Datasets/IEEE CIS/train_identity.csv
```

## Streaming

Kafka must be running first.

```text
python -m spark.streaming.stream_processor
```

Kafka topic:

```text
fraud-transactions
```

## Transaction Generation

```text
python -m spark.kafka_connector.producer_simulator
```

## Event-Time Experiment

```text
python -m spark.experiments.event_time_experiment
```

## Tests

```text
pytest spark/tests -q
```

---

# 23. Limitations

1. The implementation was developed in Spark local mode rather than a multi-node cluster.
2. Performance measurements therefore represent the local development environment.
3. Streaming processing on Windows was slower than the configured 5/10-second trigger intervals in some experiments.
4. The stateful previous-transaction operation was unstable when integrated with the main Parquet streaming pipeline.
5. The final streaming pipeline therefore uses the verified 5-minute and 10-minute features.
6. Raw IEEE-CIS datasets are stored locally and are not committed to Git.
7. Spark provides the processing and feature-engineering layer; the downstream ML model remains responsible for fraud prediction.

---

# 24. Achievements of Member 2

| Achievement                                | Result    |
| ------------------------------------------ | --------- |
| Historical Spark pipeline                  | Completed |
| IEEE-CIS transaction + identity processing | Completed |
| Historical card features                   | Completed |
| Previous-transaction batch feature         | Verified  |
| Kafka integration                          | Completed |
| Spark Structured Streaming                 | Completed |
| Event-time processing                      | Verified  |
| Out-of-order transaction handling          | Verified  |
| Watermark processing                       | Verified  |
| 5-minute features                          | Verified  |
| 10-minute features                         | Verified  |
| Checkpoint recovery                        | Verified  |
| Batch performance measurement              | Completed |
| Streaming performance measurement          | Completed |

---

# 25. Final Outcome

Member 2 successfully developed the Spark processing layer connecting historical fraud analysis with real-time transaction processing.

The completed implementation provides:

```text
Historical IEEE-CIS Data
          |
          v
      Spark Batch
          |
          v
 Historical Features
          |
          v
       Parquet


Live Transactions
          |
          v
         Kafka
          |
          v
Spark Structured Streaming
          |
          v
Event-Time + Watermark
          |
          v
5m / 10m Features
          |
          v
       Parquet
```

The implementation establishes the Spark layer required for integrating historical features and real-time transaction features with the project's downstream fraud-detection models.

---

# 26. Conclusion

Member 2 completed the Apache Spark processing implementation for both historical and real-time fraud-detection workloads.

The batch pipeline was successfully used for historical IEEE-CIS processing and temporal feature generation. The streaming pipeline was successfully integrated with Kafka and verified for event-time processing, out-of-order transactions, watermarking, sliding windows, real-time feature generation, and checkpoint recovery.

The final implementation provides a working foundation for connecting the Spark-generated features to the existing fraud-detection models.
