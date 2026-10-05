# MEMBER 2 — APACHE SPARK UNIFIED PROCESSING LAYER

## Sprint 1 — Batch and Real-Time Processing

**Research Project:** Adaptive Financial Fraud Detection
**Module:** Apache Spark Processing Layer
**Role:** Member 2 — Apache Spark

---

# 1. Executive Summary

Member 2 was responsible for developing the **Apache Spark processing layer** of the adaptive financial fraud detection system.

The module provides two processing paths:

1. **Historical batch processing** for large transaction datasets.
2. **Real-time stream processing** using Apache Kafka and Spark Structured Streaming.

The implemented layer includes:

* Spark DataFrame-based batch processing
* IEEE-CIS transaction and identity processing
* Transaction + identity joining
* Historical feature engineering
* Previous-transaction temporal features
* Kafka integration
* Spark Structured Streaming
* JSON parsing and validation
* Event-time processing
* Watermarking
* Sliding 5-minute and 10-minute windows
* Real-time feature generation
* Stateful previous-transaction processing
* Checkpointing
* Out-of-order event testing
* Performance benchmarking
* Automated testing

The implementation was developed and tested in a **Windows local Spark environment**.

---

# 2. Research Question

> **How can Apache Spark provide a unified framework for large-scale historical fraud analysis and real-time transaction stream processing?**

The work investigates how Spark can support both historical and continuous transaction processing using a common processing framework.

---

# 3. Member 2 Scope

| Component                       | Status                 |
| ------------------------------- | ---------------------- |
| Spark Batch Processing          | Complete               |
| IEEE-CIS Transaction Processing | Complete               |
| IEEE-CIS Identity Processing    | Complete               |
| Transaction + Identity Join     | Complete               |
| Historical Feature Engineering  | Complete               |
| Previous Transaction — Batch    | Complete               |
| Kafka Integration               | Complete               |
| Structured Streaming            | Complete               |
| JSON Parsing                    | Complete               |
| Transaction Validation          | Complete               |
| Event-Time Processing           | Complete               |
| Watermarking                    | Complete               |
| 5-Minute Features               | Complete               |
| 10-Minute Features              | Complete               |
| Sliding Windows                 | Complete               |
| Stateful Previous Transaction   | Implemented and tested |
| Checkpointing                   | Tested                 |
| Out-of-Order Events             | Tested                 |
| Automated Tests                 | 4/4 Passed             |
| Batch Benchmark                 | Completed              |
| Streaming Benchmark             | Completed              |

---

# 4. System Architecture

## 4.1 Historical Batch Path

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

## 4.2 Real-Time Streaming Path

```text
New Transactions
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
Sliding Windows
       |
       v
Real-Time Features
       |
       v
     Parquet
       |
       v
Downstream ML Pipeline
```

---

# 5. Implementation Environment

| Component          | Configuration        |
| ------------------ | -------------------- |
| Operating System   | Windows              |
| Python             | 3.13.15              |
| Java               | OpenJDK / Temurin 17 |
| PySpark            | 3.5.9                |
| Kafka              | 4.1.0                |
| Kafka Broker       | `localhost:9092`     |
| Spark Mode         | `local[2]`           |
| Kafka Topic        | `fraud-transactions` |
| Python Environment | `venv`               |

The implementation was tested using the project's `venv` environment.

The `venv311` directory was not used for the final implementation and is ignored by Git.

---

# 6. Historical Batch Processing

The batch pipeline processes the IEEE-CIS transaction and identity datasets.

Input files:

```text
Datasets/IEEE CIS/train_transaction.csv
Datasets/IEEE CIS/train_identity.csv
```

Processing flow:

```text
CSV Files
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
Feature Engineering
   |
   v
Parquet Output
```

The raw datasets are stored locally and are not committed to GitHub because of their size.

---

# 7. Transaction and Identity Integration

The IEEE-CIS dataset contains transaction information and additional identity information.

Spark loads both datasets into DataFrames and combines the relevant records through the transaction identifier.

```text
Transaction Data
       +
Identity Data
       |
       v
Combined Spark DataFrame
       |
       v
Feature Engineering
```

This produces a richer dataset for downstream fraud-analysis tasks.

---

# 8. Historical Feature Engineering

The batch pipeline generates card-level historical features using information available before the current transaction.

Implemented features include:

```text
card_transaction_count_before
card_amount_sum_before
card_amount_mean_before
card_amount_max_before
card_fraud_count_before
```

For example:

```text
10:00 → ₹100
10:02 → ₹250
10:05 → ₹500
```

For the 10:05 transaction:

```text
Previous transaction count = 2
Previous amount sum = ₹350
Previous amount mean = ₹175
Previous amount maximum = ₹250
```

This prevents future transaction information from being used when calculating historical features.

---

# 9. Previous-Transaction Feature

The batch implementation generates:

```text
previous_transaction_time
time_since_previous_transaction
```

Example:

```text
CARD100

10:00 → Transaction 1
10:02 → Transaction 2
10:04 → Transaction 3
```

Result:

```text
Transaction 1 → Previous = NULL

Transaction 2 → Previous = 10:00
                 Difference = 120 seconds

Transaction 3 → Previous = 10:02
                 Difference = 120 seconds
```

These features can help identify unusually frequent transaction activity.

---

# 10. Kafka Integration

Kafka is used as the real-time transaction source.

Configuration:

```text
Bootstrap Server: localhost:9092
Topic: fraud-transactions
```

The Kafka broker was verified locally on port `9092`.

Transactions are sent as JSON messages.

Example:

```json
{
  "transaction_id": "TEST001",
  "card_id": "CARD100",
  "merchant_id": "MERCHANT01",
  "amount": 100.0,
  "event_time": "2026-09-30T10:00:00"
}
```

For the installed Kafka version, the console producer uses:

```powershell
.\kafka-console-producer.bat --bootstrap-server localhost:9092 --topic fraud-transactions
```

---

# 11. Spark Structured Streaming

Spark Structured Streaming reads transactions from Kafka and processes them continuously.

The processing sequence is:

```text
Kafka
 |
 v
Read Message
 |
 v
Parse JSON
 |
 v
Validate Fields
 |
 v
Convert Event Time
 |
 v
Apply Watermark
 |
 v
Calculate Windows
 |
 v
Generate Features
 |
 v
Write Parquet
```

The Spark Kafka connector used is:

```text
org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.9
```

---

# 12. JSON Parsing and Validation

Incoming Kafka messages are parsed into structured Spark columns.

The main fields are:

```text
transaction_id
card_id
merchant_id
amount
event_time
```

Validation is performed before feature generation so that malformed or invalid transactions do not enter the processing stage.

---

# 13. Event-Time Processing

The streaming pipeline uses the transaction's actual:

```text
event_time
```

rather than relying only on message arrival time.

This is important because transactions can arrive out of order.

Example arrival order:

```text
10:01
10:04
10:02
10:03
```

Spark evaluates the transactions according to their event timestamps when performing event-time window calculations.

---

# 14. Watermarking

The streaming implementation uses:

```text
Watermark = 10 minutes
```

A watermark allows Spark to handle late-arriving events while preventing streaming state from being retained indefinitely.

It provides a boundary for how long Spark should wait for delayed events during event-time processing.

---

# 15. Sliding Windows

The implementation uses:

```text
Window Duration = 10 minutes
Slide = 1 minute
```

This creates overlapping windows:

```text
00:00 → 00:10
00:01 → 00:11
00:02 → 00:12
00:03 → 00:13
...
```

Overlapping windows allow the system to continuously calculate recent transaction behaviour.

---

# 16. Real-Time Features

The streaming layer generates two groups of features.

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

For example, if a card performs:

```text
10:00 → ₹100 → Merchant A
10:02 → ₹250 → Merchant B
10:04 → ₹500 → Merchant C
```

The corresponding 10-minute values can be:

```text
transaction_count_10m = 3
transaction_amount_10m = 850
unique_merchants_10m = 3
```

---

# 17. Streaming Output Verification

The generated streaming Parquet output was inspected directly during Windows testing.

The verified output contained:

```text
window_start
window_end
card_id
transaction_count_5m
transaction_amount_5m
transaction_count_10m
transaction_amount_10m
unique_merchants_10m
```

The verification found:

```text
Non-empty streaming output rows = 14
```

An example verified result was:

```text
card_id = CARD100

transaction_count_5m = 3
transaction_amount_5m = 850.0

transaction_count_10m = 3
transaction_amount_10m = 850.0

unique_merchants_10m = 3
```

---

# 18. Stateful Previous-Transaction Processing

A genuine Spark stateful implementation was developed using:

```text
applyInPandasWithState
```

The state is maintained independently for each:

```text
card_id
```

The state stores recent event timestamps and is used to calculate:

```text
previous_transaction_time
time_since_previous_transaction
```

The state is bounded to:

```text
100 timestamps per card
```

This prevents unbounded state growth.

Processing logic:

```text
Transaction
     |
     v
Group by card_id
     |
     v
Read Existing State
     |
     v
Find Previous Transaction
     |
     v
Calculate Time Difference
     |
     v
Add Current Event
     |
     v
Limit State to 100 Timestamps
     |
     v
Update State
```

---

# 19. Stateful Output Verification

The stateful previous-transaction processing was executed and verified.

The generated output contained:

```text
Total rows = 6
```

The output included:

```text
transaction_id
card_id
merchant_id
amount
event_time
previous_transaction_time
time_since_previous_transaction
```

For example:

```text
TEST002
event_time = 10:02
previous_transaction_time = 10:00
time_since_previous_transaction = 120 seconds
```

This confirms that the stateful temporal feature was actually implemented and produced output.

---

# 20. Local Windows Optimizations

Because the implementation was tested on a local Windows environment, several settings were adjusted to reduce processing overhead.

### Kafka records per trigger

```text
maxOffsetsPerTrigger = 10
```

### Spark shuffle partitions

```text
spark.sql.shuffle.partitions = 2
```

### Kafka starting position

```text
startingOffsets = earliest
```

The smaller Kafka trigger size makes local stateful processing easier to test.

The settings are development configurations and should not be treated as production cluster settings.

---

# 21. Checkpointing

Checkpointing was tested for streaming recovery.

The experiment followed:

```text
Start Query
   |
   v
Process Transactions
   |
   v
Checkpoint Created
   |
   v
Stop Query
   |
   v
Restart Query
   |
   v
Continue Processing
```

Checkpoint/state-store files were generated during the experiments.

Checkpointing provides the mechanism required for recovering streaming progress and state after a restart.

---

# 22. Out-of-Order Event Experiment

Out-of-order transactions were intentionally generated to verify event-time processing.

Example:

```text
10:01
10:04
10:02
10:03
```

The experiment demonstrated that Spark could use event timestamps when calculating time-based windows instead of simply treating message arrival order as transaction order.

---

# 23. Windows Hadoop NativeIO Issue

During Windows testing, Spark encountered a Hadoop NativeIO error while directly scanning some streaming Parquet output:

```text
java.lang.UnsatisfiedLinkError:
'boolean org.apache.hadoop.io.nativeio.NativeIO$Windows.access0(...)'
```

This is a Windows-specific Hadoop native I/O issue.

The generated Parquet files were nevertheless present and were independently inspected using PyArrow.

The verification showed:

```text
Streaming output files = 8
Total non-empty rows = 14
```

Therefore, the issue affected direct Spark-side inspection in the local Windows environment rather than proving that the output was absent.

---

# 24. PyArrow Verification

PyArrow was used to independently inspect generated Parquet files when Spark encountered the Windows NativeIO issue.

The streaming output contained:

```text
8 Parquet files
```

Their row counts were:

```text
0
0
0
0
6
0
0
8
```

Therefore:

```text
Total rows = 14
```

The separate previous-transaction output contained:

```text
5 files
Total rows = 6
```

This provided an independent verification of the generated outputs.

---

# 25. Automated Testing

The Spark test suite was executed using:

```powershell
venv\Scripts\python.exe -m pytest spark/tests -q
```

Result:

```text
4 passed
```

Therefore:

```text
4 / 4 automated tests passed
```

The tests provide automated verification of the implemented Spark functionality.

---

# 26. Testing Summary

| Test Area                         | Result     |
| --------------------------------- | ---------- |
| Spark startup                     | Passed     |
| Spark DataFrame processing        | Passed     |
| Batch processing                  | Passed     |
| Batch Parquet output              | Passed     |
| Historical feature generation     | Passed     |
| Previous transaction — batch      | Passed     |
| Kafka broker                      | Passed     |
| Kafka topic                       | Passed     |
| Kafka producer                    | Passed     |
| Structured Streaming startup      | Passed     |
| JSON parsing                      | Passed     |
| Transaction validation            | Passed     |
| Event-time processing             | Passed     |
| Watermark processing              | Passed     |
| Sliding windows                   | Passed     |
| 5-minute features                 | Passed     |
| 10-minute features                | Passed     |
| Out-of-order events               | Passed     |
| Stateful previous transaction     | Passed     |
| Streaming Parquet generation      | Verified   |
| Checkpoint/state-store generation | Verified   |
| Automated test suite              | 4/4 Passed |

---

# 27. Performance Benchmarks

The following measurements were obtained from the completed benchmark runs in the local Windows development environment.

## 27.1 Batch Benchmark

```text
Records              = 1,589,789
Processing Time       = 15.9130275 seconds
Throughput            = 99,904.8735 records/second
```

### Batch benchmark summary

| Metric          |                Result |
| --------------- | --------------------: |
| Records         |             1,589,789 |
| Processing Time |          15.9130275 s |
| Throughput      | 99,904.8735 records/s |

---

## 27.2 Streaming Benchmark

```text
Records                    = 20
Processing Time             = 7.7562616 seconds
Throughput                  = 2.5785618 records/second
Micro-batches               = 3
Average records/micro-batch = 6.6666667
```

### Streaming benchmark summary

| Metric                      |              Result |
| --------------------------- | ------------------: |
| Records                     |                  20 |
| Processing Time             |         7.7562616 s |
| Throughput                  | 2.5785618 records/s |
| Micro-batches               |                   3 |
| Average records/micro-batch |           6.6666667 |

---

# 28. Benchmark Interpretation

The benchmark results represent the **local Windows development environment** and should not be treated as general limits of Apache Spark.

The batch benchmark processed:

```text
1,589,789 records
```

in approximately:

```text
15.91 seconds
```

The streaming benchmark processed:

```text
20 records
```

across:

```text
3 micro-batches
```

The two measurements represent different processing models and workloads.

Batch processing works on a fixed dataset, whereas streaming includes additional overhead from:

```text
Kafka
+
micro-batch scheduling
+
event-time processing
+
watermarking
+
window computation
+
state management
+
checkpointing
+
Parquet output
```

Therefore, the batch and streaming throughput figures should be reported as separate measurements rather than directly treating one as a production comparison against the other.

---


# 29. Codebase Structure

The main Spark module is organized into separate components for batch processing, streaming, features, Kafka integration, experiments, benchmarks, and tests.

```text
spark/
│
├── batch/
│   └── historical_pipeline.py
│
├── streaming/
│   ├── stream_processor.py
│   └── state_manager.py
│
├── features/
│   ├── feature_definitions.py
│   └── realtime_features.py
│
├── kafka_connector/
│   ├── consumer_sink.py
│   └── producer_simulator.py
│
├── experiments/
│   ├── event_time_experiment.py
│   └── batch_vs_streaming.py
│
├── benchmarks/
│   └── run_benchmarks.py
│
├── configs/
│   └── spark_config.yaml
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

# 30. Reproducibility

## Run Automated Tests

```powershell
venv\Scripts\python.exe -m pytest spark/tests -q
```

Expected result:

```text
4 passed
```

## Run Batch Pipeline

```powershell
venv\Scripts\python.exe -m spark.batch.historical_pipeline
```

Input:

```text
Datasets/IEEE CIS/train_transaction.csv
Datasets/IEEE CIS/train_identity.csv
```

## Run Streaming Pipeline

Start Kafka first, then run:

```powershell
venv\Scripts\python.exe -m spark.streaming.stream_processor
```

Kafka topic:

```text
fraud-transactions
```

## Kafka Console Producer

```powershell
.\kafka-console-producer.bat --bootstrap-server localhost:9092 --topic fraud-transactions
```

---

# 31. Limitations

The implementation has several development-environment limitations:

1. Spark was tested using local mode rather than a multi-node cluster.

2. Benchmark results represent the local Windows machine and cannot be treated as production performance.

3. Windows Hadoop NativeIO issues affected some direct Spark inspection of generated Parquet files.

4. Streaming has additional overhead from Kafka, micro-batches, state, windows, watermarking, checkpointing, and output writing.

5. IEEE-CIS datasets are stored locally because of their large size.

6. Spark is responsible for data processing and feature engineering; the downstream fraud prediction model is handled separately.

---

# 32. Achievements of Member 2

| Achievement                        | Result                   |
| ---------------------------------- | ------------------------ |
| Spark batch pipeline               | Completed                |
| IEEE-CIS transaction processing    | Completed                |
| IEEE-CIS identity processing       | Completed                |
| Transaction + identity integration | Completed                |
| Historical feature engineering     | Completed                |
| Previous-transaction batch feature | Verified                 |
| Kafka integration                  | Completed                |
| Structured Streaming               | Completed                |
| JSON parsing                       | Verified                 |
| Transaction validation             | Verified                 |
| Event-time processing              | Verified                 |
| Watermarking                       | Verified                 |
| Sliding windows                    | Verified                 |
| 5-minute features                  | Verified                 |
| 10-minute features                 | Verified                 |
| Stateful previous transaction      | Implemented and verified |
| Checkpointing                      | Tested                   |
| Out-of-order events                | Tested                   |
| Streaming Parquet output           | Verified                 |
| Automated tests                    | 4/4 Passed               |
| Batch benchmark                    | Completed                |
| Streaming benchmark                | Completed                |
| Git branch                         | `spark-processing`       |
| Git commit                         | `1b47b35`                |
| GitHub push                        | Successful               |

---

# 33. Final Outcome

The completed Member 2 module provides a unified Spark processing layer for historical and real-time financial transaction workloads.

### Historical processing

```text
IEEE-CIS Data
     |
     v
Spark Batch
     |
     v
Feature Engineering
     |
     v
Historical Features
     |
     v
Parquet
```

### Real-time processing

```text
Transaction
     |
     v
Kafka
     |
     v
Spark Structured Streaming
     |
     v
JSON + Validation
     |
     v
Event Time + Watermark
     |
     v
Sliding Windows
     |
     v
5m / 10m Features
     |
     v
Parquet
```

The module also contains stateful previous-transaction processing using:

```text
applyInPandasWithState
```

with:

```text
previous_transaction_time
time_since_previous_transaction
```

The implementation was verified through automated tests, Kafka-based streaming tests, generated Parquet output, stateful processing verification, and performance benchmarks.

---

# 34. Conclusion

Member 2 completed the Apache Spark processing layer for the adaptive financial fraud detection project.

The implementation demonstrates how Spark can be used as a common processing framework for:

* historical IEEE-CIS analysis,
* transaction and identity integration,
* historical feature engineering,
* Kafka-based real-time processing,
* event-time analysis,
* watermarking,
* sliding-window feature generation,
* stateful transaction-history processing, and
* downstream ML feature preparation.

The final implementation achieved:

```text
4/4 automated tests passed

14 verified streaming feature rows

6 verified previous-transaction state rows

Batch benchmark:
1,589,789 records
15.9130275 seconds
99,904.8735 records/second

Streaming benchmark:
20 records
7.7562616 seconds
2.5785618 records/second
3 micro-batches
6.6666667 records/micro-batch
```

The completed Spark module therefore fulfills Member 2's responsibility of building the **Apache Spark batch and real-time processing layer** for the research project.
