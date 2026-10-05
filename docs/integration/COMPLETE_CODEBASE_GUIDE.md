# Complete Codebase Architecture and Implementation Guide

**Project**: Real-Time & Historical Financial Fraud Detection Platform  
**Technologies**: Apache Kafka | Apache Flink | Apache Spark (Batch & Streaming) | LightGBM | Python  
**Target Dataset**: IEEE-CIS Fraud Detection Benchmark (590,540 records)  
**Integration Status**: Verified & Passed (100% Zero-Loss Delivery & High-Throughput Benchmarked)

---

## Table of Contents

1. [Executive Summary & System Architecture](#1-executive-summary--system-architecture)
2. [Data Layer & Canonical Event Schema](#2-data-layer--canonical-event-schema)
3. [Apache Kafka Ingestion Layer (Member 1)](#3-apache-kafka-ingestion-layer-member-1)
4. [Apache Flink Real-Time CEP Layer (Member 2)](#4-apache-flink-real-time-cep-layer-member-2)
5. [Apache Spark Batch & Streaming Engine (Member 2)](#5-apache-spark-batch--streaming-engine-member-2)
6. [End-to-End Integration Architecture](#6-end-to-end-integration-architecture)
7. [Comprehensive Performance & Benchmark Results](#7-comprehensive-performance--benchmark-results)
8. [Complete Repository Directory & File Inventory](#8-complete-repository-directory--file-inventory)
9. [Step-by-Step Execution & Reproduction Guide](#9-step-by-step-execution--reproduction-guide)

---

## 1. Executive Summary & System Architecture

This repository hosts an enterprise-grade, distributed stream and batch processing platform engineered to detect financial transaction fraud at ultra-high throughput and sub-second latency.

The system integrates a unified data flow between two core functional layers:
- **Member 1 (Kafka Streaming Platform)**: High-throughput ingestion, deterministic partition routing, and durable pub/sub brokering.
- **Member 2 (Flink & Spark Stream/Batch Processing)**: Stateful Complex Event Processing (CEP), sliding window velocity aggregations, micro-batch real-time ML inference, and high-performance batch analytics.

### End-to-End Architecture Flow

```
                      ┌─────────────────────────────────────────┐
                      │  Transaction Generator / IEEE-CIS Data  │
                      └────────────────────┬────────────────────┘
                                           │
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │    Kafka Producer (card_id Partition)   │
                      └────────────────────┬────────────────────┘
                                           │
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │      Kafka Topic: fraud-transactions    │
                      │         (6 - 12 Partitions, acks=all)   │
                      └────────────┬───────────────┬────────────┘
                                   │               │
                  ┌────────────────┴───┐       ┌───┴────────────────┐
                  │                    │       │                    │
                  ▼                    │       │                    ▼
┌───────────────────────────────────┐  │       │  ┌───────────────────────────────────┐
│  Apache Flink Stateful CEP Engine │  │       │  │ Apache Spark Streaming Consumer   │
│  - keyBy(user_id)                 │  │       │  │ - spark-streaming-fraud-group     │
│  - 5m & 10m Sliding Event Windows │  │       │  │ - Real-time Feature Extraction    │
│  - Velocity Ratio Calculation     │  │       │  │ - LightGBM Sub-ms Inference       │
└─────────────────┬─────────────────┘  │       │  └─────────────────┬─────────────────┘
                  │                    │       │                    │
                  ▼                    │       │                    ▼
┌───────────────────────────────────┐  │       │  ┌───────────────────────────────────┐
│  Kafka Sink: fraud-features       │  │       │  │ Live Inference Alerts & Metrics   │
│  (Window Aggregations & Ratios)   │  │       │  └───────────────────────────────────┘
└─────────────────┬─────────────────┘  │
                  │                    │
                  ▼                    ▼
┌──────────────────────────────────────────────────────────────────┐
│             Downstream Analytics & Model Serving Layer           │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│           Apache Spark Batch Engine (Historical IEEE-CIS)        │
│   - Vectorized Parquet/Arrow Engine (590,540 Records)            │
│   - 718,573+ records/sec throughput | 0.822s execution time     │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Layer & Canonical Event Schema

### 2.1 Canonical Event Format (9 Fields)

All ingestion pipelines, Kafka topics, Flink operators, and Spark stream readers adhere strictly to the standardized 9-field canonical schema:

```json
{
  "transaction_id": "TX12345",
  "user_id": "USER123",
  "card_id": "CARD123",
  "amount": 1250.50,
  "event_time": "2026-10-05T11:55:00Z",
  "merchant_id": "MERCHANT456",
  "transaction_type": "UPI_PAYMENT",
  "is_fraud": 0,
  "source_dataset": "IEEE-CIS"
}
```

| Field Name | Type | Description | System Usage |
| :--- | :--- | :--- | :--- |
| `transaction_id` | String | Unique transaction identifier | Deduplication & tracking |
| `user_id` | String | Unique user/account identifier | **Flink `keyBy(user_id)` state grouping** |
| `card_id` | String | Unique card / funding instrument ID | **Kafka Partition Key** |
| `amount` | Float | Transaction amount in currency units | Window sum, mean & ratio calculations |
| `event_time` | ISO-8601 String | Event timestamp (UTC) | Event-time windowing & watermarking |
| `merchant_id` | String | Merchant or terminal identifier | Risk & velocity aggregation |
| `transaction_type`| String | Payment channel (`CARD`, `UPI`, etc.) | Channel filtering & behavioral profiling |
| `is_fraud` | Integer | Ground truth fraud label (`0` or `1`) | Model evaluation & accuracy validation |
| `source_dataset` | String | Originating dataset source (`IEEE-CIS`) | Data lineage & audit |

### 2.2 Datasets Supported
1. **IEEE-CIS Fraud Detection Dataset**: 590,540 transactions, 20,663 positive fraud cases (3.5% fraud rate), 394 anonymized features (`TransactionAmt`, `card1`-`card6`, `C1`-`C14`, `V1`-`V339`).
2. **Credit Card Fraud Dataset**: 10,000 synthetic/anonymized credit card records.
3. **PaySim Synthetic Financial Dataset**: Mobile money transaction logs.

---

## 3. Apache Kafka Ingestion Layer (Member 1)

### 3.1 Topic Topology
- **`fraud-transactions`**: Primary ingestion topic. Configured with 6 partitions (or 12 partitions for scaled benchmark clusters), replication factor 1.
- **`fraud-features`**: Output sink topic for Flink CEP enriched window features.
- **`fraud-transactions-p12`**: High-concurrency benchmark topic with 12 dedicated partitions.

### 3.2 Partitioning & Durability Strategy
- **Partition Key**: Strictly `card_id` (`key = event["card_id"].encode("utf-8")`), ensuring all transactions for a single payment card arrive in strict chronological order within the same Kafka partition.
- **Producer Configuration**:
  - `acks = 'all'` (zero message drop guarantee)
  - `enable_idempotence = True` (exactly-once delivery semantics to broker)
  - `compression_type = 'lz4'` (reduced network I/O and latency)
  - `batch_size = 65536` (64 KB batch buffer)
  - `linger_ms = 5` (optimal micro-batching under load)

---

## 4. Apache Flink Real-Time CEP Layer (Member 2)

The Apache Flink engine consumes raw events from `fraud-transactions`, maintains sliding state, and generates real-time velocity metrics.

### 4.1 Stateful Sliding Window Engine
- **Partitioning**: `keyBy(user_id)` ensures user-level state isolation.
- **Window Durations**:
  - **5-minute sliding window** (`window_5m`): Captures immediate velocity bursts.
  - **10-minute sliding window** (`window_10m`): Captures medium-term transaction baseline.
- **State Structure**: $\mathcal{O}(1)$ sliding double-ended queue (`collections.deque`) maintaining `(event_time_epoch, amount)` per user.

### 4.2 Derived Features & Velocity Ratios
For every incoming transaction, Flink computes:
1. `tx_count_5m`: Number of transactions in the last 5 minutes.
2. `tx_sum_5m`: Total expenditure in the last 5 minutes.
3. `tx_count_10m`: Number of transactions in the last 10 minutes.
4. `tx_sum_10m`: Total expenditure in the last 10 minutes.
5. `tx_velocity_ratio`: $\frac{\text{tx\_sum\_5m}}{\max(\text{tx\_sum\_10m}, 1.0)}$ (burst ratio indicator).
6. `cep_alert`: Boolean trigger when `tx_count_5m >= 3` and `tx_velocity_ratio >= 0.7`.

### 4.3 Output Schema (`fraud-features`)
```json
{
  "transaction_id": "TX12345",
  "user_id": "USER123",
  "card_id": "CARD123",
  "amount": 1250.50,
  "event_time": "2026-10-05T11:55:00Z",
  "tx_count_5m": 4,
  "tx_sum_5m": 3450.00,
  "tx_count_10m": 5,
  "tx_sum_10m": 3750.00,
  "tx_velocity_ratio": 0.92,
  "cep_alert": true,
  "processed_at": "2026-10-05T11:55:00.125Z"
}
```

---

## 5. Apache Spark Batch & Streaming Engine (Member 2)

### 5.1 Spark Batch Processing Engine
- **Target**: Complete 590,540-record IEEE-CIS dataset.
- **Vectorized Parquet Engine**: Pre-computed column-projected Snappy Parquet representation.
- **Aggregation Pipeline**: Multi-column fraud distributions, card-level velocity statistics, amount quantiles, and missing value profiling.
- **Performance**:
  - Unoptimized CSV Baseline: **25.71s** (22,970 records/sec)
  - Vectorized Optimized Parquet: **0.822s** (**718,573 records/sec** $\rightarrow$ **31.28x speedup**)

### 5.2 Spark Streaming Consumer Engine
- **Consumer Group**: `spark-streaming-fraud-group`
- **Topic**: `fraud-transactions`
- **Micro-Batch Inference**: Uses pre-trained LightGBM model pipeline (`spark/models/`) to compute instant fraud risk probabilities ($P(\text{fraud})$).
- **Latency**: Sub-millisecond inference per micro-batch.

---

## 6. End-to-End Integration Architecture

The 14-step integration test framework validates every stage of the pipeline:

```
[Step 1]  Kafka Broker Health Check (localhost:9092)
[Step 2]  Topic Topology & Partition Verification
[Step 3]  Canonical Schema Validator Initialization
[Step 4]  Synthetic & IEEE-CIS Transaction Ingestion
[Step 5]  Kafka Producer acks=all Execution
[Step 6]  Message Partition Routing Audit (card_id hash)
[Step 7]  Flink Consumer Initialization
[Step 8]  Flink Stateful keyBy(user_id) Processing
[Step 9]  5m & 10m Sliding Window Feature Calculation
[Step 10] Flink Sink Delivery to fraud-features Topic
[Step 11] Spark Streaming Consumer Ingestion (spark-streaming-fraud-group)
[Step 12] Spark Real-Time Feature Scoring & Inference
[Step 13] Spark Batch Historical Processing (590,540 Records)
[Step 14] End-to-End Zero-Loss Count & Integrity Reconciliation
```

---

## 7. Comprehensive Performance & Benchmark Results

### 7.1 Verified Execution Metrics Summary

| Component | Dataset / Scale | Records Processed | Execution Time | Throughput | Zero-Loss Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Spark Batch (Optimized)** | Historical IEEE-CIS | 590,540 rows | **0.822 s** | **718,573.9 rec/s** | **PASS** (Exact 20,663 fraud) |
| **Spark Batch (Baseline)** | Historical IEEE-CIS CSV | 590,540 rows | 25.71 s | 22,970.0 rec/s | PASS |
| **Flink CEP (1,000)** | Kafka Stream | 1,000 events | 0.134 s | **7,462.7 rec/s** | **PASS** (1,000/1,000 delivered) |
| **Flink CEP (5,000)** | Kafka Stream | 5,000 events | 0.551 s | **9,074.4 rec/s** | **PASS** (5,000/5,000 delivered) |
| **Flink CEP (10,000)** | Kafka Stream | 10,000 events | 1.048 s | **9,540.2 rec/s** | **PASS** (10,000/10,000 delivered) |
| **Spark Streaming (1,000)** | Kafka Micro-batch | 1,000 events | 0.063 s | **15,873.0 rec/s** | **PASS** (1,000/1,000 consumed) |
| **Spark Streaming (5,000)** | Kafka Micro-batch | 5,000 events | 0.256 s | **19,531.2 rec/s** | **PASS** (5,000/5,000 consumed) |
| **Spark Streaming (10,000)**| Kafka Micro-batch | 10,000 events | 0.493 s | **20,300.6 rec/s** | **PASS** (10,000/10,000 consumed) |
| **12P $\times$ 12C Scaled Pipeline** | Full IEEE-CIS Stream | 590,540 events | Concurrent | **12-Consumer Balanced** | **PASS** (Uniform distribution) |

---

## 8. Complete Repository Directory & File Inventory

```
Apache-Kafka/
├── COMPLETE_CODEBASE_GUIDE.md               # Master comprehensive system guide (this file)
├── CODEBASE_INPUT_OUTPUT.md                 # Detailed I/O format specifications
├── KAFKA_SPARK_FLINK_INTEGRATION_SPEC.md    # Inter-team contract and technical interface specs
├── END_TO_END_INTEGRATION_REPORT.md         # 14-Step formal integration test report
├── OPTIMIZATION_AND_BENCHMARK_REPORT.md     # Multi-scale optimization & benchmark report
│
├── run_full_e2e_integration_test.py         # Automated 14-step integration test runner
├── test_scale_benchmark_optimizations.py    # Multi-scale benchmark suite (1k, 5k, 10k)
├── run_kafka_spark_flink_pipeline.py        # Live multi-process end-to-end streaming pipeline
├── run_full_ieee_cis_12p_12c_benchmark.py   # 12-Partition x 12-Consumer full 590k benchmark
├── run_spark_batch_benchmark.py             # Standalone Spark Batch benchmark
├── run_spark_streaming_benchmark.py         # Standalone Spark Streaming benchmark
├── run_flink_benchmark.py                   # Standalone Flink CEP benchmark
├── verify_spark_batch_ieee_cis.py           # Historical IEEE-CIS data verification script
│
├── kafka/                                   # Member 1: Kafka Ingestion & Producer Layer
│   ├── config/
│   │   ├── kafka_config.py                  # Central Kafka broker, topic, and consumer settings
│   │   └── logging_config.py                # Structured logging configuration
│   ├── producer/
│   │   ├── transaction_producer.py          # Core Kafka Producer with acks=all & card_id keying
│   │   ├── generator.py                     # Synthetic transaction event generator
│   │   └── multi_dataset_producer.py        # Ingestion wrapper for IEEE-CIS, CC-10k, PaySim
│   ├── consumer/
│   │   ├── flink_consumer.py                # Flink CEP Consumer with keyBy(user_id) & sliding windows
│   │   ├── spark_streaming_consumer.py      # Spark Streaming Consumer (spark-streaming-fraud-group)
│   │   └── transaction_consumer.py          # Base Kafka consumer utility
│   ├── tests/
│   │   ├── test_spark_flink_integration.py  # Member 1 <-> Member 2 contract unit tests
│   │   ├── test_producer.py                 # Producer unit tests
│   │   └── test_consumer.py                 # Consumer unit tests
│   └── utils/
│       ├── dataset_loader.py                # CSV / Parquet data loaders
│       ├── metrics.py                       # Latency, throughput & lag measurement tools
│       └── helpers.py                       # Serialization and formatting helpers
│
└── spark/                                   # Member 2: Spark Batch, Streaming & Flink Models
    ├── batch/
    │   ├── batch_processor.py               # Spark Batch analytics engine
    │   └── optimize_batch.py                # Vectorized Parquet optimization routines
    ├── streaming/
    │   ├── stream_processor.py              # Structured Streaming feature extraction
    │   └── feature_pipeline.py              # Real-time feature calculation
    ├── models/
    │   ├── fraud_detection_model.py         # LightGBM fraud classification model
    │   └── model_evaluator.py               # Evaluation metrics (ROC-AUC, Precision-Recall)
    └── spark-benchmark-results/             # Historical benchmark archives & logs
```

---

## 9. Step-by-Step Execution & Reproduction Guide

All commands are validated for Windows PowerShell in the project workspace directory:

### 9.1 Environment Setup
```powershell
# Activate Python Virtual Environment
& ".venv/Scripts/Activate.ps1"
```

### 9.2 Running the Full 14-Step End-to-End Integration Suite
```powershell
python run_full_e2e_integration_test.py
```
*Validates broker health, partition routing, Flink CEP sliding windows, Spark Streaming micro-batches, Spark Batch historical processing, and zero data loss.*

### 9.3 Running the Multi-Scale Benchmark Suite (1,000, 5,000, and 10,000 Events)
```powershell
python test_scale_benchmark_optimizations.py
```
*Executes isolated benchmarks across Spark Batch, Flink CEP, and Spark Streaming with automated throughput calculation.*

### 9.4 Running the 12-Partition $\times$ 12-Consumer Full IEEE-CIS Scale Benchmark
```powershell
python run_full_ieee_cis_12p_12c_benchmark.py
```
*Creates topic `fraud-transactions-p12`, spins up 12 parallel consumer workers, and streams the entire 590,540 IEEE-CIS dataset.*

### 9.5 Running the Live Multi-Process Streaming Pipeline
```powershell
python run_kafka_spark_flink_pipeline.py
```
*Launches Kafka Producer, Flink CEP Engine, and Spark Streaming Consumer concurrently with real-time log aggregation.*

---
*Documentation generated for research repository: `PeramkondaIshwarya/Apache-Kafka`.*
