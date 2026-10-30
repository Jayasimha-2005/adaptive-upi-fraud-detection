# Real-Time Financial Fraud Detection using Apache Kafka
## Member 1 – Kafka / Event Ingestion Layer (Dataset: `Datasets/Base.csv`)

This module implements and benchmarks the **Apache Kafka** event ingestion layer for an adaptive real-time financial fraud detection system. It provides end-to-end event streaming, key-based partitioning, fault tolerance, and offset recovery using the **NeurIPS Bank Account Fraud (BAF) suite (`Datasets/Base.csv`)**.

---

### Core Research Objective
> **"How can financial transactions be reliably ingested, distributed, and consumed in a scalable real-time fraud detection system?"**

📄 **Full Member 1 Research Report:** See [`reports/MEMBER1_RESEARCH_REPORT.md`](reports/MEMBER1_RESEARCH_REPORT.md) for deep-dive theoretical analyses, academic literature citations, fault-recovery proofs, and full benchmark metrics.  
📊 **Final Deliverables Specification:** See [`reports/MEMBER1_FINAL_DELIVERABLES.md`](reports/MEMBER1_FINAL_DELIVERABLES.md) for architecture diagrams, component contracts, and Spark/Flink downstream readiness.

---

## 1. System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                 Financial Benchmark Dataset                 │
│              Datasets/Base.csv (1,000,000 Records)          │
│       • NeurIPS Bank Account Fraud (BAF) Ground-Truth Suite │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│             Transaction Generator & Normalizer              │
│            kafka/transaction_generator/                     │
│  - dataset_loader.py: Streams Base.csv & normalizes schema  │
│  - transaction_generator.py: Configurable rate & card pool  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              Kafka Producer Layer (Member 1)                │
│            kafka/producer/producer.py                       │
│    - Partition Key: card_id (Strict FIFO per Card)          │
│    - Durability: acks=all, Idempotence=True, Retries=5      │
│    - Batching: batch.size=64KB, linger.ms=5                 │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│             Apache Kafka Cluster (KRaft Mode)               │
│       Topics: transactions, transactions_p[1,3,6]           │
│       Brokers: localhost:9092                               │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│             Kafka Consumer Group Engine                     │
│       kafka/consumer/fraud_detection_consumer.py            │
│   - Manual Offset Commits (At-Least-Once Delivery)          │
│   - Idempotent Transaction Deduplication (Zero Loss)        │
│   - Real-Time Risk Scoring & Ground Truth Verification      │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure

```text
kafka/
│
├── config/
│   ├── config.py                           # Global Kafka broker configuration
│   └── __init__.py
│
├── transaction_generator/
│   ├── dataset_loader.py                   # Base.csv streaming normalizer
│   ├── transaction_generator.py            # Dataset-powered transaction generator
│   └── __init__.py
│
├── producer/
│   ├── producer.py                         # Primary financial transaction producer
│   ├── benchmark_producer.py               # Throughput & latency benchmarking
│   ├── benchmark_producer_async.py         # Async pipelined high-rate benchmark
│   ├── partition_scaling_benchmark.py      # 1 vs 3 vs 6 partition producer
│   ├── ordering_key_producer.py            # Key-partition order verification
│   ├── idempotence_test_producer.py        # Idempotent deduplication test
│   ├── producer_failure_test.py            # Producer network outage test
│   └── __init__.py
│
├── consumer/
│   ├── fraud_detection_consumer.py         # Real-time risk detection engine
│   ├── consumer.py                         # Standard consumer with manual commit
│   ├── failure_test_consumer.py            # Consumer crash & offset recovery test
│   ├── multiprocess_parallel_consumer_benchmark.py # Multi-worker consumer scaling
│   └── __init__.py
│
├── tests/
│   ├── test_member1_pipeline.py            # Unit & contract tests (Base.csv)
│   ├── test_kafka_connection.py            # Broker health & topic verification
│   └── test_spark_flink_integration.py     # Schema contract tests
│
├── reports/
│   ├── MEMBER1_RESEARCH_REPORT.md          # Comprehensive PhD-level research report
│   ├── MEMBER1_FINAL_DELIVERABLES.md       # Final architectural deliverables
│   └── benchmarks/
│       └── member1_dataset_experiment_results.csv # Recorded experiment metrics
│
├── run_member1_dataset_experiments.py      # Automated 4-experiment test runner
└── README.md
```

---

## 3. Canonical Financial Transaction Schema (`Base.csv`)

Every transaction streamed from `Datasets/Base.csv` matches the requested schema:

```json
{
  "transaction_id": "TX100001",
  "card_id": "CARD041",
  "amount": 102.45,
  "merchant_id": "M126",
  "timestamp": "2026-10-19T16:31:32",
  "epoch_timestamp": 1792427492.1069,
  "device_type": "web",
  "country": "IN",
  "is_fraud": false,
  "dataset_source": "base.csv",
  "income": 0.3,
  "customer_age": 40,
  "proposed_credit_limit": 1500.0,
  "payment_type": "AA",
  "credit_risk_score": 163,
  "velocity_6h": 13096.04,
  "velocity_24h": 7850.96,
  "device_os": "linux",
  "source": "INTERNET",
  "foreign_request": 0
}
```

---

## 4. How to Run the Pipeline

### Step 1: Verify Kafka Broker Connection
```powershell
python kafka/tests/test_kafka_connection.py
```

### Step 2: Run Unit & Contract Tests
```powershell
python -m pytest kafka/tests/test_member1_pipeline.py -v
```

### Step 3: Run the Transaction Stream Generator
```powershell
# Stream 10 transactions from Base.csv at 5 tx/sec
python kafka/transaction_generator/transaction_generator.py --dataset Base.csv --count 10 --rate 5 --cards 10
```

### Step 4: Stream Transactions into Kafka Topic
```powershell
# Produces transactions into topic 'transactions' with acks=all and idempotence
python kafka/producer/producer.py --topic transactions --dataset Base.csv --count 50 --rate 20
```

### Step 5: Consume & Detect Fraud in Real Time
```powershell
python kafka/consumer/fraud_detection_consumer.py --topic transactions --max-messages 50
```

---

## 5. Automated Experiments Execution & Benchmark Results

### 🚀 Run All 4 Experiments Automatically
```powershell
python kafka/run_member1_dataset_experiments.py --dataset Base.csv --count 300
```

### Benchmark Deliverable Table (Dataset: `Base.csv`)

```text
================================================================================
  FINAL DELIVERABLE: PARTITION SCALING BENCHMARK TABLE
  (Dataset: Base.csv - NeurIPS Bank Account Fraud Suite)
================================================================================
Partitions   | Throughput (tx/s)    | P95 Latency      | Lost Events 
--------------------------------------------------------------------------------
1            | 976.17 tx/s          | 195.53 ms        | 0           
3            | 695.73 tx/s          | 303.89 ms        | 0           
6            | 957.33 tx/s          | 169.77 ms        | 0           
================================================================================
```

### Experiment Summary:
1. **Experiment 1 (Throughput & Latency Scaling):**
   - 100 tx/s: 97.97 tx/s sustained, 0 loss
   - 500 tx/s: 490.71 tx/s sustained, 0 loss
   - 1,000 tx/s: 905.44 tx/s sustained, P95: 176.68 ms, 0 loss
   - 5,000 tx/s: 981.24 tx/s sustained, P95: 173.40 ms, 0 loss
2. **Experiment 2 (Partition Scaling & Parallelism):**
   - 1 Partition (1 Consumer): 976.17 tx/s, Strict FIFO per card verified
   - 3 Partitions (3 Consumers): 695.73 tx/s, Strict FIFO per card verified
   - 6 Partitions (6 Consumers): 957.33 tx/s, Balanced partition distribution, Strict FIFO verified
3. **Experiment 3 (Consumer Failure & Offset Recovery):**
   - Consumer crashed before offset commit.
   - Restarted consumer in same group safely replayed uncommitted messages.
   - Handled via idempotent deduplication cache: **0 lost events (PASS)**.
4. **Experiment 4 (Producer Failure & Retry Recovery):**
   - Producer abruptly stopped midway and restarted with `enable_idempotence=True`.
   - Continued sequence without gap or duplicate log corruption: **PASS**.
