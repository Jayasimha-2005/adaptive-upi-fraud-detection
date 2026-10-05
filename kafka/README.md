# Real-Time Financial Fraud Detection using Apache Kafka
## Member 1 – Kafka / Event Ingestion Layer (Dataset-Driven)

This project implements and evaluates **Apache Kafka** as the event ingestion and streaming layer for a real-time financial fraud detection system using real benchmark datasets (IEEE-CIS Fraud Detection, Credit Card Fraud 10k, PaySim).

---

### Research Question
> **"How can financial transactions be reliably ingested, distributed, and consumed in a scalable real-time fraud detection system?"**

📄 **Full Member 1 Deliverables & Report:** See [MEMBER1_FINAL_DELIVERABLES.md](MEMBER1_FINAL_DELIVERABLES.md) for complete technical answers, architecture diagrams, benchmark tables, and research conclusions.  
🔌 **Spark & Flink Integration Specification:** See [KAFKA_SPARK_FLINK_INTEGRATION_SPEC.md](KAFKA_SPARK_FLINK_INTEGRATION_SPEC.md) for the exact Input/Output schemas, message contracts, and PySpark/Flink integration code.

---

## 1. System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│             Real-World Financial Datasets                   │
│   (IEEE-CIS 590k TXs / Credit Card Fraud 10k / PaySim)     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│          Dataset Loader & Stream Normalizer                 │
│      (Standardizes transaction_id, card_id, is_fraud)       │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              Kafka Producer Layer (Member 1)                │
│    - Partition Key: card_id (Guarantees Strict Ordering)    │
│    - Durability: acks=all, Idempotence=True, Retries=5      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│             Apache Kafka Cluster (KRaft Mode)               │
│               Topic: transactions (1, 3, 6 Partitions)      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│             Kafka Consumer Group (Member 1)                 │
│   - Manual Offset Commits (At-Least-Once Delivery)          │
│   - Duplicate Detection & Rebalance Fault Recovery          │
│   - Real-Time Fraud Detection & Ground-Truth Verification   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure

```text
Apache-Kafka/
│
├── Datasets/
│   ├── IEEE CIS-.../
│   │   └── train_transaction.csv           # 590,540 real transactions
│   ├── Credit card Fraud detection/
│   │   └── credit_card_fraud_10k.csv       # 10,000 real credit card transactions
│   └── Paysim/
│       └── paysim dataset.csv              # Mobile money fraud records
│
├── kafka/
│   ├── transaction_generator/
│   │   ├── dataset_loader.py               # Dataset discovery & stream normalizer
│   │   └── transaction_generator.py        # Dataset & synthetic stream generator
│   │
│   ├── producer/
│   │   ├── producer.py                     # Primary dataset stream producer
│   │   ├── benchmark_producer.py           # Throughput & latency benchmarking
│   │   ├── partition_scaling_benchmark.py  # 1 vs 3 vs 6 partition scaling
│   │   ├── acks_test_producer.py           # acks=0 vs acks=1 vs acks=all
│   │   ├── ordering_key_producer.py        # Key-partition order verification
│   │   ├── idempotence_test_producer.py    # Idempotent deduplication test
│   │   ├── producer_failure_test.py        # Producer network fault test
│   │   └── ieee_cis_replay_producer.py     # IEEE-CIS high-throughput replay
│   │
│   ├── consumer/
│   │   ├── consumer.py                     # Primary consumer with manual offset commit
│   │   ├── fraud_detection_consumer.py     # Real-time fraud detection engine
│   │   ├── benchmark_consumer.py           # Consumer latency/throughput bench
│   │   ├── failure_test_consumer.py        # Consumer crash & recovery test
│   │   └── ieee_cis_consumer.py            # IEEE-CIS real-time stream monitor
│   │
│   ├── config/
│   │   └── config.py                       # Global Kafka broker configuration
│   ├── tests/
│   │   └── test_kafka_connection.py        # Cluster connection & health test
│   │
│   └── run_member1_dataset_experiments.py  # Automated experiment runner
│
└── member1_dataset_experiment_results.csv  # Benchmark output records
```

---

## 3. How to Run the System with Datasets

### Step 1: Verify Kafka Broker Connection
```powershell
python kafka/tests/test_kafka_connection.py
```

### Step 2: Start Real-Time Fraud Detection Consumer
```powershell
python kafka/consumer/fraud_detection_consumer.py --topic transactions
```

### Step 3: Stream Real Dataset Transactions
```powershell
# Stream IEEE-CIS dataset (Default)
python kafka/producer/producer.py --dataset ieee_cis --count 100 --rate 20

# Or stream Credit Card Fraud 10k dataset
python kafka/producer/producer.py --dataset credit_card_10k --count 100 --rate 20
```

---

## 4. Running Member 1 Experiments on Datasets

### 🚀 Automated Full Benchmark Suite
Run all experiments on the dataset and generate benchmark results automatically:
```powershell
python kafka/run_member1_dataset_experiments.py --dataset ieee_cis
```

---

### Individual Experiments:

#### Experiment 1 — Throughput & Latency Scaling (100, 500, 1000, 5000 tx/sec)
```powershell
# 100 tx/sec
python kafka/producer/benchmark_producer.py --topic exp1_throughput --rate 100 --count 1000

# 500 tx/sec
python kafka/producer/benchmark_producer.py --topic exp1_throughput --rate 500 --count 2000

# 1,000 tx/sec
python kafka/producer/benchmark_producer.py --topic exp1_throughput --rate 1000 --count 2000

# 5,000 tx/sec
python kafka/producer/benchmark_producer.py --topic exp1_throughput --rate 5000 --count 5000
```

#### Experiment 2 — Partition Scaling (1 vs 3 vs 6 Partitions)
```powershell
# 1 Partition
python kafka/producer/partition_scaling_benchmark.py --topic transactions_p1 --count 2000 --rate 200

# 3 Partitions
python kafka/producer/partition_scaling_benchmark.py --topic transactions_p3 --count 2000 --rate 200

# 6 Partitions
python kafka/producer/partition_scaling_benchmark.py --topic transactions_p6 --count 2000 --rate 200
```

#### Experiment 3 — Consumer Failure & Offset Recovery
```powershell
# Terminal 1: Start consumer set to simulate crash after 1 message before commit
python kafka/consumer/failure_test_consumer.py --topic transactions --group fail-test-grp --consumer-id C1 --crash-after 1

# Terminal 2: Send dataset transaction
python kafka/producer/producer.py --topic transactions --count 1

# Terminal 1: Restart consumer (re-reads uncommitted offset with zero message loss)
python kafka/consumer/failure_test_consumer.py --topic transactions --group fail-test-grp --consumer-id C2 --crash-after 5
```

#### Experiment 4 — Producer Failure & Retry Continuity
```powershell
python kafka/producer/producer_failure_test.py --topic producer_failure_test --count 50
```

---

## 5. Comprehensive Benchmark Results Table

### A. Parallel Consumer Scaling Benchmark (`member1_parallel_consumer_results.csv`)

| Partitions | Total Consumers | Active / Standby Workers | Producer Rate | Aggregate Consumer Throughput | Consumer Lag | Lost Events |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 Partition** | **1 Consumer** | 1 Active / 0 Standby | 996.67 tx/s | **695.76 tx/sec** | 0 msgs | **0 (0.0%)** |
| **3 Partitions** | **3 Consumers** | 3 Active / 0 Standby | 999.69 tx/s | **3,142.90 tx/sec** | 0 msgs | **0 (0.0%)** |
| **6 Partitions** | **6 Consumers** | 6 Active / 0 Standby | 1,997.60 tx/s | **8,905.49 tx/sec** | 0 msgs | **0 (0.0%)** |
| **6 Partitions** | **12 Consumers** | 6 Active / 6 Standby | 1,997.70 tx/s | **7,418.40 tx/sec** | 0 msgs | **0 (0.0%)** |
| **12 Partitions**| **12 Consumers** | 12 Active / 0 Standby | 2,996.70 tx/s | **3,692.31 tx/sec** | 0 msgs | **0 (0.0%)** |

---

### B. Producer Ingestion & Latency Benchmark (`member1_dataset_experiment_results.csv`)

| Experiment / Mode | Producer Architecture | Target Rate | Ingestion Rate | P95 Latency | Lost Events |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp 1: Single Sync ACK** | Single Thread `future.get()` | 100 tx/s | 55.82 tx/s | 30.91 ms | 0 (0.0%) |
| **Exp 1: Pipelined Streaming** | Pipelined Async Stream | 1,000 tx/s | 998.35 tx/s | 1.84 ms | 0 (0.0%) |
| **Exp 1: High-Throughput Buffer** | 64KB Batch (`linger=5ms`) | 5,000 tx/s | 1,997.60 tx/s | 0.27 ms | 0 (0.0%) |
| **Exp 1: Multi-Process Optimized** | **4 Parallel Workers (128KB Batch)** | **Max Speed** | **29,020.38 tx/s** | **0.04 ms** | **0 (0.0%)** |

---

## 6. How to Run the Parallel Consumer Benchmark

```powershell
# Run the complete multi-consumer scaling benchmark (1, 3, 6 parallel workers)
python kafka/consumer/multiprocess_parallel_consumer_benchmark.py --dataset ieee_cis
```
