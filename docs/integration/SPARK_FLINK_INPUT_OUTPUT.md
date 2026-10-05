# Input and Output Specification

## 1. Main Inputs

### A. Raw Datasets (Data Source Layer)
The system ingests transaction records from raw CSV datasets located in `Datasets/`:

| Dataset | Path | Records | Columns / Features | Target Label |
| :--- | :--- | :--- | :--- | :--- |
| **IEEE-CIS Fraud Detection** *(Primary)* | `Datasets/IEEE CIS-.../IEEE CIS/train_transaction.csv` | 590,540 | 434 columns | `isFraud` (0 or 1) |
| **Credit Card Fraud** *(Secondary)* | `Datasets/Credit card Fraud detection/credit_card_fraud_10k.csv` | 10,000 | 10 columns | `is_fraud` (0 or 1) |
| **PaySim Mobile Money** | `Datasets/Paysim/paysim dataset.csv` | 6,362,620 | 11 columns | `isFraud` (0 or 1) |

### B. Ingestion Stream Inputs (Kafka Producer)
- **Message Key**: `card1` or `card_id` (used for deterministic card-level partition routing).
- **Producer Configuration**:
  - Broker: `localhost:9092`
  - Replay rate: Configurable (`--rate` parameter, e.g., 1,000–2,000 tx/sec).
  - Batch size / count: Configurable (`--count` parameter).

---

## 2. Main Outputs

### A. Kafka Ingestion Stream (Transport Output)
- **Topics**: `ieee_cis_transactions`, `transactions_p3`, `transactions_p6`, `transactions_p12`.
- **Serialization**: UTF-8 JSON.
- **Payload Modes**:
  - **Standard / Compact JSON**:
    ```json
    {
      "TransactionID": 2987000,
      "isFraud": 0,
      "TransactionDT": 86400,
      "TransactionAmt": 68.50,
      "ProductCD": "W",
      "card1": "13926",
      "card2": "150.0",
      "card3": "150.0",
      "card4": "discover",
      "card5": "142.0",
      "card6": "credit",
      "addr1": "315.0",
      "addr2": "87.0",
      "P_emaildomain": "gmail.com",
      "R_emaildomain": "",
      "timestamp": 1791130000.123
    }
    ```
  - **Full Payload Mode**: All 434 raw features (`C1..C14`, `D1..D15`, `M1..M9`, `V1..V339`).

### B. Downstream Processing Outputs

1. **Apache Spark (Micro-batch ML Inference & Analytics)**:
   - **Feature Dataframe**: Real-time parsed columns with card-partitioned keys.
   - **Model Inference Scores**: Fraud probability / risk predictions from LightGBM (`E1`).
   - **Sink Targets**: Streaming Console output / Parquet storage sinks / Alert queues.

2. **Apache Flink (Low-Latency Stream & CEP)**:
   - **Velocity Alerts**: Flags for rapid transactions (e.g., `> 3 transactions within 60 seconds` per card).
   - **Stateful Anomaly Indicators**: Sliding-window aggregations and rule-based fraud signals.
