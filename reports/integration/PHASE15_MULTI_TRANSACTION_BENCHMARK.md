# Phase 15: End-to-End Deterministic Multi-Transaction Benchmark Report

- **Date / Time**: `2026-10-05 21:28:08 UTC`
- **Scope**: Multi-transaction continuous streaming replay across Member 1 (Kafka), Member 2 (Flink), and Member 3 (Bridge & E1 Serving)
- **Dataset Replayed**: `train_transaction.csv` (Authorized IEEE-CIS Fraud Benchmark)
- **Transactions Evaluated**: `100`
- **Unique Cardholder Entities**: `85`

---

## 1. Acceptance Gates Verification (P15.1 – P15.18)

| Gate | Requirement | Measured Result | Verdict |
| :--- | :--- | :--- | :---: |
| **P15.1** | Deterministic Input Set | Replayed first 100 rows from `train_transaction.csv` | 🟢 **PASS** |
| **P15.2** | Complete Ingestion | 100 / 100 transactions ingested to `ieee_cis_transactions` | 🟢 **PASS** |
| **P15.3** | Monotonic Kafka Offsets | Monotonically increasing offsets across all active partitions | 🟢 **PASS** |
| **P15.4** | Partition Key Affinity | Deterministic positive hash routing preserved per `card1` | 🟢 **PASS** |
| **P15.5** | Flink Stateful Processing | 100 / 100 events processed through 5m & 10m sliding windows | 🟢 **PASS** |
| **P15.6** | Output Features Topic | Enriched velocity payloads emitted to `fraud-features` | 🟢 **PASS** |
| **P15.7** | Causal Bridge Correlation | Watermarked temporal buffer correlated velocity context | 🟢 **PASS** |
| **P15.8** | Hydration on Valid Profiles | 100 / 100 hydrated successfully with 0 fallback fabrications | 🟢 **PASS** |
| **P15.9** | Hard Hydration Gate | Incomplete transactions rejected; zero unhydrated model calls | 🟢 **PASS** |
| **P15.10** | 406 Feature Vector | Exactly 406 canonical features generated per transaction | 🟢 **PASS** |
| **P15.11** | Canonical Preprocessing | `preprocessing.joblib` executed without target labels | 🟢 **PASS** |
| **P15.12** | Canonical E1 Model | `experiments/E1_lightgbm/model.txt` invoked directly | 🟢 **PASS** |
| **P15.13** | Threshold Invariance | Constant at `0.616521` across all inferences | 🟢 **PASS** |
| **P15.14** | Zero Target Leakage | `isFraud` strictly excluded prior to feature matrix $X$ | 🟢 **PASS** |
| **P15.15** | Causal Point-in-Time | History queries enforce `ts < current_dt`; no future leakage | 🟢 **PASS** |
| **P15.16** | Deterministic Repeatability | Run 1 vs. Run 2 yielded 100% bit-for-bit identical probabilities | 🟢 **PASS** |
| **P15.17** | 100% Lineage Completeness | Complete audit trail recorded from Kafka offset to score | 🟢 **PASS** |
| **P15.18** | Zero Silent Drops | 0 unhandled exceptions; 0 unrecorded transactions | 🟢 **PASS** |

---

## 2. Benchmark Throughput & Latency Profile

| Metric | Measured Value |
| :--- | :--- |
| **Total Ingested Events** | 100 |
| **Successfully Scored Events** | 100 (100.0%) |
| **Fraud Decisions (`>= 0.616521`)** | 0 |
| **Legitimate Decisions (`< 0.616521`)** | 100 |
| **Latency (p50 / Median)** | 86.00 ms |
| **Latency (Mean)** | 87.96 ms |
| **Latency (p95)** | 98.25 ms |
| **Latency (Max)** | 141.37 ms |

---

## 3. Repeatability Audit (Run 1 vs Run 2)

- **Total Transactions Checked**: 100
- **Repeatability Status**: `🟢 PASS — 100% Identical`
- **Mismatch Count**: `0`

---

## 4. Sample Transaction Lineage Traces

```json
[
  {
    "transaction_id": "2987000",
    "card_id": "CARD-13926",
    "amount": 68.5,
    "timestamp": 86400.0,
    "raw_topic": "ieee_cis_transactions",
    "raw_partition": 0,
    "raw_offset": 0,
    "flink_features_generated": true,
    "features_topic": "fraud-features",
    "features_partition": 0,
    "features_offset": 0,
    "velocity_count_5m": 1,
    "velocity_amount_5m": 68.5,
    "correlated_velocity": true,
    "hydration_scoreable": true,
    "model_invoked": true,
    "fraud_probability": 0.027435,
    "decision": "LEGIT",
    "threshold": 0.616521,
    "latency_ms": 141.37219998519868,
    "error": null
  },
  {
    "transaction_id": "2987001",
    "card_id": "CARD-2755",
    "amount": 29.0,
    "timestamp": 86401.0,
    "raw_topic": "ieee_cis_transactions",
    "raw_partition": 4,
    "raw_offset": 0,
    "flink_features_generated": true,
    "features_topic": "fraud-features",
    "features_partition": 4,
    "features_offset": 0,
    "velocity_count_5m": 1,
    "velocity_amount_5m": 29.0,
    "correlated_velocity": true,
    "hydration_scoreable": true,
    "model_invoked": true,
    "fraud_probability": 0.041384,
    "decision": "LEGIT",
    "threshold": 0.616521,
    "latency_ms": 86.3653999986127,
    "error": null
  }
]
```

---

## 5. Certification Sign-off

The End-to-End Multi-Transaction Streaming Pipeline demonstrates deterministic stability across continuous transaction streams, strict causal windowing, and 100% feature vector integrity without modifying frozen research artifacts.
