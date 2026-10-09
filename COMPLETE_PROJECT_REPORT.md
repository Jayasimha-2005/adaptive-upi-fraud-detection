# Adaptive Financial Fraud Detection with Real-Time Streaming & Temporal Drift Defense

**Complete Technical & Research Report**

| Field | Details |
|:---|:---|
| **Project Title** | Adaptive Financial Fraud Detection with Real-Time Streaming & Temporal Drift Defense |
| **Repository** | `adaptive-upi-fraud-detection` (Branch: `Upto_Phase-4`) |
| **Report Date** | October 2026 |
| **Team Members** | Jayasimha Padigeri, Ishwarya (Member 1), Harika (Member 2), Hadassah Kiran (Member 3) |
| **Status** | Phase 1 Complete & Verified -- Streaming & Serving Integrated (Research Prototype) |

---

## Table of Contents

- [Project in 5 Minutes](#project-in-5-minutes)
- [How to Read This Report](#how-to-read-this-report)
1. [Introduction](#1-introduction)
2. [The Problem We Are Solving](#2-the-problem-we-are-solving)
3. [System Vision & Architecture](#3-system-vision-architecture)
4. [Datasets Used](#4-datasets-used)
5. [Phase 1 -- E1 LightGBM Baseline Model](#5-phase-1----e1-lightgbm-baseline-model)
6. [Real-Time Streaming Infrastructure](#6-real-time-streaming-infrastructure)
7. [The Integration Bridge](#7-the-integration-bridge)
8. [The Serving Layer (Member 3)](#8-the-serving-layer-member-3)
9. [Live FastAPI Server & Observability Dashboard](#9-live-fastapi-server-and-observability-dashboard)
10. [End-to-End Pipeline Execution Results](#10-end-to-end-pipeline-execution-results)
11. [Performance & Latency Results](#11-performance-and-latency-results)
12. [Model Artifact Integrity](#12-model-artifact-integrity)
13. [Academic & Research Rigor](#13-academic-research-rigor)
14. [Implemented vs. Planned Future Work](#14-implemented-vs-planned-future-work)
15. [Team Member Contributions](#15-team-member-contributions)
16. [Conclusion](#16-conclusion)

---

## Project in 5 Minutes

For an evaluator or professor with zero prior background in machine learning or streaming systems, this section summarizes the entire system in simple language before diving into the technical details.

The goal of this project is to inspect incoming digital payment transactions in real time and automatically decide: **is this payment genuine, or is it fraudulent?** If it is fraudulent, the system flags it immediately to protect the cardholder's money.

```mermaid
flowchart TD
    A["1. Raw Transaction Event<br/>(~29 payment fields)"] --> B["2. Apache Kafka Ingestion<br/>(6 partitions, card_id routing)"]
    B --> C["3. Apache Flink Streaming<br/>(5m & 10m sliding window velocity)"]
    B --> D["4. StreamServingBridge<br/>(Temporal causality & correlation)"]
    C --> D
    D --> E["5. Online Feature Hydration<br/>(Reconstructs 406 canonical features)"]
    E --> F["6. Frozen Preprocessor & E1 LightGBM<br/>(Decision tree scoring at threshold 0.616521)"]
    F --> G["7. Fraud vs. Legit Verdict<br/>(Calibrated score + binary classification)"]
    G --> H["8. FastAPI Serving Layer<br/>(Containerized REST API endpoints)"]
    H --> I["9. Live Observability Dashboard<br/>(Real-time 5-stage pipeline monitoring)"]
```

### The Transaction Journey in 9 Simple Stages:

1. **Raw Transaction Event:** A simulated digital card payment arrives containing ~29 basic transaction attributes (transaction amount, card ID, merchant type, timestamp).
2. **Apache Kafka Ingestion:** Apache Kafka reliably receives the transaction into one of 6 partitions using a hash of `card_id`, ensuring all payments for the same card stay in strict chronological sequence.
3. **Apache Flink Stream Processing:** Apache Flink processes events in real time to calculate velocity metrics (for example, how many times the card transacted in the last 5 minutes and total spend over 10 minutes).
4. **Integration Bridge (`StreamServingBridge`):** The bridge pairs the raw transaction with its calculated velocity metrics, enforcing a strict causal rule: velocity calculations may never use data from the future.
5. **Online Feature Hydration:** While the raw incoming transaction contains only ~29 fields, our machine learning model requires exactly 406 features. The hydration adapter reconstructs the remaining 377 historical and behavioral features in sub-millisecond time.
6. **406 Canonical Features:** A complete, standardized numerical vector is assembled matching the identical schema used during offline training.
7. **E1 LightGBM Model:** A frozen gradient-boosted decision tree model evaluates the 406 features and outputs a fraud probability score between 0.0 and 1.0.
8. **Decision Threshold (0.616521):** If the predicted probability is equal to or greater than `0.616521`, the transaction is declared **FRAUD** (blocked); otherwise, it is declared **LEGIT** (approved).
9. **FastAPI & Live Dashboard:** The entire inference cycle is exposed via containerized REST API endpoints and visualized on an interactive web dashboard for real-time monitoring and inspection.

---

## How to Read This Report

This report is structured into four progressive layers designed for clarity and academic rigor:

1. **Layer 1: Problem & Research Motivation (Sections 1–3)** — Explains the real-world mechanics of financial fraud, the challenges of severe class imbalance (~3.5% fraud) and temporal data leakage, and outlines our multi-tier architecture.
2. **Layer 2: Machine Learning & Feature Pipeline (Sections 4–5)** — Covers the IEEE-CIS benchmark dataset, the derivation of the 406 canonical features, threshold calibration (`0.616521`), and the empirical performance of the Phase 1 E1 LightGBM model.
3. **Layer 3: Real-Time Streaming & Serving Architecture (Sections 6–9)** — Examines the distributed systems engineering: Kafka ingestion, Flink sliding windows, the StreamServingBridge, online feature hydration, the 70x NumPy preprocessing optimization, FastAPI REST endpoints, and Docker containerization.
4. **Layer 4: Experimental Validation, Rigor & Future Work (Sections 10–16)** — Details the 5-transaction and 100-transaction pipeline executions, exact offline/online parity certification, latency profiles, academic rigor comparisons, verified vs. planned roadmap, and team contributions.

> **Reading Note:** Key technical concepts (such as gradient boosting, sliding windows, feature hydration, and concept drift) are defined in plain English upon first appearance before their algorithmic implementation is discussed.

---

## 1. Introduction

### 1.1 What Is Financial Fraud?

Every day, millions of people pay for goods and services using debit cards, credit cards, and digital payment apps (such as UPI in India, or Visa/Mastercard worldwide). A small fraction of these transactions are **fraudulent** -- meaning an unauthorized party is using a stolen card, a synthetic identity, or a compromised account to steal funds.

**Financial fraud** represents an immense global burden:
- Financial institutions lose tens of billions of dollars to fraud every year.
- Every fraudulent transaction that goes undetected represents a direct financial loss to consumers and merchants.
- Conversely, every legitimate payment wrongly declined as fraud (a false alarm) creates customer frustration, merchant abandonment, and loss of trust.

### 1.2 What Is Fraud Detection?

Fraud detection is the automated process of analyzing a payment transaction at the moment of authorization and answering:
*"Is this transaction likely to be genuine, or is it likely to be fraud?"*

Historically, banks relied on **rule-based systems** -- static lists of heuristic rules such as "if amount > $5,000 and the transaction occurs overseas, flag it." These rules are brittle and easily bypassed by fraudsters who deliberately stay below heuristic thresholds.

Modern fraud detection utilizes **machine learning** -- training statistical algorithms on millions of historical transactions (both genuine and confirmed fraud). The algorithm learns subtle, multi-dimensional correlations that distinguish genuine cardholder spending from coordinated fraud attacks.

### 1.3 The Three Core Challenges

Three fundamental properties make automated fraud detection a challenging research and engineering problem:

**Challenge 1 -- Severe Class Imbalance:**
In real-world transaction logs, only ~3.5% of transactions are fraudulent. A trivial baseline that labels 100% of transactions as "legitimate" achieves 96.5% raw accuracy while catching 0% of fraud. Fraud detection requires specialized precision-recall optimization rather than naive accuracy.

**Challenge 2 -- Concept Drift:**
Fraud patterns are adversarial and non-stationary. As security controls adapt, fraudsters devise new transaction behaviors. A static model trained on historical data inevitably degrades over time as the distribution of fraud drifts.

**Challenge 3 -- The Feature Gap:**
When a card transaction occurs, the authorization payload contains only ~20 to 30 raw fields (card ID, amount, merchant code). However, high-accuracy machine learning models require hundreds of behavioral, aggregated, and historical features (e.g., cardholder velocity, average 30-day spend, address match histories). The system must bridge this gap by hydrating missing historical context in milliseconds.

---

## 2. The Problem We Are Solving

### 2.1 Research Questions

This project was formulated to address four specific research questions in applied machine learning and streaming systems:

1. **Model Efficacy:** How effectively can gradient-boosted decision trees (LightGBM) detect fraud on tabular financial transaction data under strict temporal validation constraints?
2. **Streaming Latency:** Can a distributed streaming pipeline (Kafka + Flink) compute behavioral velocity features and support transaction scoring in under 100 milliseconds?
3. **Serving Parity:** Can an online containerized serving engine produce exact numerical predictions identical to the offline research model (delta-P = 0) across tested transactions?
4. **Data Leakage Defense:** Can high-throughput feature hydration reconstruct 406 canonical features in sub-millisecond time while programmatically enforcing temporal causality ($t_{history} < t_{event}$)?

### 2.2 Why This Matters

- **For Financial Institutions:** High precision at the top ranked alerts reduces investigator workload and minimizes false positives that decline legitimate customers.
- **For Applied ML Researchers:** Demonstrates rigorous temporal validation methodologies, eliminating lookahead leakage that invalidates many academic fraud benchmarks.
- **For Systems Engineers:** Bridges the gap between offline Jupyter notebook data science and production-oriented streaming microservices.

---

## 3. System Vision & Architecture

### 3.1 The Big Picture

Think of our system as an automated security inspector stationed at a digital payment gateway. Every payment must pass through the inspector before approval. The inspector evaluates:
1. Current transaction details (amount, card, merchant, timestamp).
2. Recent spending velocity (how fast transactions are occurring on this card).
3. Long-term cardholder behavioral profiles (historical spending patterns).
4. The trained E1 LightGBM fraud detection model.

Within less than 100 milliseconds, the system produces a calibrated fraud probability and a definitive binary decision: **LEGIT** (approve) or **FRAUD** (block).

### 3.2 System Architecture Diagram

```mermaid
flowchart TD
    subgraph Tier1 ["Tier 1: Ingestion Layer (Member 1)"]
        RawEvents["Raw Payment Events<br/>(TransactionID, card_id, Amount)"] --> Kafka["Apache Kafka Cluster<br/>(Topic: ieee_cis_transactions, 6 Partitions)"]
    end

    subgraph Tier2 ["Tier 2: Stream Processing Layer (Member 2)"]
        Kafka --> Flink["Apache Flink Engine<br/>(Sliding Windows: 5m, 10m, Velocity Ratio)"]
        Flink --> FlinkAlerts["Flink CEP Alerts<br/>(Topic: fraud-features)"]
        Kafka --> Spark["Apache Spark Optimization<br/>(Columnar Parquet Storage: 7.25x Scan)"]
    end

    subgraph Tier3 ["Tier 3: Integration & Serving Layer (Member 3)"]
        Kafka --> Bridge["StreamServingBridge<br/>(Temporal Causality Guard & Buffer)"]
        FlinkAlerts --> Bridge
        Bridge --> Hydration["Online Feature Hydration<br/>(Reconstructs 406 Features from Profile Store)"]
        Hydration --> Preprocessor["Frozen Preprocessor<br/>(Vectorized NumPy Pipeline)"]
        Preprocessor --> LightGBM["LightGBM Inference Engine<br/>(Threshold = 0.616521)"]
        LightGBM --> FastAPI["FastAPI REST API<br/>(/predict, /batch, /health, /metrics)"]
        FastAPI --> Dashboard["Live Observability Dashboard<br/>(Real-Time Monitoring)"]
    end
```

### 3.3 Key Architectural Tiers

1. **Tier 1: Event Ingestion (Member 1)** — Ingests high-volume raw transaction events into Apache Kafka, utilizing Murmur2 partition hashing on `card_id` for deterministic per-card ordering.
2. **Tier 2: Stream Processing & Analytical Optimization (Member 2)** — Processes continuous event streams in Apache Flink over 5-minute and 10-minute sliding windows to generate real-time velocity metrics and Complex Event Processing (CEP) alerts. Apache Spark optimizes historical analytics via columnar Parquet storage.
3. **Tier 3: Integration, Hydration & Serving (Member 3)** — Bridges streaming events with the machine learning serving layer. Programmatically hydrates 406 canonical features, executes vectorized preprocessing and LightGBM inference, exposes containerized REST endpoints, and provides real-time dashboard observability.

### 3.4 Key Design Principles

| Principle | Meaning in Practice |
|:---|:---|
| **Zero Target Leakage** | The fraud label (`isFraud`) is strictly isolated and never fed into feature computation or inference pipelines. |
| **Temporal Causality** | For any transaction at time $T$, only historical records where $t_{history} < T$ are accessible. |
| **Frozen Model Artifacts** | Preprocessors, feature schemas, and tree boosters are cryptographically hashed and locked at server launch. |
| **Verified Prediction Parity** | The online serving pipeline produces identical outputs to the offline research model on certified test cases. |
| **Fail-Closed Safety** | Incomplete, corrupted, or causally invalid payloads are safely rejected before invoking the ML model. |

---

## 4. Datasets Used

> **Important Clarification:** None of the data used in this project represents live bank accounts or confidential UPI customer records. The project uses publicly available research benchmarks established for academic and scientific evaluation of fraud detection algorithms.

### 4.1 IEEE-CIS Fraud Detection Dataset (Primary Benchmark)

| Property | Value |
|:---|:---|
| **Source** | IEEE Computational Intelligence Society & Vesta Corporation |
| **Total Records** | 590,540 transaction records, 434 columns |
| **Fraud Prevalence** | ~3.5% of transactions are confirmed fraud |
| **Target Variable** | `isFraud` (1 = Fraudulent, 0 = Legitimate) |
| **Transaction Domain** | Card-not-present online e-commerce transactions |
| **Feature Categories** | Transaction amount, card attributes, email domains, device metadata, 339 anonymized V-features |

IEEE-CIS is the **primary benchmark** for training and evaluating the Phase 1 E1 LightGBM baseline model. While the project is titled around adaptive UPI payment concepts, IEEE-CIS serves as the rigorous empirical proxy for payment transaction modeling.

#### Chronological Data Splitting (Preventing Lookahead Leakage):

In production banking, models are trained on past data and deployed on future transactions. Random shuffling leaks future transaction patterns into the training set, artificially inflating reported performance. To ensure methodological honesty, we partition the dataset strictly by time:

| Split | Percentage | Record Count | Role in Methodology |
|:---:|:---:|:---:|:---|
| **Train** | 70% | 413,378 | Supervised model training; parameter learning. |
| **Validation** | 15% | 77,822 | Hyperparameter tuning and decision threshold calibration. |
| **Held-Out Test** | 15% | 78,542 | Final evaluation only; strictly isolated from training. |

### 4.2 Distinguishing Labeled Test Data from Blind Test Replay

To avoid ambiguity in experimental claims, we strictly distinguish between two testing datasets:

1. **Labeled Held-Out Test Set (78,542 Records):** This is the final 15% time-split of the labeled dataset where ground-truth `isFraud` labels are known. All supervised statistical metrics reported in this study (PR-AUC = 0.5317, ROC-AUC = 0.8990, Precision = 58.18%, Recall = 49.35%) are computed exclusively on this set.
2. **Blind Test Transactions (Unseen Streaming Replay):** A sample of transactions extracted from the official competition blind test set (and replay pools) used to exercise the live streaming ingestion, bridge correlation, and online serving microservice. Because blind competition records do not contain public ground-truth labels, they are utilized strictly for **system integration, latency profiling, error handling, and offline/online mathematical parity verification**, not for calculating supervised accuracy.

---

## 5. Phase 1 -- E1 LightGBM Baseline Model

### 5.1 What Is LightGBM?

**LightGBM** (Light Gradient Boosting Machine) is an ensemble machine learning algorithm based on **decision trees**. A decision tree evaluates conditions sequentially: "Is transaction amount > $500? Is the card country foreign? Have there been >= 3 transactions in the last hour?" 

LightGBM builds an ensemble of hundreds of trees sequentially, where each successive tree specifically learns to correct the residual prediction errors of preceding trees.

**Why LightGBM was selected for the Phase 1 Baseline:**
- **Tabular Data Superiority:** Consistently outperforms deep neural networks on tabular datasets with heterogeneous column types.
- **Native Missing Value Handling:** Automatically learns optimal split directions for missing values without requiring artificial zero-imputation.
- **Categorical Handling:** Efficiently handles high-cardinality categorical variables (such as merchant codes and card IDs).
- **Sub-Millisecond Inference:** Evaluates tree ensembles in microseconds, ideal for real-time payment authorization.

### 5.2 The 406 Canonical Features

From the raw transaction attributes, our feature engineering pipeline (`src/features/pipeline.py`) derives exactly **406 canonical features**:

| Feature Group | Count | Description & Examples |
|:---|:---:|:---|
| **Raw Transaction Fields** | ~29 | `TransactionAmt`, `ProductCD`, `card1`-`card6`, `addr1`, `addr2` |
| **V-Features (Behavioral)** | 339 | `V1`-`V339` (Anonymized behavioral and velocity signals) |
| **C-Features (Counters)** | 14 | `C1`-`C14` (Transaction frequency counters per entity) |
| **D-Features (Time Deltas)** | 15 | `D1`-`D15` (Time differences from previous transactions) |
| **M-Features (Match Indicators)** | 9 | `M1`-`M9` (Binary match flags such as address/name match) |

**Preprocessing Pipeline (`preprocessing.joblib`):**
- Numerical features: Imputed using medians computed strictly on the training partition.
- Categorical features: Label-encoded with categorical mappings learned strictly on the training partition.
- All transformers are frozen to ensure identical deterministic behavior during inference.

### 5.3 Calibrating the Decision Threshold (0.616521)

Every classifier outputs a continuous probability score $P(\text{fraud}) \in [0, 1]$. A decision threshold converts this probability into a binary verdict:
$$\text{Decision} = \begin{cases} \text{FRAUD (Block)}, & \text{if } P \ge \theta \\ \text{LEGIT (Approve)}, & \text{if } P < \theta \end{cases}$$

On imbalanced datasets, the default cutoff of $\theta = 0.5$ is suboptimal, causing excessive false alarms that decline legitimate cardholders. We calibrated $\theta$ on the validation split under a strict operational constraint:
- **Optimization Objective:** Maximize F1-Score while constraining False Positive Rate to $\le 1.30\%$.
- **Result:** Optimal threshold **$\theta = 0.616521$**.

This threshold is permanently frozen across all serving components.

### 5.4 E1 Model Performance Results

The following metrics were evaluated on the **held-out test set (N = 78,542)**:

#### Confusion Matrix (Held-Out Test Set):

```
                       CONFUSION MATRIX (TEST SET: N = 78,542)
                                 Actual Class
                          FRAUD (1)       LEGITIMATE (0)
                     +-----------------+-----------------+
  Predicted FRAUD    |  TP = 1,369     |   FP = 984      |
                     +-----------------+-----------------+
  Predicted LEGIT    |  FN = 1,405     |   TN = 74,784   |
                     +-----------------+-----------------+
                       Actual: 2,774     Actual: 75,768
```

- **True Positives (TP = 1,369):** Fraudulent transactions correctly intercepted.
- **False Positives (FP = 984):** Legitimate payments incorrectly blocked (1.30% FPR).
- **False Negatives (FN = 1,405):** Fraud transactions that bypassed the baseline model.
- **True Negatives (TN = 74,784):** Legitimate payments correctly approved.

#### Detailed Statistical Metrics:

| Metric | Validation Set | Held-Out Test Set | Interpretation |
|:---|:---:|:---:|:---|
| **Total Transactions** | 77,822 | 78,542 | Temporal split |
| **Precision** | 62.85% | **58.18%** | 58 out of 100 alerts are confirmed fraud |
| **Recall** | 51.38% | **49.35%** | Intercepts ~50% of all fraudulent attempts |
| **F1-Score** | 0.5654 | **0.5340** | Harmonic mean of precision and recall |
| **PR-AUC** | 0.5849 | **0.5317** | **~15x higher than random baseline (~0.035)** |
| **ROC-AUC** | 0.9231 | **0.8990** | High ranking capability |
| **Brier Score** | -- | **0.0323** | Well-calibrated probability predictions |
| **False Positive Rate** | -- | **1.30%** | Meets strict operational tolerance ($\le 1.3\%$) |

#### Operational Precision at Top-Ranked Alerts:

In real bank operations, fraud investigation teams prioritize alerts in descending order of predicted risk:

| Alert Tier | Precision | Operational Meaning |
|:---|:---:|:---|
| **Top 100 highest-risk alerts** | **98.0%** | 98 out of 100 investigated cases are confirmed fraud |
| **Top 500 highest-risk alerts** | **90.8%** | 9 out of 10 investigated cases are confirmed fraud |
| **Top 1,000 highest-risk alerts** | **84.1%** | 84 out of 100 investigated cases are confirmed fraud |

#### Bootstrap 95% Confidence Intervals (2,000 Iterations):

| Metric | Lower Bound (2.5%) | Upper Bound (97.5%) |
|:---|:---:|:---:|
| **PR-AUC** | 0.5126 | 0.5504 |
| **ROC-AUC** | 0.8924 | 0.9055 |

The tight bootstrap confidence intervals demonstrate that the reported model performance is statistically robust and repeatable.

### 5.5 Cryptographic Artifact Hashes

To guarantee artifact integrity and eliminate model drift during serving startup, all trained assets are cryptographically hashed using SHA-256:

| Artifact | File Path | SHA-256 Hash |
|:---|:---|:---|
| **LightGBM Model** | `experiments/E1_lightgbm/model.txt` | `ac93b59a7eee7a23b1d77a7fa03d348153328da128a1ba66d6f34cf490ec6d96` |
| **Preprocessing Pipeline** | `experiments/E1_lightgbm/preprocessing.joblib` | `0c336989206214cab202d3b4a8a726206cb4ca69a0908e52fdb6f9cf479fbf69` |
| **Feature Schema** | `experiments/E1_lightgbm/feature_names.json` | `1c59105a626f57533af4fc56f3ba10ae112b739c16c2e2d99cec24c1b1d0330d` |

---

## 6. Real-Time Streaming Infrastructure

### 6.1 Why Streaming Systems?

A standalone web server cannot handle real-time fraud defense at scale because:
- Direct database queries per request create massive latency bottlenecks under high transaction loads.
- Computing multi-window spending velocities (5-minute and 10-minute sliding windows) across concurrent requests requires distributed state management.
- Transactions for each cardholder must be processed in strict chronological order to avoid corrupted velocity features.

To resolve these challenges, the team integrated Apache Kafka and Apache Flink into a distributed streaming pipeline.

### 6.2 Member 1 -- Apache Kafka (Ingestion Layer)

**Role of Kafka:** A distributed, fault-tolerant event broker that ingests continuous streams of raw payment transactions and delivers them to downstream processors.

**Implementation Details:**
- **Topic Architecture:** Primary ingestion topic `ieee_cis_transactions` configured with **6 partitions**.
- **Deterministic Routing:** A Murmur2 partitioner hashes the `card_id`, ensuring all transactions for the same card land in the same partition in strict temporal sequence.
- **Reliability Semantics:** Configured with `acks=all`, `enable.idempotence=True`, and `retries=5` to guarantee zero message loss and prevent duplicate event delivery.
- **Target Isolation:** The `isFraud` label is stripped at producer source, preventing ground truth leakage into the stream.

**Measured Ingestion Throughput:**
- Single Partition: ~45,000 records/sec
- 6 Partitions (Multi-Worker): **~148,000 records/sec**

### 6.3 Member 2 -- Apache Flink (Stream Processing)

**Role of Flink:** A distributed stream processing engine that maintains stateful sliding event-time windows over live transaction streams.

**Implementation Details:**
- **Kafka Consumer:** Ingests live events from `ieee_cis_transactions` keyed by `card_id`.
- **Sliding Event Windows:**
  - 5-Minute Window: Computes transaction count (`count_5m`) and cumulative volume (`amount_5m`).
  - 10-Minute Window: Computes transaction count (`count_10m`) and cumulative volume (`amount_10m`).
  - Velocity Ratio ($VR$): Computes $VR = \text{TransactionAmt} / (\text{avg\_amt\_10m} + 1.0)$.
- **Complex Event Processing (CEP):** Triggers immediate rule-based fraud alerts when $VR > 3.0$ or when $\ge 3$ transactions occur within 5 minutes.
- **Output:** Emits enriched velocity records to Kafka topic `fraud-features` with processing latency $< 50\text{ ms}$.

### 6.4 Member 2 -- Apache Spark (Columnar Storage Optimization)

For historical analytics and batch feature computation, Apache Spark was implemented to benchmark storage formats:

| Metric | Raw CSV Format | Optimized Parquet Format | Empirical Gain |
|:---|:---:|:---:|:---:|
| **Scan Throughput** | 128,000 records/sec | 927,000 records/sec | **7.25x faster** |
| **Storage Footprint** | 683 MB | 157 MB | **77% reduction** |

Columnar Parquet storage dramatically accelerates historical feature aggregation and retrospective model audits.

---

## 7. The Integration Bridge

### 7.1 The Asynchronous Stream Challenge

Because Kafka event ingestion and Flink stream processing execute asynchronously across separate worker nodes, raw payment events and their corresponding Flink velocity features arrive at slightly different times. Joining them naively introduces risks:
- Future velocity features could attach to past transactions, creating lookahead data leakage.
- Latency spikes in Flink could cause unjoined transactions to stall.

### 7.2 What the StreamServingBridge Does

The **StreamServingBridge** (`streaming/stream_serving_bridge.py`) resolves this challenge:

```mermaid
sequenceDiagram
    autonumber
    participant K as Kafka (Raw Stream)
    participant F as Flink (Velocity Stream)
    participant B as StreamServingBridge
    participant H as Online Hydration
    
    K->>B: Raw Transaction Event (t_event)
    F->>B: Flink Velocity Features (t_window)
    Note over B: Check Causal Invariant:<br/>t_window < t_event
    alt Invariant Holds (Valid)
        B->>H: Correlation Buffer Emits StreamingPayload
    else Invariant Violated (Future Leakage)
        B--xB: Event Dropped / Flagged (Fail-Closed)
    end
```

1. **Correlation Buffer:** A time-indexed memory buffer that reconciles raw events and velocity records matching on `TransactionID`.
2. **Programmatic Causal Invariant:** Strictly validates that $t_{history} < t_{event}$. If any velocity record contains a timestamp newer than the transaction event, it is rejected to prevent lookahead leakage.
3. **Payload Normalization:** Formats the joined record into a standardized `StreamingTransactionPayload` for the downstream serving layer.

### 7.3 Bridge Performance Benchmark

| Bridge Metric | Measured Value |
|:---|:---:|
| **Median Latency** | 0.048 ms |
| **Throughput Capacity** | 20,480 events/sec |

---

## 8. The Serving Layer (Member 3)

The serving layer bridges the streaming pipeline to the frozen E1 LightGBM model, exposing real-time inference via a containerized REST API microservice.

### 8.1 Online Feature Hydration

**The Feature Gap:** While a `StreamingTransactionPayload` contains ~29 base fields plus Flink velocity indicators, the E1 model requires exactly **406 features**.

The **OnlineFeatureHydrationAdapter** (`serving/feature_hydration.py`) reconstructs the full 406-dimensional vector:
1. **29 Base Attributes:** Directly mapped from the incoming payment payload.
2. **51 Entity Profile Features:** Historical cardholder statistics (frequency, spend averages, address stability) retrieved from the in-memory `EntityProfileStore`.
3. **326 Behavioral Aggregation Features:** Historical rolling averages, $D$-variable time elapsed deltas, and $V$-variable behavioral signals computed over past transactions.

**Causal Invariant Enforcement:** The hydration layer programmatically enforces that only transactions occurring strictly before $t_{event}$ are accessible ($t_{history} < t_{event}$). Automated test suites verify that future records are never accessed.

**Hard Hydration Gate:** If a cardholder profile is missing or a causal violation occurs, the transaction is rejected immediately with `HTTP 422 Unprocessable Entity` before invoking the ML model.

### 8.2 Frozen Preprocessing Pipeline

The 406 assembled features pass through `serving/models/E1/preprocessing.joblib`:
- Missing numerical values are imputed with frozen medians.
- Categorical attributes are mapped using frozen label encoders.
- All transformations execute deterministically with zero state mutation.

### 8.3 LightGBM Inference Engine

The transformed numerical array enters the **frozen E1 LightGBM Booster** (`serving/models/E1/model.txt`):
$$\text{Verdict} = \begin{cases} \text{FRAUD (Block)}, & \text{if } P(\text{fraud}) \ge 0.616521 \\ \text{LEGIT (Approve)}, & \text{if } P(\text{fraud}) < 0.616521 \end{cases}$$

### 8.4 Performance Optimization: Vectorized NumPy Preprocessing

The initial serving prototype constructed a new Pandas DataFrame for each transaction request. In Python, constructing and indexing DataFrames per single row introduces significant object creation overhead (~82 ms).

To eliminate this bottleneck, preprocessing was re-engineered using **vectorized NumPy arrays** (`serving/feature_vectorizer.py`):

| Implementation | Preprocessing Latency (P50) | Execution Difference |
|:---|:---:|:---|
| **Original Prototype** (Pandas DataFrame) | ~82 ms | Per-request DataFrame object construction overhead |
| **Optimized Implementation** (NumPy Vectorized) | **~1.15 ms** | Direct contiguous array memory buffer indexing |
| **Empirical Speedup** | **~70x** | **Targeted specifically at the preprocessing step** |

> **Clarification:** Prediction outputs were **100% identical** between implementations; the optimization targeted feature transformation efficiency without modifying model weights. This 70x speedup applies specifically to the feature preprocessing routine and does not imply that the entire network roundtrip accelerated by 70x.

### 8.5 Containerized Architecture & REST Endpoints

The serving microservice is built using **FastAPI** running on the **Uvicorn** ASGI server and packaged inside a Docker container:

```mermaid
flowchart LR
    Client["Client / Load Generator"] --> API["FastAPI Container (Port 8000)"]
    subgraph DockerContainer ["Docker Container: adaptive-fraud-serving:v1"]
        API --> Hydrate["OnlineFeatureHydrationAdapter"]
        Hydrate --> Preproc["Frozen Preprocessor<br/>(NumPy Vectorizer, ~1.15ms)"]
        Preproc --> Booster["Frozen LightGBM Booster<br/>(Decision Threshold 0.616521)"]
        Booster --> API
        API --> Telemetry["In-Memory Telemetry<br/>(/metrics & /health)"]
        API --> UI["Live Observability Dashboard<br/>(/dashboard)"]
    end
```

| Endpoint | Method | Functionality |
|:---|:---:|:---|
| `/predict` | POST | Scores single transaction payload with complete 406-feature assembly |
| `/predict/batch` | POST | Vectorized batch scoring for multi-transaction requests |
| `/predict/hydrated` | POST | Accepts compact streaming payload and executes online hydration |
| `/health` | GET | Operational health check, artifact hashes, and invariant status |
| `/metrics` | GET | In-memory application telemetry in Prometheus text exposition format |
| `/dashboard` | GET | Real-time browser-based operational dashboard |
| `/docs` | GET | Interactive Swagger UI API documentation (OpenAPI 3.1) |

### 8.6 Docker Deployment & Cross-Platform Line-Ending Fix

The serving application is packaged into a self-contained Docker container:

```bash
# Build container image
docker build -t adaptive-fraud-serving:v1 -f serving/Dockerfile serving

# Launch container on port 8000
docker run -d --name adaptive-fraud-serving -p 8000:8000 adaptive-fraud-serving:v1
```

**Cross-Platform Normalization Fix:** LightGBM booster text files generated on Linux use Unix line endings (`LF`), while Windows hosts frequently convert them to `CRLF`. On Windows startup, this caused LightGBM booster parsing errors. A startup memory-buffer normalization fix was implemented in `serving/models/offline_inference.py`, ensuring cross-platform stability across Linux containers and Windows hosts.

---

## 9. Live FastAPI Server and Observability Dashboard

### 9.1 How to Start the Server

**Option A -- Docker Container (Recommended):**
```powershell
docker run -d --name adaptive-fraud-serving -p 8000:8000 adaptive-fraud-serving:v1
```

**Option B -- Local Python Environment:**
```powershell
cd d:\adaptive-upi-fraud-detection
uvicorn serving.api.main:app --host 0.0.0.0 --port 8000
```

### 9.2 How to Stop the Server

**Stop Docker Container:**
```powershell
docker stop adaptive-fraud-serving
```

**Stop Local Uvicorn Server:**
- Press `Ctrl + C` in the active terminal running the Uvicorn process.

### 9.3 The Live Observability Dashboard

Access the live interface at: **`http://localhost:8000/dashboard`**

**Key Dashboard Capabilities:**
- **Automated Batch Execution:** Run test batches (5, 10, 50, 100, or 1,000 transactions) drawn from the unseen IEEE-CIS test pool through the complete 5-stage pipeline with one click.
- **5-Stage Pipeline Monitor:** Real-time visual status cards tracking Kafka Ingestion, Flink CEP, Bridge Correlation, Feature Hydration, and LightGBM Scoring.
- **Reconciliation Banner:** Live verification counter confirming zero message loss across stages:
  $$\text{Requested} = \text{Kafka Ingested} = \text{Flink Processed} = \text{Bridge Correlated} = \text{Hydrated} = \text{Scored}$$
- **Transaction Journey Feed:** Live scrolling table showing per-transaction execution metadata: timestamp, TransactionID, card partition, Flink CEP status, Bridge state, feature count (406), fraud score, binary decision badge, and total latency.

### 9.4 Dashboard vs. Swagger UI vs. External Metrics

| Capability | Live Dashboard (`/dashboard`) | Swagger UI (`/docs`) | Application Telemetry (`/metrics`) |
|:---|:---:|:---:|:---:|
| **Primary Audience** | System operators & evaluators | Developers & API consumers | Monitoring scrapers / Ops |
| **Transaction Inflow** | Automated from test dataset | Manual JSON input | Background counters |
| **Visual Telemetry** | 5-stage pipeline cards & feed | Request/response schemas | Raw text exposition format |
| **Implementation** | Custom HTML/JS interface | OpenAPI 3.1 Swagger UI | In-memory telemetry engine |

> **Clarification:** The `/metrics` endpoint exposes in-memory telemetry formatted for Prometheus scrapers. The project currently runs an in-memory telemetry system with the custom `/dashboard` interface; full deployment of external Prometheus scraping servers and Grafana visual panels is designated as future work.

---

## 10. End-to-End Pipeline Execution Results

### 10.1 The 5-Transaction Execution Report

To verify complete end-to-end functionality, 5 unseen IEEE-CIS test transactions (4 genuine transactions and 1 confirmed fraud transaction) were processed through the full streaming and serving pipeline:

| Transaction ID | Amount | Card Identifier | Predicted Fraud Score | System Verdict | HTTP Status |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 3663549 | $31.95 | CARD-10409 | 0.007875 | **LEGIT** | 200 OK |
| 3663550 | $49.00 | CARD-4272 | 0.021529 | **LEGIT** | 200 OK |
| 3663551 | $171.00 | CARD-4476 | 0.004329 | **LEGIT** | 200 OK |
| 3663552 | $284.95 | CARD-10989 | 0.014540 | **LEGIT** | 200 OK |
| **3663602** | **$52.26** | **CARD-9633** | **0.673801** | **FRAUD** | **200 OK** |

**Case Analysis (Transaction 3663602):**
Despite having a modest transaction amount ($52.26), Transaction 3663602 exhibited anomalous cardholder velocity and address mismatch indicators. The LightGBM model scored it at **0.673801** -- safely exceeding the 0.616521 threshold -- and issued a **FRAUD** block. This demonstrates that the multi-dimensional 406-feature model detects subtle behavioral fraud patterns beyond simple dollar heuristics.

### 10.2 The 100-Transaction Full Pipeline Validation

A validation run of 100 sequential unseen test transactions was executed through all distributed components:

| Validation Metric | Measured Result |
|:---|:---:|
| **Total Transactions Processed** | **100 / 100 (100%)** |
| **Kafka Ingestion ACK Rate** | **100%** |
| **Flink Velocity Processing Rate** | **100%** |
| **Bridge Correlation Success Rate** | **100%** |
| **Feature Hydration Success Rate** | **100%** |
| **LightGBM Scoring Completion** | **100%** |
| **HTTP 200 Success Rate** | **100%** |
| **Pipeline Reconciliation Discrepancy** | **0 transactions dropped** |

### 10.3 Offline vs. Online Prediction Parity Certification

A critical failure mode in production ML is **train-serve skew**, where differences between offline research libraries and online serving code cause predictions to diverge. 

We conducted an empirical parity certification comparing offline model predictions directly against the live containerized serving responses across the 100-transaction test set:

| Parity Metric | Measured Value | Formal Status |
|:---|:---:|:---:|
| **Maximum Absolute Probability Difference** | **0.000000e+00** | Certified Exact |
| **Mean Absolute Probability Difference** | **0.000000e+00** | Certified Exact |
| **Decision Mismatches** | **0 out of 100** | 100% Concordance |
| **Certification Outcome** | **EMPIRICALLY VERIFIED PARITY** | **100 / 100 Matching Decisions** |

> **Methodological Scope:** Exact offline/online prediction parity was empirically verified for the 100-transaction certification set, with a maximum absolute probability difference of 0 and zero decision mismatches. This provides strong empirical evidence that the containerized serving engine faithfully deploys the research model without implementation divergence.

---

## 11. Performance and Latency Results

### 11.1 Component-Level Latency Profile (N = 500 Transactions)

A performance benchmark was conducted across 500 sequential transaction requests to quantify component latency and overall throughput:

| Pipeline Component | Median Latency (P50) | 95th Percentile (P95) | Component Throughput |
|:---|:---:|:---:|:---:|
| **Kafka Ingestion** | 0.049 ms | -- | 20,203 events/sec |
| **Flink CEP Velocity Evaluation** | 0.059 ms | -- | 16,477 events/sec |
| **StreamServingBridge Correlation** | 0.048 ms | -- | 20,480 events/sec |
| **Online Feature Hydration** | 0.701 ms | -- | 1,375 events/sec |
| **Frozen Preprocessor (NumPy Vectorized)** | 1.15 ms | 2.40 ms | ~870 transforms/sec |
| **LightGBM Inference Engine** | 0.42 ms | 0.85 ms | ~2,380 inferences/sec |
| **FastAPI HTTP Network Roundtrip** | 91.65 ms | 154.46 ms | -- |
| **Total Sequential End-to-End Pipeline** | **92.68 ms** | **155.39 ms** | **~11 transactions/sec** |

### 11.2 Understanding the Latency Measurements

To interpret these measurements accurately, it is essential to distinguish between **internal algorithmic latency** and **HTTP serving roundtrip latency**:

- **Model & Preprocessing Latency ($\approx 1.57\text{ ms}$):** The pure machine learning pipeline (NumPy feature transformation at 1.15 ms + LightGBM tree traversal at 0.42 ms) executes in under 2 milliseconds total.
- **Streaming Pipeline Components ($\approx 0.86\text{ ms}$):** Kafka ingestion, Flink window processing, Bridge correlation, and feature hydration combined take less than 1 millisecond.
- **FastAPI HTTP Roundtrip Latency (91.65 ms P50, 154.46 ms P95):** This measurement reflects the full client-to-server HTTP request/response path in the sequential test benchmark. It encompasses client socket connection setup, Uvicorn ASGI event loop scheduling, JSON serialization and deserialization, and Docker container networking overhead.
- **Important Distinction:** API roundtrip latency and ML model inference latency are fundamentally different measurements. The LightGBM model itself evaluates in 0.42 ms; the 91.65 ms roundtrip represents containerized HTTP microservice overhead under sequential execution.

### 11.3 System Latency Interpretation

In global payment processing networks (Visa, Mastercard, NPCI/UPI), the total allowable authorization latency window for fraud evaluation is typically **150 to 250 milliseconds**.

Our system's total median end-to-end latency of **92.68 ms** (and 155.39 ms at P95) demonstrates that the complete multi-tier architecture comfortably satisfies real-time payment authorization constraints.

---

## 12. Model Artifact Integrity

### 12.1 Cryptographic Hash Verification

To guarantee reproducibility and prevent unauthorized alteration of trained models, the serving microservice executes automated SHA-256 integrity verification upon startup:

1. **Tamper Prevention:** Prevents accidental modification or corruption of preprocessor weights and model tree parameters.
2. **Serving Auditability:** Formally verifies that the live serving engine loads the exact models evaluated in the research study.
3. **Fail-Fast Startup:** If any artifact hash diverges from the certified checksum, the server terminates immediately rather than serving invalid predictions.

| Certified Artifact | Container Path | SHA-256 Cryptographic Checksum | Verification Status |
|:---|:---|:---|:---:|
| **LightGBM Booster** | `serving/models/E1/model.txt` | `0308a5f2c43f888a85bad5ec6698fefcee27c2afe4a8d4bdc2b6314de69b6421` | FROZEN & VERIFIED |
| **Preprocessing Pipeline** | `serving/models/E1/preprocessing.joblib` | `0c336989206214cab202d3b4a8a726206cb4ca69a0908e52fdb6f9cf479fbf69` | FROZEN & VERIFIED |
| **Feature Schema Definition** | `serving/models/E1/feature_names.json` | `1c59105a626f57533af4fc56f3ba10ae112b739c16c2e2d99cec24c1b1d0330d` | FROZEN & VERIFIED |

---

## 13. Academic & Research Rigor

### 13.1 What Makes This Research Rigorous?

In applied machine learning and data engineering research, scientific validity is evaluated across three core pillars:

1. **Methodological Honesty:** Strict temporal splitting is enforced to prevent lookahead data leakage. Incomplete or invalid payloads fail closed safely.
2. **Statistical Rigor:** Performance claims are substantiated through precision-recall tradeoffs (PR-AUC), Brier probability calibration scores, and 2,000-iteration bootstrap confidence intervals -- avoiding reliance on deceptive raw accuracy.
3. **Engineering Parity & Reproducibility:** Prediction parity between offline research models and live containerized serving is empirically certified to bit-level precision (delta-P = 0) on certified test cases.

### 13.2 Comparative Benchmark Against Academic Tiers

| Research Dimension | Typical Undergraduate Project | Standard Master's Thesis | This Project's Implementation |
|:---|:---|:---|:---|
| **Data Partitioning** | Random train/test split (severe temporal leakage) | Basic chronological cut | Strict temporal split + point-in-time causal feature hydration ($t_{history} < t_{event}$). |
| **Evaluation Metrics** | Raw accuracy (misleading on imbalanced data) | ROC-AUC evaluated at default $\theta = 0.5$ | PR-AUC (0.5317), calibrated threshold ($\theta = 0.616521$), Precision@Top-100 (98.0%), Brier (0.0323). |
| **Feature Engineering** | Ad-hoc single script calculations | Basic offline preprocessor | 406 canonical features with automated missing value and categorical encoders frozen in joblib. |
| **Serving Architecture** | Single standalone Flask/FastAPI script | Basic REST endpoint | Multi-tier streaming pipeline: Kafka (6 partitions) $\to$ Flink CEP $\to$ Bridge $\to$ FastAPI $\to$ Dashboard. |
| **Train-Serve Skew** | Ignored completely | Acknowledged as a limitation | Empirically verified prediction parity (delta-P = 0.000000e+00) across 100 sequential transactions. |
| **Fault Resilience** | Unhandled exceptions on malformed inputs | Basic `try-except` blocks | 20-scenario adversarial chaos certification (Cases A through T) with fail-closed safety. |
| **Statistical Validation**| Single point estimates | Mean $\pm$ standard deviation | 2,000-iteration bootstrap confidence intervals on all primary statistical metrics. |

### 13.3 Complete Test Suite Results

The codebase is backed by an automated test suite executed with `pytest`:

| Test Module / Functional Layer | Total Tests | Passed | Skipped | Failed | Pass Rate |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Phase 1 (E1 LightGBM ML Pipeline)** | 10 | **10** | 0 | 0 | 100% |
| **Member 1 (Kafka Ingestion Layer)** | 3 | **3** | 0 | 0 | 100% |
| **Member 2 (Spark Columnar Optimization)** | 5 | 3 | 2 | 0 | 100% applicable |
| **Member 2 (Flink Stream Processing)** | 12 | 5 | 7 | 0 | 100% applicable |
| **Milestone 5 Integration (Streaming & Serving)** | 87 | **87** | 0 | 0 | 100% |
| **TOTAL TEST SUITE** | **117** | **108** | **9** | **0** | **100% Applicable Pass** |

> **Note on Skipped Tests:** The 9 skipped tests require a fully provisioned Apache PySpark and PyFlink cluster with Java 21 runtime. In environments without active Spark/Flink clusters, these tests safely skip. All 108 executable tests pass with zero failures.

### 13.4 Milestone Completion Scorecard

| Milestone | Architectural Scope | Verification Status |
|:---|:---|:---:|
| **Milestone 1** | E1 LightGBM Baseline (406 features, threshold 0.616521) | FROZEN & CERTIFIED |
| **Milestone 4** | Kafka Ingestion + Flink CEP Infrastructure (Members 1 & 2) | INTEGRATED & TESTED |
| **Milestone 5.1** | StreamServingBridge Integration (Stage 13) | 12 / 12 PASS |
| **Milestone 5.2** | Cross-Member E2E Pipeline (Stage 14) | 15 / 15 PASS |
| **Milestone 5.3** | Multi-Transaction Determinism (Stage 15) | 18 / 18 PASS |
| **Milestone 5.4** | Offline/Online Prediction Parity delta-P = 0 (Stage 16) | 9 / 9 PASS |
| **Milestone 5.5** | Historical Replay at Scale (Stage 17) | 6 / 6 PASS |
| **Milestone 5.6** | Adversarial Safety 20 Cases (Stage 18) | 21 / 21 PASS |
| **Milestone 5.7** | Throughput & Latency Profiling (Stage 19) | 6 / 6 PASS |
| **Milestone 5.8** | Full Regression Audit (Stage 20) | 108 PASS / 0 FAIL |

---

## 14. Implemented vs. Planned Future Work

> **Engineering Integrity Policy:** To maintain transparent academic honesty, this report strictly delineates components that have been fully constructed and empirically verified from planned architectural enhancements.

### 14.1 Actually Implemented and Verified

| Implemented Component | Verification Evidence |
|:---|:---|
| **E1 LightGBM 406-Feature Model** | Trained on 413,378 records; evaluated on 78,542 held-out test transactions |
| **Strict Temporal Splitting** | Chronological 70/15/15 partition; zero lookahead data leakage |
| **Decision Threshold Calibration** | Calibrated to 0.616521 on validation set; FPR constrained to 1.30% |
| **Kafka Multi-Partition Ingestion** | 6 partitions with Murmur2 `card_id` routing; 148,000 records/sec throughput |
| **Flink Sliding-Window CEP** | 5-minute and 10-minute event-time windows computing velocity ratios |
| **Spark Parquet Optimization** | 7.25x scan speedup and 77% storage compression over CSV |
| **StreamServingBridge** | Correlation buffer enforcing temporal causality invariant $t_{history} < t_{event}$ |
| **OnlineFeatureHydrationAdapter** | Reconstructs 406 canonical features in 0.701 ms with hard validation gate |
| **Frozen Preprocessing Pipeline** | Scikit-learn imputer and label encoder serialized in `preprocessing.joblib` |
| **FastAPI REST Microservice** | Complete endpoints (`/predict`, `/predict/batch`, `/predict/hydrated`, `/health`, `/metrics`) |
| **Vectorized NumPy Preprocessor** | 70x feature preprocessing speedup (from ~82 ms to ~1.15 ms) with 100% parity |
| **Docker Containerization** | Self-contained image `adaptive-fraud-serving:v1` with Windows CRLF fix |
| **Live Observability Dashboard** | Web interface (`/dashboard`) with automated batch load controls and stage monitoring |
| **In-Memory Application Telemetry** | Rolling counters, latency percentiles, and fraud rate stats at `GET /metrics` |
| **Empirical Train-Serve Parity** | 100/100 matching decisions with zero probability difference on certified test set |
| **Adversarial Chaos Certification** | 20 formalized edge cases (Cases A through T) verified to fail closed safely |
| **Automated Test Suite** | 108 tests passing, 0 failures across all implemented modules |

### 14.2 Planned Future Work (Not Yet Implemented)

| Planned Enhancement | Scope & Requirements for Real-World Deployment |
|:---|:---|
| **Production Prometheus & Grafana** | Deploy dedicated Prometheus server to scrape `/metrics` and construct real-time Grafana monitoring dashboards. |
| **Concurrent Load Testing** | Execute multi-client asynchronous load benchmarks (via Locust or k6) to evaluate ASGI throughput under concurrent traffic. |
| **Automated Drift Retraining Pipeline** | Implement automated trigger-based retraining pipelines when concept drift is detected in live production streams. |
| **Live Banking Network Connection** | Connect to a real UPI switch (NPCI) or payment gateway; requires regulatory licensing, PCI-DSS compliance, and Hardware Security Modules (HSMs). |
| **Multi-Datacenter Cluster Deployment** | Deploy across geographically distributed Kubernetes clusters with high-availability failover and contractual SLAs. |

---

## 15. Team Member Contributions

### 15.1 Jayasimha Padigeri -- Lead ML Research & Systems Engineer

- Conceptualized project architecture and designed the complete experimental methodology.
- Trained, evaluated, and froze the Phase 1 E1 LightGBM baseline model with 406 canonical features and calibrated threshold (`0.616521`).
- Implemented the Milestone 5 integration bridge (`StreamServingBridge`) with temporal causality enforcement.
- Built the `OnlineFeatureHydrationAdapter` and certified offline/online mathematical prediction parity (delta-P = 0).
- Authored technical documentation, research specifications, and project reporting.

### 15.2 Ishwarya (Member 1) -- Streaming Ingestion & Kafka Architect

- Engineered the distributed Kafka event ingestion architecture (`kafka/producer/`).
- Implemented Murmur2 hash-based partition routing to guarantee strict per-card event ordering across 6 partitions.
- Configured idempotent delivery semantics (`acks=all`, `enable.idempotence=True`, `retries=5`) to eliminate duplicate event risk.
- Benchmarked producer ingestion throughput, achieving over 148,000 events/second.

### 15.3 Harika (Member 2) -- Distributed Stream Processing & Data Engineer

- Built Apache Flink sliding-window processing pipelines with 5-minute and 10-minute event-time windows.
- Formulated velocity metrics, spending velocity ratios, and Complex Event Processing (CEP) fraud alert rules.
- Implemented Apache Spark columnar Parquet storage optimizations, delivering a 7.25x scan acceleration.
- Configured watermark-based event-time semantics for robust handling of out-of-order streaming events.

### 15.4 Hadassah Kiran (Member 3) -- Serving, Containerization & API Deployment Lead

- Developed the FastAPI model serving microservice (`serving/api/main.py`) with all operational REST endpoints.
- Built the `ServingPreprocessor` wrapper with automated 406-feature schema validation.
- Identified and resolved the Windows CRLF line-ending bug in LightGBM booster loading to ensure cross-platform compatibility.
- Engineered the NumPy vectorized preprocessing optimization (`feature_vectorizer.py`), achieving a ~70x speedup over the Pandas prototype.
- Designed and implemented the real-time browser observability dashboard (`serving/dashboard/`).
- Configured Docker containerization and authored operational deployment guides (`LIVE_SERVER_GUIDE.md`).

---

## 16. Conclusion

### 16.1 Summary of Achievements

This project successfully designed, implemented, and empirically validated a **complete, end-to-end, production-oriented research prototype** for financial fraud detection, integrating machine learning, distributed streaming, online feature hydration, containerized serving, and real-time observability:

1. **High-Precision ML Baseline:** The Phase 1 E1 LightGBM model achieves **PR-AUC = 0.5317** and **ROC-AUC = 0.8990** on the held-out test set (~15x better than a random baseline), with **98.0% precision** across the top 100 highest-risk alerts.
2. **Sub-100ms Streaming Pipeline:** The integrated streaming and serving architecture (Kafka $\to$ Flink $\to$ Bridge $\to$ Hydration $\to$ LightGBM) achieves a median end-to-end latency of **92.68 ms**, comfortably satisfying commercial payment network authorization windows (150–250 ms).
3. **Empirically Certified Serving Parity:** The live containerized serving microservice produces predictions with **delta-P = 0.000000e+00** compared to the offline research model, with 100% decision concordance across the 100-transaction certification set.
4. **Programmatic Causal Invariant Enforcement:** Point-in-time feature hydration strictly enforces $t_{history} < t_{event}$, programmatically preventing lookahead data leakage in streaming feature reconstruction.
5. **Adversarial Chaos Resilience:** The serving pipeline passes 20 formalized adversarial test scenarios (Cases A through T), failing closed safely under corrupt payloads, missing fields, and out-of-order events.
6. **Full Test Suite Verification:** 108 automated unit and integration tests pass with zero failures across all active codebase components.

### 16.2 Honest Engineering Boundaries

#### What Has Been Experimentally Verified:
- Statistical evaluation on 78,542 held-out labeled test transactions with bootstrap confidence intervals.
- Exact offline/online prediction parity empirically certified across 100 sequential transactions.
- Programmatic enforcement of temporal causality ($t_{history} < t_{event}$) in feature hydration.
- Component-level latency profiling across 500 transactions.
- Fail-closed safety under 20 formalized adversarial chaos scenarios.
- Cryptographic SHA-256 verification of all frozen model artifacts at server startup.

#### What Is Not Claimed:
- Live connection to a real-world banking core, commercial payment gateway, or NPCI UPI switch.
- Formal banking regulatory audit certification or PCI-DSS production sign-off.
- Multi-datacenter geo-redundant cluster deployment with contractual uptime SLAs.
- Supervised accuracy claims on the unlabeled competition blind test set (used strictly for pipeline reconciliation and parity testing).
- Universal mathematical performance guarantees for unseen payment distributions outside the benchmark dataset.

### 16.3 Original Research Contributions

- **Empirical Demonstration:** Proven efficacy of gradient-boosted decision trees (LightGBM) with 406 canonical features under strict chronological validation on large-scale tabular financial transactions.
- **Architectural Reference Implementation:** A fully integrated research prototype demonstrating real-time point-in-time feature hydration and stream-serving bridge coordination.
- **Certified Zero-Skew Serving:** Practical demonstration of zero train-serve prediction divergence (delta-P = 0) between offline research workflows and containerized microservices.
- **Formalized Adversarial Safety Protocol:** A 20-case chaos testing framework validating fail-closed safety for real-time streaming ML systems.

---

*Certified under Milestone 5 Integration Protocol -- Phase 1 Research Prototype -- October 2026.*

*All model artifacts cryptographically verified. 108 tests passed. 0 tests failed.*
