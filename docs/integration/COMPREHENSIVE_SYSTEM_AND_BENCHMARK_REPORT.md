# Adaptive UPI Fraud Detection: Comprehensive System, Research & Benchmark Report

**Project Title:** Adaptive Financial & UPI Fraud Detection System  
**Repository:** `Jayasimha-2005/adaptive-upi-fraud-detection`  
**Current Git Branch:** `spark-benchmark-results`  
**Scope Covered:** Architecture, Data Pipelines (Spark & Flink), ML Experiments (E1–E3, Phase 4), Streaming Benchmarks, and Storage Optimizations  
**Date:** October 2026  

---

## 1. Executive Summary

This report aggregates and synthesizes findings from all research documents, technical specifications, and benchmark audits in the repository. The project delivers a production-grade, adaptive fraud detection platform combining:
1. **Machine Learning Core:** A high-performing LightGBM baseline (**E1**, PR-AUC: 0.5267) alongside temporal sequence research (GRU, Hybrids) and concept drift adaptation (Phase 4 PSI-triggered retraining).
2. **Real-Time Stream Processing (Apache Flink 2.2):** Event-time windowing, keyed state, and velocity feature engineering capable of sustaining **499.96 records/s** with sub-second scoring latency.
3. **High-Throughput Batch Processing (Apache Spark 3.5.9):** Columnar Parquet conversion achieving a **7.25x speedup** (86.20% processing time reduction) over raw CSV across 590,540 IEEE-CIS transactions.
4. **Reliable Messaging & Ingestion (Apache Kafka):** Unified pub/sub message backbone bridging synthetic transaction generators, stream processors, and model serving layers.

---

## 2. End-to-End System Architecture

```
                                +-----------------------------+
                                |  Live / Replayed UPI Stream |
                                |   (IEEE-CIS & Synthetic)    |
                                +--------------+--------------+
                                               |
                                               v
                                 [Kafka: fraud-transactions]
                                               |
                     +-------------------------+-------------------------+
                     |                                                   |
                     v                                                   v
          +-----------------------+                           +---------------------+
          |   Apache Flink 2.2    |                           |  Apache Spark 3.5.9 |
          | (Real-Time Streaming) |                           | (Batch Optimization)|
          +-----------+-----------+                           +----------+----------+
                      |                                                  |
              * Bounded Watermarks                               * 590K Records Agg.
              * Keyed by user_id                                 * Parquet Conversion
              * 5m & 10m Windows                                 * 927K records/s (7.25x)
              * Velocity Ratios                                          |
                      |                                                  v
                      v                                         [Parquet Data Lake]
           [Kafka: fraud-features]                                       |
                      |                                                  v
                      v                                         [Drift Audits & PSI]
         +-------------------------+                                     |
         |  Serving Layer Engine   | <-----------------------------------+
         | (E1 LightGBM Classifier)|
         +-------------------------+
```

---

## 3. Machine Learning Research & Model Evaluation

### 3.1 Model Performance Comparison (IEEE-CIS & BAF)
From [README.md](README.md) and [`reports/RESEARCH_STATUS_COMPLETE_REVIEW.md`](reports/RESEARCH_STATUS_COMPLETE_REVIEW.md):

| Experiment | Model Architecture | Dataset | PR-AUC | ROC-AUC | Status & Recommendation |
|---|---|---|---:|---:|---|
| **E1** | **LightGBM (Tabular Baseline)** | IEEE-CIS (590K) | **0.5267** | **0.9324** | ✅ **Production Model for Serving** |
| **E2** | GRU (Temporal Recurrent) | IEEE-CIS (590K) | 0.1683 | 0.7912 | ❌ Suboptimal (sparse temporal chains) |
| **E3A** | Hybrid (LightGBM + GRU Emb) | IEEE-CIS (590K) | 0.4271 | 0.9015 | ❌ Lower accuracy than E1 standalone |
| **E3B** | Hybrid (Late Fusion Ensemble) | IEEE-CIS (590K) | 0.1654 | 0.7890 | ❌ Dominated by GRU noise |
| **E3C** | Hybrid (Attention Feature Fusion) | IEEE-CIS (590K) | 0.2974 | 0.8450 | ❌ Fails to beat E1 baseline |
| **Phase 4** | Adaptive Retraining (PSI-Triggered)| BAF (1M) | 0.2010* | 0.8841 | 🔬 Proves drift adaptation feasibility |

> **Key Rule:** **E1 LightGBM** is the sole model artifact deployed for live inference. Phase 4 models are trained on BAF (Bank Account Fraud) data and are not compatible with IEEE-CIS schemas.

---

## 4. Apache Flink: Real-Time Streaming Pipeline

### 4.1 Engineering Highlights
* **Version:** Apache Flink 2.2 / PyFlink on Python 3.11 (`upi-fraud-flink-22` environment).
* **Kafka Connectors:** `flink-connector-kafka-5.0.0-2.2.jar` & `kafka-clients-4.2.0.jar`.
* **Stateful Stream Processing:**
  * **Event Time & Watermarks:** Bounded out-of-orderness watermarking handles late/delayed UPI network packets.
  * **Keyed Partitioning:** Streams partitioned by `user_id` to manage isolated state accumulators.
  * **Multi-Window Feature Engineering:**
    * 5-minute sliding window: short-term burst count, sum, average amount.
    * 10-minute sliding window: baseline spend metrics.
    * Velocity Ratios: $\frac{\text{Count}_{5m}}{\text{Count}_{10m}}$ and $\frac{\text{Sum}_{5m}}{\text{Sum}_{10m}}$ to spot automated attack spikes.
  * **Sink:** Serialized feature vectors dispatched to `fraud-features` topic.

### 4.2 Streaming Ingestion & Throughput Benchmark Results

| Test Scenario | Total Records | Target Rate | Duration | Achieved Throughput | Success Rate |
|---|---:|---:|---:|---:|---:|
| **Burst Ingestion Test** | 1,000 | 500 rec/s | 2.006 s | **498.40 rec/s** | 100% |
| **Sustained Load Test** | 5,000 | 500 rec/s | 10.001 s | **499.96 rec/s** | 100% |

---

## 5. Apache Spark: Batch & Storage Optimization

### 5.1 Engineering Highlights
* **Version:** Apache Spark 3.5.9 / PySpark.
* **Storage Optimization:** Converted raw CSV transactions (590,540 rows, 400+ attributes) to columnar Snappy-compressed Parquet.
* **Query Execution:** Applied column pruning and predicate pushdown to eliminate unnecessary I/O overhead.

### 5.2 Batch Benchmark Results: CSV vs. Parquet (590,540 Records)

| Metric | Raw CSV File | Columnar Parquet File | Improvement / Delta |
|---|---:|---:|---|
| **Median Execution Time** | **4.6154 seconds** | **0.6370 seconds** | **86.20% time reduction** |
| **Median Throughput** | **127,949.66 rec/s** | **927,010.66 rec/s** | **7.25x speedup** |
| **Record Integrity** | 590,540 rows (20,663 fraud) | 590,540 rows (20,663 fraud) | Exact 1:1 match |
| **One-time Conversion Time** | — | 58.52 seconds | Offline preparation |

### 5.3 Spark Structured Streaming Baseline

| Metric | Measured Value |
|---|---:|
| **Records Processed** | 1,000 synthetic records |
| **Processing Duration** | 11.71 seconds |
| **Streaming Throughput** | 85.40 rec/s |
| **Micro-Batches** | 4 batches (250 records/batch) |

---

## 6. Comparative Evaluation: Flink vs. Spark

| Dimension | Apache Flink 2.2 (Streaming) | Apache Spark 3.5.9 (Batch & Structured) |
|---|---|---|
| **Processing Model** | Continuous, event-driven | Micro-batch (streaming) & RDD/DataFrame (batch) |
| **Latency Profile** | Sub-second (millisecond level) | Multi-second batch trigger overhead |
| **Throughput (Observed)** | 499.96 rec/s (at target rate) | 927,010.66 rec/s (Parquet in-memory batch) |
| **Window State Handling** | Native keyed memory state with sliding windows | Checkpointed StateStore across micro-batches |
| **Project Role** | Live UPI fraud feature calculation & scoring | Large-scale historical ETL, drift audits, and retraining |

---

## 7. Artifacts & Codebase Inventory

| Artifact / Path | Description |
|---|---|
| [`SPARK_AND_FLINK_IMPLEMENTATION_AND_RESULTS.md`](SPARK_AND_FLINK_IMPLEMENTATION_AND_RESULTS.md) | Dedicated Spark & Flink implementation and benchmark documentation |
| [`STREAM_AND_BATCH_PROCESSING_REPORT.md`](STREAM_AND_BATCH_PROCESSING_REPORT.md) | Distributed data engineering work report |
| [`spark/benchmarks/WSL_BENCHMARK_RESULTS.md`](spark/benchmarks/WSL_BENCHMARK_RESULTS.md) | Empirical Spark benchmark measurements |
| [`spark/reports/SPARK_RESEARCH_REPORT.md`](spark/reports/SPARK_RESEARCH_REPORT.md) | Comprehensive Spark research and feature engineering specification |
| [`flink/reports/FLINK_RESEARCH_REPORT.md`](flink/reports/FLINK_RESEARCH_REPORT.md) | Flink research objectives and streaming evaluation framework |
| [`reports/RESEARCH_STATUS_COMPLETE_REVIEW.md`](reports/RESEARCH_STATUS_COMPLETE_REVIEW.md) | Full multi-phase research review (E1–E3, Phase 4) |
| [`reports/DATASET_SUITABILITY_AUDIT.md`](reports/DATASET_SUITABILITY_AUDIT.md) | Forensic dataset suitability audit (IEEE-CIS, BAF, ULB) |
| [`reports/REVIEWER_QUESTIONS.md`](reports/REVIEWER_QUESTIONS.md) | Defense manual and reviewer FAQ covering methodology choices |
