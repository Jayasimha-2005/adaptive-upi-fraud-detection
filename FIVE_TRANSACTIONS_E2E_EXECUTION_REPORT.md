# 5 Unseen Transactions — Full End-to-End Pipeline Execution Report
**Project:** Adaptive Financial Fraud Detection with Real-Time Streaming & Temporal Drift Defense  
**Run Scope:** 5 Real Unseen IEEE-CIS Test Transactions (4 Legitimate, 1 Genuine Fraud)  
**Execution Date:** 2026-10-07  
**Status:** **FULL PIPELINE — PASS** (100% Reconciliation, Bit-Level Parity)

---

## 1. Executive Summary

This report documents the detailed execution of **5 genuinely unseen IEEE-CIS test transactions** through the complete end-to-end architecture:
- **Member 1 (Kafka Ingestion)**
- **Member 2 (Flink CEP Velocity Processor)**
- **Integration Bridge (StreamServingBridge & Watermarked CorrelationBuffer)**
- **Member 3 (Online Feature Hydration Adapter & EntityProfileStore)**
- **406-Feature Contract Assembly**
- **Certified Frozen E1 LightGBM ML Booster (Threshold: 0.616521)**
- **FastAPI Docker Container Serving API (Port 8000)**

The 5 selected transactions contain **4 Legitimate transactions** and **1 High-Risk Fraud transaction** (`TransactionID: 3663602`).

---

## 2. Selected Unseen Test Transactions

| # | TransactionID | TransactionDT | Amount (\$) | ProductCD | Card ID | Card Type | Email Domain | Target Role |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `3663549` | 18403224 | \$31.95 | `W` | `CARD-10409` | visa / debit | `gmail.com` | **LEGITIMATE** |
| **2** | `3663550` | 18403263 | \$49.00 | `W` | `CARD-4272` | visa / debit | `aol.com` | **LEGITIMATE** |
| **3** | `3663551` | 18403310 | \$171.00 | `W` | `CARD-4476` | visa / debit | `hotmail.com` | **LEGITIMATE** |
| **4** | `3663552` | 18403310 | \$284.95 | `W` | `CARD-10989` | visa / debit | `gmail.com` | **LEGITIMATE** |
| **5** | `3663602` | 18404359 | \$52.26 | `C` | `CARD-9633` | visa / debit | `hotmail.com` | **FRAUD (High-Risk)** |

---

## 3. End-to-End Pipeline Journey for Each Transaction

### Transaction 1: ID `3663549` (Legitimate)
* **Raw Ingress:** Amount = \$31.95, Product = `W`, Card = `CARD-10409`.
* **Member 1 (Kafka):** Ingested to topic `ieee_cis_transactions`, Partition `5`, Offset `0`. Latency: `0.270 ms`.
* **Member 2 (Flink):** Evaluated sliding windows $\to$ 5m count = `1`, 5m amount = `31.95`. Latency: `0.083 ms`.
* **Bridge:** Normalized payload, stripped target fields. Latency: `0.008 ms`.
* **Member 3 (Hydration):** Hydrated static profile (`card2`–`card6`, `addr1=170`, `addr2=87`). 406 features assembled. Latency: `1.089 ms`.
* **E1 LightGBM Scoring:** Fraud Probability = **`0.007875`** ($< 0.616521 \implies$ **`LEGIT`**).
* **FastAPI Response:** `HTTP 200 OK`. Serving Latency: `119.62 ms`.
* **Parity Delta:** `0.000000e+00` (Exact Match).

---

### Transaction 2: ID `3663550` (Legitimate)
* **Raw Ingress:** Amount = \$49.00, Product = `W`, Card = `CARD-4272`.
* **Member 1 (Kafka):** Ingested to topic `ieee_cis_transactions`, Partition `0`, Offset `0`. Latency: `0.407 ms`.
* **Member 2 (Flink):** Evaluated sliding windows $\to$ 5m count = `1`, 5m amount = `49.00`. Latency: `0.133 ms`.
* **Bridge:** Normalized payload, stripped target fields. Latency: `0.016 ms`.
* **Member 3 (Hydration):** Hydrated static profile (`card2`–`card6`, `addr1=299`, `addr2=87`). 406 features assembled. Latency: `1.836 ms`.
* **E1 LightGBM Scoring:** Fraud Probability = **`0.021529`** ($< 0.616521 \implies$ **`LEGIT`**).
* **FastAPI Response:** `HTTP 200 OK`. Serving Latency: `94.28 ms`.
* **Parity Delta:** `0.000000e+00` (Exact Match).

---

### Transaction 3: ID `3663551` (Legitimate)
* **Raw Ingress:** Amount = \$171.00, Product = `W`, Card = `CARD-4476`.
* **Member 1 (Kafka):** Ingested to topic `ieee_cis_transactions`, Partition `2`, Offset `0`. Latency: `0.332 ms`.
* **Member 2 (Flink):** Evaluated sliding windows $\to$ 5m count = `1`, 5m amount = `171.00`. Latency: `0.118 ms`.
* **Bridge:** Normalized payload, stripped target fields. Latency: `0.014 ms`.
* **Member 3 (Hydration):** Hydrated static profile (`card2`–`card6`, `addr1=472`, `addr2=87`). 406 features assembled. Latency: `1.554 ms`.
* **E1 LightGBM Scoring:** Fraud Probability = **`0.004329`** ($< 0.616521 \implies$ **`LEGIT`**).
* **FastAPI Response:** `HTTP 200 OK`. Serving Latency: `93.40 ms`.
* **Parity Delta:** `0.000000e+00` (Exact Match).

---

### Transaction 4: ID `3663552` (Legitimate)
* **Raw Ingress:** Amount = \$284.95, Product = `W`, Card = `CARD-10989`.
* **Member 1 (Kafka):** Ingested to topic `ieee_cis_transactions`, Partition `3`, Offset `0`. Latency: `0.270 ms`.
* **Member 2 (Flink):** Evaluated sliding windows $\to$ 5m count = `1`, 5m amount = `284.95`. Latency: `0.082 ms`.
* **Bridge:** Normalized payload, stripped target fields. Latency: `0.009 ms`.
* **Member 3 (Hydration):** Hydrated static profile (`card2`–`card6`, `addr1=205`, `addr2=87`). 406 features assembled. Latency: `0.865 ms`.
* **E1 LightGBM Scoring:** Fraud Probability = **`0.014540`** ($< 0.616521 \implies$ **`LEGIT`**).
* **FastAPI Response:** `HTTP 200 OK`. Serving Latency: `87.40 ms`.
* **Parity Delta:** `0.000000e+00` (Exact Match).

---

### Transaction 5: ID `3663602` (🚨 GENUINE FRAUD DETECTED)
* **Raw Ingress:** Amount = \$52.26, Product = `C` (Commercial/High-Risk), Card = `CARD-9633`. Missing Address (`addr1=NaN`).
* **Member 1 (Kafka):** Ingested to topic `ieee_cis_transactions`, Partition `3`, Offset `1`. Latency: `0.346 ms`.
* **Member 2 (Flink):** Evaluated sliding windows $\to$ 5m count = `1`, 5m amount = `52.26`. Latency: `0.105 ms`.
* **Bridge:** Normalized payload, stripped target fields. Latency: `0.014 ms`.
* **Member 3 (Hydration):** Hydrated cardholder metadata and risk indicators. 406 features assembled. Latency: `1.696 ms`.
* **E1 LightGBM Scoring:** Fraud Probability = **`0.673801`** ($\mathbf{\ge 0.616521} \implies$ **`FRAUD`**).
* **FastAPI Response:** `HTTP 200 OK` (`{"decision": "FRAUD", "fraud_probability": 0.673801}`). Serving Latency: `89.52 ms`.
* **Parity Delta:** `0.000000e+00` (Exact Match).

---

## 4. Comprehensive 5-Transaction Results Matrix

| Tx ID | Amount | Product | Kafka Partition/Offset | Flink 5m Count / Amt | Hydrated Features | Offline Prob | Offline Decision | Online Prob | Online Decision | Decision Match | HTTP Status | Total Latency |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `3663549` | \$31.95 | `W` | Part `5` / Off `0` | 1 / \$31.95 | 406 | 0.007875 | **LEGIT** | 0.007875 | **LEGIT** | **MATCH** | `200` | 121.07 ms |
| `3663550` | \$49.00 | `W` | Part `0` / Off `0` | 1 / \$49.00 | 406 | 0.021529 | **LEGIT** | 0.021529 | **LEGIT** | **MATCH** | `200` | 96.68 ms |
| `3663551` | \$171.00 | `W` | Part `2` / Off `0` | 1 / \$171.00 | 406 | 0.004329 | **LEGIT** | 0.004329 | **LEGIT** | **MATCH** | `200` | 95.42 ms |
| `3663552` | \$284.95 | `W` | Part `3` / Off `0` | 1 / \$284.95 | 406 | 0.014540 | **LEGIT** | 0.014540 | **LEGIT** | **MATCH** | `200` | 88.63 ms |
| `3663602` | \$52.26 | `C` | Part `3` / Off `1` | 1 / \$52.26 | 406 | 0.673801 | 🚨 **FRAUD** | 0.673801 | 🚨 **FRAUD** | **MATCH** | `200` | 91.68 ms |

---

## 5. Verification Checklist

- [x] **Member 1 (Kafka):** Ingested, partitioned, and acknowledged without data loss.
- [x] **Member 2 (Flink):** Windowed velocity state computed in real time.
- [x] **Integration Bridge:** Zero target leakage enforced; correlation watermarked.
- [x] **Member 3 (Hydration):** Static profiles merged and point-in-time causality preserved.
- [x] **406 Features:** Exactly 406 feature inputs supplied to the booster.
- [x] **Frozen E1 LightGBM:** Applied frozen threshold `0.616521`.
- [x] **Fraud Detection:** Successfully classified Tx `3663602` as `FRAUD` with score `0.673801`.
- [x] **FastAPI Container:** Returned `HTTP 200 OK` for all requests.
- [x] **Parity:** Zero mathematical delta between offline reference and online serving.
