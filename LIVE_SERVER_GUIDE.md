# Adaptive Financial Fraud Detection — Live Server & Observability Guide

> **Project:** Adaptive Financial Fraud Detection with Real-Time Streaming & Temporal Drift Defense  
> **Serving Engine:** Member 3 — High-Throughput FastAPI ML Serving Layer  
> **Model:** Frozen E1 LightGBM (`406` canonical features, decision threshold `0.616521`)

---

## 1. Quick Start Commands

### A. Start the Server Using Docker (Recommended)
To run the certified containerized serving application on port `8000`:
```powershell
docker rm -f adaptive-fraud-serving
docker run -d --name adaptive-fraud-serving -p 8000:8000 adaptive-fraud-serving:v1
```

To view live container logs:
```powershell
docker logs -f adaptive-fraud-serving
```

To stop the server:
```powershell
docker stop adaptive-fraud-serving
```

To restart an existing container:
```powershell
docker restart adaptive-fraud-serving
```

---

### B. Start the Server Locally (Python / Uvicorn)
If running outside of Docker:
```powershell
uvicorn serving.api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 2. Interactive Web Interfaces & Endpoints

Once the server is running, open any of the following URLs in your web browser:

| Interface / Endpoint | URL | Description |
| :--- | :--- | :--- |
| **Live Observability Dashboard** | [http://localhost:8000/dashboard](http://localhost:8000/dashboard) | Full real-time pipeline visualization, test transaction loader (5 to 1000+), live streaming stages, latency breakdown, and session transaction trace feed (up to 2,000 records). |
| **Interactive API Metrics** | [http://localhost:8000/metrics](http://localhost:8000/metrics) | Swagger/Docs-styled metrics explorer showing request volume, success rates, latency percentiles (**P50**, **P95**, **P99**), fraud decisions, and raw JSON export. |
| **Health Diagnostics** | [http://localhost:8000/health](http://localhost:8000/health) | Interactive health portal verifying model status, 406-feature contract, target isolation, and research invariants. |
| **Swagger API Documentation** | [http://localhost:8000/docs](http://localhost:8000/docs) | Interactive OpenAPI 3.1 schema testing page for `/predict`, `/predict/batch`, and `/predict/hydrated`. |

---

## 3. How to Use the Live Observability Dashboard

Navigate to [http://localhost:8000/dashboard](http://localhost:8000/dashboard) to monitor and trigger tests.

```
                    ┌─────────────────────────────────────────────────────────┐
                    │               LOAD TEST TRANSACTIONS UI                 │
                    │  [ 5 ]  [ 10 ]  [ 50 ]  [ 100 ]  [ 1000 ]  [ Random ]   │
                    └────────────────────────────┬────────────────────────────┘
                                                 │
                                                 ▼
Official IEEE-CIS Test Data ──► Kafka ──► Flink CEP ──► Bridge ──► Hydration Gate (406) ──► E1 LightGBM ──► Dashboard Table
```

### A. Running Automated Test Sessions
1. In the **Load Test Transactions** panel at the top:
   - Click **[ 5 ]**, **[ 10 ]**, **[ 50 ]**, **[ 100 ]**, or **[ 1000 ]** to stream unseen IEEE-CIS transactions sequentially.
   - Or enter a custom number in **Random Bulk** (e.g., `500`) and click **START**.
2. When triggered:
   - The dashboard **automatically creates a fresh test session** and resets previous counters.
   - You will see the **Progress Indicator** track completed transactions in real time.

### B. Monitoring the 5 Real-Time Pipeline Stages
* **Stage 1 (Kafka Ingestion):** Displays total produced/consumed messages, 6-partition load distribution, and ingestion latency.
* **Stage 2 (Flink CEP Velocity):** Displays stateful rolling 5-minute and 10-minute transaction counts/amounts and sliding velocity ratios.
* **Stage 3 (Stream Serving Bridge):** Confirms event normalization and strict target isolation (`isFraud` stripped).
* **Stage 4 (Online Feature Hydration):** Verifies the **Hard Hydration Gate** and constructs the exact **406 canonical feature vector** without future lookahead (`t < T`).
* **Stage 5 (E1 LightGBM Inference):** Executes sub-millisecond scoring against threshold `0.616521` and categorizes decisions into `FRAUD` or `LEGIT`.

### C. End-to-End Reconciliation Banner
* Directly verifies that zero transactions were dropped or duplicated across the pipeline:
  $$\text{Requested} = \text{Kafka Ingest} = \text{Flink CEP} = \text{Bridge} = \text{Hydrated} = \text{E1 Scored}$$
* Displays **`🟢 N / N — 100% RECONCILED (PASS)`** upon completion.

### D. Live Transaction Journey Feed
* Displays the detailed trace table with time, Transaction ID, Kafka partition, Flink CEP tag, Bridge status, 406 features, fraud probability, decision badge, and total roundtrip latency.
* Retains up to **2,000 transactions** per test session.

---

## 4. How to Send API Requests (cURL / Python)

### A. Single Prediction (`POST /predict`)
```powershell
curl -X POST "http://localhost:8000/predict" `
     -H "Content-Type: application/json" `
     -d '{
       "TransactionID": "4166283",
       "TransactionAmt": 57.95,
       "TransactionDT": 86400,
       "ProductCD": "W",
       "card1": 12577,
       "card4": "visa",
       "card6": "debit"
     }'
```

**Example JSON Response:**
```json
{
  "transaction_id": "4166283",
  "fraud_probability": 0.053899,
  "decision": "LEGIT",
  "actual_label": "UNKNOWN",
  "preprocessing_time_ms": 0.39,
  "model_prediction_time_ms": 0.22,
  "total_inference_time_ms": 0.78
}
```

---

### B. Batch Predictions (`POST /predict/batch`)
```powershell
curl -X POST "http://localhost:8000/predict/batch" `
     -H "Content-Type: application/json" `
     -d '{
       "transactions": [
         {"TransactionID": "TX-1", "TransactionAmt": 100.0, "TransactionDT": 86400, "ProductCD": "W", "card1": 1000},
         {"TransactionID": "TX-2", "TransactionAmt": 450.0, "TransactionDT": 86450, "ProductCD": "W", "card1": 2000}
       ]
     }'
```

---

### C. Online Feature Hydration Gate (`POST /predict/hydrated`)
Sends compact streaming events. If essential entity profile fields are missing, the **Hard Hydration Gate** returns HTTP `422` with a detailed diagnostic reason:
```powershell
curl -X POST "http://localhost:8000/predict/hydrated" `
     -H "Content-Type: application/json" `
     -d '{
       "transaction_id": "3663549",
       "card_id": "CARD-10409",
       "amount": 31.95,
       "event_time": 86400.0,
       "product_code": "W"
     }'
```

---

## 5. Automated Testing & Benchmarking Scripts

### A. Run Automated Client Test
Send real test transactions sequentially from the command line:
```powershell
python tools/auto_api_test.py --count 50 --new-session
```

### B. Reset Dashboard Session Remotely
```powershell
curl -X POST "http://localhost:8000/api/dashboard/session/reset"
```

---

## 6. Frozen Model Artifacts & Research Hashes

The serving application strictly enforces SHA-256 integrity checks on startup. These frozen files must never be modified:

| File Path | SHA-256 Hash |
| :--- | :--- |
| `serving/models/E1/model.txt` | `0308a5f2c43f888a85bad5ec6698fefcee27c2afe4a8d4bdc2b6314de69b6421` |
| `serving/models/E1/preprocessing.joblib` | `0c336989206214cab202d3b4a8a726206cb4ca69a0908e52fdb6f9cf479fbf69` |
| `serving/models/E1/feature_names.json` | `1c59105a626f57533af4fc56f3ba10ae112b739c16c2e2d99cec24c1b1d0330d` |

To verify hashes anytime:
```powershell
python -c "
import hashlib
for p in ['serving/models/E1/model.txt', 'serving/models/E1/preprocessing.joblib', 'serving/models/E1/feature_names.json']:
    with open(p, 'rb') as f:
        print(p, hashlib.sha256(f.read()).hexdigest())
"
```

---

## 7. Troubleshooting & Diagnostics

1. **Port `8000` is already in use:**
   ```powershell
   docker rm -f adaptive-fraud-serving
   ```
2. **Container exits or fails to start:**
   ```powershell
   docker logs adaptive-fraud-serving
   ```
3. **Verify API is reachable:**
   ```powershell
   curl http://localhost:8000/health
   ```
4. **Rebuild Docker Image after modifications:**
   ```powershell
   docker build -t adaptive-fraud-serving:v1 -f serving/Dockerfile serving
   docker rm -f adaptive-fraud-serving
   docker run -d --name adaptive-fraud-serving -p 8000:8000 adaptive-fraud-serving:v1
   ```

---

## 8. How to Stop the Server

### A. Stop Docker Container
To gracefully stop the running container:
```powershell
docker stop adaptive-fraud-serving
```


```
//NEVER REMOVE THE CONTAINER BECAUSE IF WE REMOVE THE CONTAINER THEN AGAIN WE HAVE TO BUILT THE COMPLETE CONTAIER SO NEVER REMOVE THE CONTAINER 


### B. Stop Local Server (Uvicorn / Python)
In the terminal window where Uvicorn is running:
- Press `Ctrl + C` to shut down the server process.
