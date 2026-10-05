# Stage-A Audit Report: Member 3 (ML Serving)

**Target Repository:** [`Hadassah627/fraud-model-serving`](https://github.com/Hadassah627/fraud-model-serving)  
**Host Repository:** `adaptive-upi-fraud-detection`  
**Target Branch for Preparation:** `integration/member3-serving` (dedicated, isolated branch)  
**Protected Branch:** `Upto_Phase-4` (locked research baseline — strictly untouched)  
**Date:** October 2026  
**Auditor:** Senior ML Research, MLOps & Systems Integration Engineer  
**Stage-A Verdict:** 🟡 **MODIFY — DO NOT MERGE YET**

---

## 1. Executive Summary & Verdict

A comprehensive Stage-A architectural, forensic, cryptographic, and security audit was executed on Member 3 (`fraud-model-serving`, commit `280eeaa`). 

### Core Finding:
Member 3 is **substantially better aligned with our E1 serving requirements** than previously anticipated. The core model execution pipeline is genuine: it loads the authentic frozen E1 LightGBM booster (`model.txt`), the fitted preprocessor pipeline (`preprocessing.joblib`), and enforces the exact 406-feature schema (`feature_names.json`) with zero target leakage.

However, **it must NOT be merged into `Upto_Phase-4` at this stage**. The primary blocker is specific and architectural:
> **The Feature Hydration Gap:** Member 3 executes valid inference when presented with complete raw IEEE-CIS transaction records. However, it lacks an **Online Feature Hydration Adapter** capable of bridging the compact 6–10 field streaming payloads produced by Member 1 (Kafka) and Member 2 (Spark/Flink) into the canonical 406-feature tabular representation required by E1. 

In addition, hygiene defects (tracked `__pycache__` bytecode files) and machine-specific local developer paths (`C:\Users\HADASSAH KIRAN\...`) require systematic remediation on the dedicated `integration/member3-serving` branch before any merge into `Upto_Phase-4` can be considered.

---

## 2. Complete Repository Inventory

The Member 3 repository (`Hadassah627/fraud-model-serving`) contains the following structural layout:

```text
fraud-model-serving/
├── .dockerignore
├── Dockerfile
├── README.md
├── requirements.txt
├── project_walkthrough_full.md
├── run_single_transaction_demo.py
│
├── api/
│   ├── __init__.py
│   ├── main.py                        # FastAPI serving application (lifespan manager, /predict, /health)
│   ├── schemas.py                     # Pydantic schemas (TransactionRequest, PredictionResponse)
│   └── __pycache__/                   # ⚠️ Tracked bytecode (.pyc) files [VIOLATION]
│
├── models/
│   └── E1/
│       ├── model.txt                  # E1 LightGBM Booster (1,000 trees)
│       ├── model_lf.txt               # E1 LightGBM Booster (LF newlines)
│       ├── preprocessing.joblib       # Fitted IEEECISPreprocessor pipeline
│       ├── feature_names.json         # Approved 406 feature schema
│       └── test_clean_model.txt       # Test artifact
│
├── preprocessing/
│   ├── __init__.py
│   ├── serving_wrapper.py             # ServingPreprocessor: validates 406 features, zero target leakage
│   └── __pycache__/                   # ⚠️ Tracked bytecode (.pyc) files [VIOLATION]
│
├── src/
│   ├── __init__.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── ieee_cis_features.py       # IEEECISPreprocessor class definition (stripped comments/docstrings)
│   │   └── __pycache__/               # ⚠️ Tracked bytecode (.pyc) files [VIOLATION]
│   └── __pycache__/                   # ⚠️ Tracked bytecode (.pyc) files [VIOLATION]
│
├── inference/
│   ├── __init__.py
│   ├── offline_inference.py           # OfflineInferenceEngine (contains hardcoded HADASSAH KIRAN paths)
│   ├── phase2_benchmarking.py         # Offline latency & throughput benchmarking
│   ├── phase4_consistency.py          # 1,000-transaction offline-vs-online consistency harness
│   ├── phase5_api_benchmarking.py     # Live FastAPI concurrency & RPS benchmarking
│   ├── phase8_monitoring_experiment.py# Logging & latency aggregation experiment
│   └── __pycache__/                   # ⚠️ Tracked bytecode (.pyc) files [VIOLATION]
│
├── monitoring/
│   ├── __init__.py
│   ├── api_monitor.py                 # In-memory MonitoringState request counter & latency accumulator
│   ├── inference_logger.py            # Structured JSON log formatter (zero payload leakage)
│   ├── model_metadata.py              # Single source of truth MODEL_METADATA dictionary
│   └── __pycache__/                   # ⚠️ Tracked bytecode (.pyc) files [VIOLATION]
│
├── benchmarks/
│   ├── offline_inference_report.md
│   ├── phase2_benchmark_report.md
│   ├── phase4_consistency_report.md
│   ├── phase5_api_performance_report.md
│   ├── phase6_error_handling_report.md
│   ├── phase7_docker_consistency_report.md
│   ├── phase7_docker_deployment_report.md
│   ├── phase8_monitoring_versioning_report.md
│   └── results/                       # CSV benchmarks for latency, errors, and Docker validation
│
└── tests/
    ├── __init__.py
    ├── test_offline_inference.py
    ├── test_phase2_benchmarks.py
    ├── test_api.py
    ├── test_phase4_consistency.py
    ├── test_phase5_api_benchmarks.py
    ├── test_phase6_error_handling.py
    ├── test_phase7_docker.py
    ├── test_phase8_monitoring.py
    └── __pycache__/                   # ⚠️ Tracked bytecode (.pyc) files [VIOLATION]
```

---

## 3. Cryptographic & Forensic Artifact Integrity

### 3.1 Git Blob Identity Verification (100% Exact Match)
Git blob hashes directly compute the SHA-1 of the exact uncompressed object data stored in the repository object database. Comparing the Git tree blobs between canonical research and Member 3:

| Artifact | Canonical Path (`adaptive-upi`) | Member 3 Path (`fraud-model-serving`) | Canonical Git Blob | Member 3 Git Blob | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **E1 LightGBM Model** | `experiments/E1_lightgbm/model.txt` | `models/E1/model.txt` | `27c44a2a43d618368c5cbb94b7a1f0c763599b23` | `27c44a2a43d618368c5cbb94b7a1f0c763599b23` | 🔒 **EXACT MATCH** |
| **E1 Preprocessor** | `experiments/E1_lightgbm/preprocessing.joblib` | `models/E1/preprocessing.joblib` | `7468a67fcbfb85dbbfab44a73577928efbf9c604` | `7468a67fcbfb85dbbfab44a73577928efbf9c604` | 🔒 **EXACT MATCH** |
| **Feature Schema** | `experiments/E1_lightgbm/feature_names.json` | `models/E1/feature_names.json` | `b73ceca5b99276b8d3d33230f44426cb709a3800` | `b73ceca5b99276b8d3d33230f44426cb709a3800` | 🔒 **EXACT MATCH** |

**Conclusion:** Member 3 did not retrain, replace, or alter the E1 champion model or preprocessor.

---

### 3.2 Forensic Resolution of the SHA-256 Documentation Discrepancy

In Member 3's `README.md`, the documented SHA-256 for `models/E1/model.txt` is:
```text
d04dff4f765801b196a5eda7d24288e8fd792f5fc06a0576d0de2249f98cc219
```
whereas the canonical manifest on the Windows research host reports:
```text
ac93b59a7eee7a23b1d77a7fa03d348153328da128a1ba66d6f34cf490ec6d96
```

**Forensic Investigation Finding:**
- When `experiments/E1_lightgbm/model.txt` is checked out on Windows with `core.autocrlf = true`, line endings are converted to CRLF (`\r\n`), which computes to `ac93b59a...`.
- In standard Linux environments (such as the Docker container `python:3.11-slim` where Member 3 executed Phase 7 benchmarking), line endings remain native LF (`\n`).
- Hashing the raw Git blob directly (`git cat-file -p 27c44a2a43d618368c5cbb94b7a1f0c763599b23`) computes **identically** to:
  $$\text{SHA256}(\text{raw blob with LF}) = \texttt{d04dff4f765801b196a5eda7d24288e8fd792f5fc06a0576d0de2249f98cc219}$$
- The binary artifacts (`preprocessing.joblib` and `feature_names.json`) are newline-invariant and match across both environments:
  - `preprocessing.joblib`: `0c336989206214cab202d3b4a8a726206cb4ca69a0908e52fdb6f9cf479fbf69`
  - `feature_names.json`: `1c59105a626f57533af4fc56f3ba10ae112b739c16c2e2d99cec24c1b1d0330d`

**Verdict:** There is zero model divergence. The documentation in Member 3 reflects the Linux/Docker environment SHA-256. Both repositories hold the exact identical model weights and booster tree structures.

---

## 4. E1 Model & Preprocessing Serving Architecture

### 4.1 FastAPI Serving Execution Path
The live inference execution flow implemented in `api/main.py` is:

```text
HTTP POST /predict (JSON Body)
        │
        ▼
Pydantic Validation (TransactionRequest)
        │
        ▼
DataFrame Construction (pd.DataFrame([raw_dict]))
        │
        ▼
OfflineInferenceEngine.predict_transaction()
        │
        ▼
ServingPreprocessor.transform()
        ├── Inject dummy isFraud = 0 (serving edge case)
        ├── Reindex missing raw columns with np.nan
        ├── Execute IEEECISPreprocessor.transform()
        │     ├── Generate cyclic temporal features (hour_sin, hour_cos, day_index)
        │     ├── Apply median imputers (_num_medians)
        │     ├── Apply categorical encoders (_cat_encoders)
        │     └── Generate missingness indicators (*_missing)
        ├── Strictly enforce 406 column ordering from feature_names.json
        └── Assert isFraud NOT in X, TransactionID NOT in X
        │
        ▼
LightGBM Booster Model (model.predict(X))
        │
        ▼
Fraud Probability (float in [0.0, 1.0])
        │
        ▼
Decision Rule: prob >= 0.616521 ? "FRAUD" : "LEGIT"
        │
        ▼
HTTP 200 JSON Response (PredictionResponse)
```

### 4.2 Target Leakage & ID Isolation (Audit: ✅ PASS)
A critical inspection of `preprocessing/serving_wrapper.py` and `tests/test_offline_inference.py` confirms:
1. `isFraud` is explicitly stripped before model prediction. If present in raw input, it is extracted solely for offline validation metadata comparison (`actual_label`).
2. An explicit runtime guard asserts:
   ```python
   if "isFraud" in X.columns:
       raise ValueError("Leakage detection: 'isFraud' target column detected in model feature matrix!")
   ```
3. `TransactionID` is similarly stripped and preserved solely as request metadata.
4. Searches for `fraud_bool` across the repository yielded zero matches.
5. **Verdict: Target and identifier isolation standards are fully satisfied.**

---

## 5. Architectural Blocker: The Feature Hydration Gap

### 5.1 The Root Problem: Streaming vs Tabular Representations
The fundamental incompatibility between Member 1/2 streaming infrastructure and Member 3 serving lies in the feature representations:

```text
MEMBER 1 / MEMBER 2 STREAMING EVENT (Compact)
{
  "transaction_id": "TX-2987000",
  "card_id": "CARD-13926",
  "amount": 125.50,
  "event_time": 1791212210000,
  "merchant_id": "M-W",
  "device_type": "mobile",
  "country": "US"
}
                          VS
MEMBER 3 EXPECTED RAW INPUT (Raw IEEE-CIS Record)
- 394 raw columns: TransactionAmt, ProductCD, card1..card6, addr1..addr2,
  dist1, P_emaildomain, R_emaildomain, C1..C14, D1..D15, M1..M9, V1..V339,
  id_01..id_38, DeviceType, DeviceInfo
                          ▼
E1 PREPROCESSOR (preprocessing.joblib)
- 406 engineered tabular columns (temporal sine/cosine, medians, categories)
```

### 5.2 Feature Completion vs True Feature Hydration
In `preprocessing/serving_wrapper.py` (lines 118–124), Member 3 handles missing raw columns via:
```python
all_expected_raw = set(self.preprocessor._num_medians.keys()) | set(self.preprocessor._cat_encoders.keys()) | {"TransactionDT"}
missing_raw = [c for c in all_expected_raw if c not in df.columns]
if missing_raw:
    df = df.reindex(columns=list(df.columns) + missing_raw, fill_value=np.nan)
```
When presented with a 6-field streaming event:
1. ~388 raw fields are populated with `np.nan`.
2. The preprocessor fills these `np.nan` values with population medians or default "MISSING" categorical levels.
3. The LightGBM booster computes a prediction based on ~95% synthetic default values.

**Scientific Assessment:** This is **feature completion / fallback imputation**, not **feature hydration**. It allows the code to run without crashing, but it destroys model fidelity because the cardholder’s true historical behavior ($C$-counts, $D$-deltas, $V$-anonymized patterns) is lost.

### 5.3 The Required Solution: Online Feature Hydration Adapter
Before Member 3 can be connected to Member 1/2, an **Online Feature Hydration Adapter** must be deployed between Flink/Spark and the Serving API:

```text
┌───────────────────────┐      ┌────────────────────────┐
│ Member 1 Kafka Stream │      │ Member 2 Flink State   │
│ (Raw compact event)   │      │ (5m/10m velocity $V_R$)│
└───────────┬───────────┘      └───────────┬────────────┘
            │                              │
            └──────────────┬───────────────┘
                           │
                           ▼
         ┌───────────────────────────────────┐
         │ ONLINE FEATURE HYDRATION ADAPTER  │
         │                                   │
         │ 1. Lookup cardholder profile:     │
         │    - card1..card6, addr1..addr2   │
         │    - P_emaildomain, device attrs  │
         │ 2. Hydrate behavioral state:      │
         │    - Cumulative counts (C1..C14)  │
         │    - Time deltas (D1..D15)        │
         │    - Behavioral V-aggregates      │
         │ 3. Enforce Point-in-Time rules    │
         │    (Zero lookahead leakage)       │
         └─────────────────┬─────────────────┘
                           │ Hydrated Raw Transaction
                           ▼
         ┌───────────────────────────────────┐
         │ MEMBER 3 FASTAPI SERVING ENGINE   │
         │                                   │
         │ 1. ServingPreprocessor (406 feats)│
         │ 2. Frozen LightGBM Booster        │
         │ 3. Threshold 0.616521             │
         └─────────────────┬─────────────────┘
                           │
                           ▼
                  VERDICT: FRAUD / LEGIT
```

---

## 6. Hygiene, Security & Portability Findings

### 6.1 Tracked Bytecode Violation (🟡 MODIFY)
Member 3 tracks 22 compiled bytecode (`.cpython-311.pyc` and `.cpython-314.pyc`) files across `api/`, `inference/`, `preprocessing/`, `src/`, `monitoring/`, and `tests/`.
- **Remediation:** Remove all tracked `.pyc` and `__pycache__` directories using `git rm -r --cached` and enforce via `.gitignore`.

### 6.2 Hardcoded Machine Paths (🟡 MODIFY)
In `inference/offline_inference.py` (lines 47–48):
```python
DEFAULT_DATASET_TX_PATH = r"C:\Users\HADASSAH KIRAN\Downloads\UPI Fraud Detection\IEEE CIS-20260902T145234Z-1-001\IEEE CIS\train_transaction.csv"
DEFAULT_DATASET_ID_PATH = r"C:\Users\HADASSAH KIRAN\Downloads\UPI Fraud Detection\IEEE CIS-20260902T145234Z-1-001\IEEE CIS\train_identity.csv"
```
- **Remediation:** Replace with relative repository paths, environment variables (`IEEE_CIS_DATASET_DIR`), or optional CLI arguments.

### 6.3 Duplicate Model Storage (🟡 MODIFY)
Member 3 duplicates `model.txt` and `preprocessing.joblib` inside `models/E1/` (as well as redundant copies `model_lf.txt` and `test_clean_model.txt`).
- **Remediation:** In our integrated repository, `serving/` must reference or symlink the single authoritative source of truth under `experiments/E1_lightgbm/`, eliminating file duplication.

### 6.4 Dependency Isolation (🟢 PASS)
Member 3 maintains its own `requirements.txt` (`fastapi`, `uvicorn`, `lightgbm`, `pydantic`, `httpx`).
- **Remediation:** Isolate into `serving/requirements.txt`. The root research `requirements.txt` remains strictly untouched.

---

## 7. Stage-A Evaluation Scorecard

| # | Evaluation Dimension | Target Requirement | Member 3 Status | Audit Verdict |
| :---: | :--- | :--- | :--- | :---: |
| 1 | **Repository & Main Branch** | Clean Git repository structure | Exists (`Hadassah627/fraud-model-serving`) | 🟢 **PASS** |
| 2 | **E1 Model Weights Identity** | Match Git blob `27c44a...` | Exact blob match (`27c44a2a43d6...`) | 🟢 **PASS** |
| 3 | **E1 Preprocessor Identity** | Match Git blob `7468a6...` | Exact blob match (`7468a67fcbfb...`) | 🟢 **PASS** |
| 4 | **E1 Feature Schema Identity** | Match Git blob `b73ceca...` | Exact blob match (`b73ceca5b992...`) | 🟢 **PASS** |
| 5 | **Feature Dimensionality** | Exactly 406 features validated | Preprocessor & LightGBM enforce 406 | 🟢 **PASS** |
| 6 | **Feature Ordering Parity** | Strict adherence to `feature_names.json` | Explicit list alignment enforced | 🟢 **PASS** |
| 7 | **Authoritative Threshold** | Exact E1 threshold = `0.616521` | Configured and enforced in engine | 🟢 **PASS** |
| 8 | **Zero Target Leakage** | `isFraud` excluded from model matrix $X$ | Asserted absent in code & tests | 🟢 **PASS** |
| 9 | **Zero Identifier Leakage** | `TransactionID` excluded from $X$ | Excluded; preserved only in metadata | 🟢 **PASS** |
| 10 | **FastAPI Implementation** | Async `/predict`, `/batch`, `/health` | Clean, robust lifespan implementation | 🟢 **PASS** |
| 11 | **Docker Packaging** | `python:3.11-slim`, port 8000, libgomp1 | Verified working container build | 🟢 **PASS** |
| 12 | **Offline/Online Parity** | 100% decision match on raw IEEE-CIS | 1,000 txs tested: 0.00000000 diff | 🟢 **PASS** (Raw IEEE-CIS) |
| 13 | **Logging & Monitoring** | Structured logs, zero payload leak | Phase 8 in-memory monitor & logger | 🟢 **PASS** |
| 14 | **Bytecode Hygiene** | Zero tracked `__pycache__` | 22 `.pyc` files tracked in repository | 🟡 **MODIFY** |
| 15 | **Path Portability** | Zero machine-specific user paths | `C:\Users\HADASSAH KIRAN` paths present | 🟡 **MODIFY** |
| 16 | **Model SHA-256 Docs** | Accurate multi-platform hashes | Documented Linux hash, needs CRLF note | 🟡 **MODIFY** |
| 17 | **M1/M2 Streaming Hydration**| Hydrate compact Kafka events to 406 feats| **Not implemented** (falls back to NaN) | 🔴 **BLOCKED** |

---

## 8. Final Stage-A Decision & Stage-B Roadmap

### Verdict: 🟡 MODIFY — DO NOT MERGE YET

The core serving components of Member 3 are functionally sound, mathematically consistent with E1, and free of target leakage. However, integration must follow a strictly controlled process on branch `integration/member3-serving`:

### 12-Step Remediation & Integration Protocol:
1. **Branch Isolation:** Maintain all work on `integration/member3-serving`. Keep `Upto_Phase-4` locked.
2. **Directory Placement:** Unpack Member 3 into top-level `serving/`.
3. **Bytecode Elimination:** Purge all tracked `__pycache__` and `.pyc` files.
4. **Path Sanitization:** Parameterize `C:\Users\HADASSAH KIRAN` paths in `offline_inference.py`.
5. **Deduplicate Artifacts:** Point `serving/` to canonical `experiments/E1_lightgbm/` artifacts.
6. **Documentation Parity:** Document both LF (`d04dff4f...`) and CRLF (`ac93b59a...`) hashes for `model.txt`.
7. **Dependency Isolation:** Place dependencies in `serving/requirements.txt` without touching root requirements.
8. **Feature Hydration Specification:** Author [`docs/architecture/MEMBER3_FEATURE_HYDRATION_DESIGN.md`](file:///c:/Users/Harini/Documents/GitHub/Jayasimha-github/adaptive-upi-fraud-detection/docs/architecture/MEMBER3_FEATURE_HYDRATION_DESIGN.md).
9. **Controlled Merge Map:** Author [`docs/integration/MEMBER3_MERGE_MAP.md`](file:///c:/Users/Harini/Documents/GitHub/Jayasimha-github/adaptive-upi-fraud-detection/docs/integration/MEMBER3_MERGE_MAP.md).
10. **Hydration Adapter Implementation:** Implement the prototype `FeatureHydrationAdapter` mapping streaming events to hydrated E1 vectors.
11. **Comprehensive Regression Suite:** Execute all 173 repository tests plus serving test suites.
12. **Review & Approval:** Solicit user approval via GitHub PR before merging into `Upto_Phase-4`.
