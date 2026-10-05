# Controlled Merge Map & Integration Specification: Member 3 (ML Serving)

**Target Repository Source:** [`Hadassah627/fraud-model-serving`](https://github.com/Hadassah627/fraud-model-serving) (Commit `280eeaa`)  
**Host Repository Destination:** `adaptive-upi-fraud-detection`  
**Target Branch:** `integration/member3-serving` (strictly isolated from `Upto_Phase-4`)  
**Component Destination Directory:** `serving/`  
**Date:** October 2026  
**Auditor:** Senior ML Research, MLOps & Systems Integration Engineer  

---

## 1. Integration Principles & Boundary Safeguards

To maintain absolute research reproducibility and infrastructure isolation:
1. **Host Research Freeze:** The canonical research directories (`src/`, `experiments/`, `phase2_gru/`, `reports/`, `configs/`) and the root `requirements.txt` are **100% frozen and MUST NOT be modified**.
2. **Dedicated Serving Namespace:** All incoming Member 3 code is unpacked into a top-level `serving/` directory.
3. **No Duplicate Model Blobs:** Member 3's duplicated `models/E1/` directory will not be stored redundantly. Serving components will resolve the canonical artifacts directly from `experiments/E1_lightgbm/`.
4. **Bytecode Elimination:** All 22 tracked `__pycache__` and `.pyc` files present in the external repository are strictly excluded.
5. **Path Sanitization:** Hardcoded Windows paths referencing `C:\Users\HADASSAH KIRAN\` must be replaced with portable relative paths and environment variable fallbacks.
6. **Feature Hydration Integration:** The Online Feature Hydration Adapter (`serving/hydration/`) is introduced to bridge Member 1/2 streaming events into 406-feature E1 vectors.

---

## 2. Source-to-Target File Mapping Table

| External Source (`fraud-model-serving/`) | Integrated Target (`serving/`) | Action | Rationale / Transformation Notes |
| :--- | :--- | :---: | :--- |
| `api/__init__.py` | `serving/api/__init__.py` | **COPY** | Preserved. |
| `api/main.py` | `serving/api/main.py` | **MODIFY** | Update default model paths to `experiments/E1_lightgbm/`; integrate hydration endpoint. |
| `api/schemas.py` | `serving/api/schemas.py` | **MODIFY** | Add `HydratedTransactionRequest` schema alongside `TransactionRequest`. |
| `api/__pycache__/*` | *EXCLUDED* | **DROP** | Bytecode hygiene violation. |
| `preprocessing/__init__.py` | `serving/preprocessing/__init__.py` | **COPY** | Preserved. |
| `preprocessing/serving_wrapper.py` | `serving/preprocessing/serving_wrapper.py` | **MODIFY** | Update default paths to `experiments/E1_lightgbm/`. |
| `preprocessing/__pycache__/*`| *EXCLUDED* | **DROP** | Bytecode hygiene violation. |
| `src/features/ieee_cis_features.py` | *EXCLUDED* | **DROP** | Host repository already possesses canonical `src/features/ieee_cis_features.py`. |
| `src/__pycache__/*` | *EXCLUDED* | **DROP** | Bytecode hygiene violation. |
| `inference/__init__.py` | `serving/inference/__init__.py` | **COPY** | Preserved. |
| `inference/offline_inference.py` | `serving/inference/offline_inference.py` | **MODIFY** | Sanitize `C:\Users\HADASSAH KIRAN\` paths; point defaults to `experiments/E1_lightgbm/`. |
| `inference/phase2_benchmarking.py`| `serving/inference/phase2_benchmarking.py` | **MODIFY** | Update artifact paths. |
| `inference/phase4_consistency.py` | `serving/inference/phase4_consistency.py` | **MODIFY** | Update artifact paths. |
| `inference/phase5_api_benchmarking.py`| `serving/inference/phase5_api_benchmarking.py` | **COPY** | Preserved. |
| `inference/phase8_monitoring_experiment.py`| `serving/inference/phase8_monitoring_experiment.py`| **COPY** | Preserved. |
| `inference/__pycache__/*` | *EXCLUDED* | **DROP** | Bytecode hygiene violation. |
| `monitoring/__init__.py` | `serving/monitoring/__init__.py` | **COPY** | Preserved. |
| `monitoring/api_monitor.py` | `serving/monitoring/api_monitor.py` | **COPY** | Preserved. |
| `monitoring/inference_logger.py`| `serving/monitoring/inference_logger.py` | **COPY** | Preserved. |
| `monitoring/model_metadata.py` | `serving/monitoring/model_metadata.py` | **MODIFY** | Update documented artifact paths. |
| `monitoring/__pycache__/*` | *EXCLUDED* | **DROP** | Bytecode hygiene violation. |
| `models/E1/*` | *EXCLUDED* | **DROP** | Redundant duplication; use `experiments/E1_lightgbm/` directly. |
| `benchmarks/*` | `serving/benchmarks/*` | **COPY** | Preserve all benchmark reports and CSV result files. |
| `tests/__init__.py` | `serving/tests/__init__.py` | **COPY** | Preserved. |
| `tests/test_api.py` | `serving/tests/test_api.py` | **MODIFY** | Update imports and fixture paths. |
| `tests/test_offline_inference.py`| `serving/tests/test_offline_inference.py` | **MODIFY** | Update imports and fixture paths. |
| `tests/test_phase2_benchmarks.py`| `serving/tests/test_phase2_benchmarks.py` | **MODIFY** | Update imports and fixture paths. |
| `tests/test_phase4_consistency.py`| `serving/tests/test_phase4_consistency.py`| **MODIFY** | Update imports and fixture paths. |
| `tests/test_phase5_api_benchmarks.py`| `serving/tests/test_phase5_api_benchmarks.py`| **MODIFY** | Update imports and fixture paths. |
| `tests/test_phase6_error_handling.py`| `serving/tests/test_phase6_error_handling.py`| **COPY** | Preserved. |
| `tests/test_phase7_docker.py` | `serving/tests/test_phase7_docker.py` | **COPY** | Preserved. |
| `tests/test_phase8_monitoring.py`| `serving/tests/test_phase8_monitoring.py`| **COPY** | Preserved. |
| `tests/__pycache__/*` | *EXCLUDED* | **DROP** | Bytecode hygiene violation. |
| `Dockerfile` | `serving/Dockerfile` | **MODIFY** | Adjust build context for root repo or self-contained `serving/` directory. |
| `.dockerignore` | `serving/.dockerignore` | **COPY** | Preserved. |
| `requirements.txt` | `serving/requirements.txt` | **COPY** | Isolated serving dependencies (`fastapi`, `uvicorn`, `lightgbm`, etc.). |
| `run_single_transaction_demo.py`| `serving/run_single_transaction_demo.py` | **MODIFY** | Update artifact paths. |
| `README.md` | `serving/README.md` | **MODIFY** | Update documentation to clarify research-serving architecture and CRLF/LF hashes. |
| *(NEW)* `serving/hydration/` | `serving/hydration/` | **CREATE** | `entity_store.py`, `adapter.py`, `test_hydration.py`. |

---

## 3. Detailed File Transformations

### 3.1 Hardcoded Path Sanitization in `serving/inference/offline_inference.py`
**Before (External Source):**
```python
DEFAULT_DATASET_TX_PATH = r"C:\Users\HADASSAH KIRAN\Downloads\UPI Fraud Detection\IEEE CIS-20260902T145234Z-1-001\IEEE CIS\train_transaction.csv"
DEFAULT_DATASET_ID_PATH = r"C:\Users\HADASSAH KIRAN\Downloads\UPI Fraud Detection\IEEE CIS-20260902T145234Z-1-001\IEEE CIS\train_identity.csv"
```
**After (Integrated Target):**
```python
DATASET_DIR = Path(os.getenv("IEEE_CIS_DATASET_DIR", "data/raw"))
DEFAULT_DATASET_TX_PATH = DATASET_DIR / "train_transaction.csv"
DEFAULT_DATASET_ID_PATH = DATASET_DIR / "train_identity.csv"
```

### 3.2 Canonical Artifact Path Resolution
Across `offline_inference.py`, `serving_wrapper.py`, and `api/main.py`:
**Before:**
```python
DEFAULT_MODEL_PATH = "models/E1/model.txt"
DEFAULT_PREPROCESSOR_PATH = "models/E1/preprocessing.joblib"
DEFAULT_FEATURE_NAMES_PATH = "models/E1/feature_names.json"
```
**After:**
```python
DEFAULT_MODEL_PATH = PROJECT_ROOT / "experiments" / "E1_lightgbm" / "model.txt"
DEFAULT_PREPROCESSOR_PATH = PROJECT_ROOT / "experiments" / "E1_lightgbm" / "preprocessing.joblib"
DEFAULT_FEATURE_NAMES_PATH = PROJECT_ROOT / "experiments" / "E1_lightgbm" / "feature_names.json"
```

---

## 4. Stage-B Implementation Roadmap

1. **Phase 1: Structure Creation & File Migration**
   - Create directory layout under `serving/` (`api/`, `inference/`, `monitoring/`, `preprocessing/`, `benchmarks/`, `tests/`, `hydration/`).
   - Copy non-excluded files from scratch clone.
2. **Phase 2: Codebase Sanitization**
   - Apply path parameterization and canonical artifact references.
   - Verify that no `__pycache__` or `.pyc` files are added.
3. **Phase 3: Feature Hydration Adapter Development**
   - Implement `serving/hydration/entity_store.py` and `serving/hydration/adapter.py`.
   - Add unit test `serving/tests/test_feature_hydration.py`.
4. **Phase 4: Serving Test Suite Validation**
   - Execute all serving tests against canonical E1 artifacts:
     - `test_offline_inference.py`
     - `test_api.py`
     - `test_phase6_error_handling.py`
     - `test_phase8_monitoring.py`
     - `test_feature_hydration.py`
5. **Phase 5: Full Repository Regression Verification**
   - Execute the 173 baseline research and infrastructure tests to guarantee zero regression.
   - Confirm E1 SHA-256 and Git blob hashes remain 100% invariant.
6. **Phase 6: Stage-B Review & Sign-Off**
   - Present final integrated serving report.
   - Await explicit user instruction before any PR or merge into `Upto_Phase-4`.
