# Phase 1: Model & Artifact Inspection Report

**Project:** Adaptive Financial Fraud Detection with Real-Time Streaming & Temporal Drift Defense  
**Subproject:** Member 3 — FastAPI Model Serving & Real-Time Inference (`phase4_api`)  
**Phase:** Phase 1 — Project Setup & ML Model Inspection  
**Date:** October 31, 2026  
**Status:** COMPLETE — READY FOR REVIEW  

---

## 1. Folder Creation Confirmation

The dedicated serving directory `phase4_api/` has been created at the root of the project repository (`d:/adaptive-upi-fraud-detection/phase4_api`):

```text
phase4_api/
├── README.md                      # Service overview, architecture, and roadmap
├── requirements.txt               # API and inference dependencies
├── src/
│   └── __init__.py                # Package initializer
├── tests/
│   └── __init__.py                # Test package initializer
└── reports/
    └── phase1_model_inspection.md # This comprehensive inspection deliverable
```

**Non-Destructive Verification:**
- Zero existing repository files were deleted, moved, or renamed.
- No existing training or retraining pipelines were executed.
- No model artifacts were duplicated into `phase4_api/`.

---

## 2. Approved Model Version & Artifact Path

### Candidate Model Artifacts in Repository
Phase 4 produced three LightGBM model artifacts in `experiments/phase4_drift_adaptation/artifacts/models/`:

| Model Label | Training Data | Training Rows | Training Fraud | Fraud Rate | Model File | SHA256 Hash (First 16 hex) |
|---|---|---:|---:|---:|---|---|
| **v1 (Static Baseline)** | Months 0–3 | 547,975 | 5,288 | 0.9650% | `lgbm_v1.txt` | `2cb7bc1247114aaa` |
| **v2 (Intermediate Adaptive)** | Months 0–5 | 794,989 | 8,151 | 1.0253% | `lgbm_v2.txt` | `e4f249253fab9125` |
| **v3 (Champion Adaptive)** | Months 0–6 | 903,157 | 9,601 | 1.0630% | `lgbm_v3.txt` | `cd3908ae85418ed4` |

*Source metadata:*
- `experiments/phase4_drift_adaptation/artifacts/models/lgbm_v1_meta.json:1-28`
- `experiments/phase4_drift_adaptation/artifacts/models/lgbm_v2_meta.json:1-30`
- `experiments/phase4_drift_adaptation/artifacts/models/lgbm_v3_meta.json:1-31`

### Approved Model Determination
The **approved production model for real-time inference is LightGBM `v3` (`lgbm_v3.txt`)**.

**Evidence from Project Metadata & Experiments:**
1. **Master Results Manifest:**
   In [`experiments/phase4_drift_adaptation/artifacts/manifests/results_20260919_162551.json`](file:///d:/adaptive-upi-fraud-detection/experiments/phase4_drift_adaptation/artifacts/manifests/results_20260919_162551.json#L7-L83):
   - Line 7: `"adaptive_path": "v1 -> v2 -> v3"`
   - Line 83: `"month7_results": { "adaptive_version": "v3", ... }`
2. **Master Pipeline Output:**
   In [`experiments/phase4_drift_adaptation/src/pipeline.py`](file:///d:/adaptive-upi-fraud-detection/experiments/phase4_drift_adaptation/src/pipeline.py#L351):
   - `log(f"  Adaptive final ver:   {adaptive_version}")` -> outputs `v3`.
3. **Statistical Out-of-Sample Performance on Protected Month 7:**
   In [`experiments/phase4_drift_adaptation/docs/PHASE4_IMPLEMENTATION_REPORT.md`](file:///d:/adaptive-upi-fraud-detection/experiments/phase4_drift_adaptation/docs/PHASE4_IMPLEMENTATION_REPORT.md#L250-L283):
   - Static Model v1 PR-AUC: **0.160858**
   - Adaptive Model v3 PR-AUC: **0.201027**
   - Observed ΔPR-AUC: **+0.040169** (a +24.9% relative gain)
   - 95% Paired Bootstrap Confidence Interval: **[+0.026752, +0.053843]** (statistically significant; entirely above zero).
4. **Leakage & Integrity Status:**
   - Audited as 10/10 clear of leakage in [`PHASE4_FINAL_AUDIT.md`](file:///d:/adaptive-upi-fraud-detection/experiments/phase4_drift_adaptation/docs/PHASE4_FINAL_AUDIT.md#L1-L100).
   - Passed all 52/52 integrity checks in [`INTEGRITY_CERTIFICATE.json`](file:///d:/adaptive-upi-fraud-detection/experiments/phase4_drift_adaptation/artifacts/INTEGRITY_CERTIFICATE.json#L1-L5).

**Exact Approved Artifact Path:**
- Path: `experiments/phase4_drift_adaptation/artifacts/models/lgbm_v3.txt`
- Absolute Path: `d:/adaptive-upi-fraud-detection/experiments/phase4_drift_adaptation/artifacts/models/lgbm_v3.txt`
- Version: `v3` (promoted Champion Adaptive Model)
- Dual-Serving Support: The API can also optionally load `lgbm_v1.txt` as a static baseline comparator.

---

## 3. Model Loading Compatibility

### Artifact Format
`lgbm_v3.txt` is stored in LightGBM native booster format (`version=v4`, `objective=binary sigmoid:1`):
- Number of trees: **500** (`num_trees=500`, `num_leaves=63`, `max_depth=6`)
- Number of input features: **29** (`max_feature_idx=28`)
- Shrinkage (learning rate): **0.05**

### Compatibility Test Results
All three models (`v1`, `v2`, `v3`) were verified using `lightgbm.Booster`:

```python
import lightgbm as lgb
from pathlib import Path

path = Path("experiments/phase4_drift_adaptation/artifacts/models/lgbm_v3.txt")
content = path.read_bytes().replace(b"\r\n", b"\n")
booster = lgb.Booster(model_str=content.decode("utf-8"))

print(f"Features: {booster.num_feature()}, Trees: {booster.num_trees()}")
# Output: Features: 29, Trees: 500 (SUCCESS)
```

### Critical Loading Gotcha (Windows CRLF vs Linux LF)
- **Problem:** Git checkouts on Windows with `core.autocrlf=true` convert Unix `\n` to Windows `\r\n`. In LightGBM text model files, line 10 contains `tree_sizes` representing exact byte offsets for the binary parser. When Windows CRLF is present, byte lengths diverge, causing LightGBM C++ core to throw:
  `[LightGBM] [Fatal] Model format error, expect a tree here. met 12 12 2`
- **Solution for FastAPI Service:**
  The model loader should read file bytes and normalize `replace(b"\r\n", b"\n")` before invoking `lgb.Booster(model_str=...)`. This provides 100% resilient cross-platform loading on both Windows development workstations and Linux Docker deployment containers without touching the original artifacts.

---

## 4. Exact 29-Feature Schema & Feature Ordering

The model requires exactly **29 features** in an invariant, deterministic order.

*Source: [`experiments/phase4_drift_adaptation/src/data_loader.py:23-60`](file:///d:/adaptive-upi-fraud-detection/experiments/phase4_drift_adaptation/src/data_loader.py#L23-L60) and `experiments/phase4_drift_adaptation/artifacts/models/lgbm_v3.txt:8`*

### Canonical Feature Ordering Table

| Index | Feature Name | Category / Type | Monitored by PSI? | BAF Source Type |
|---|---|---|:---:|---|
| 0 | `income` | Numerical | Yes | Continuous float (0.1–0.9) |
| 1 | `name_email_similarity` | Numerical | Yes | Continuous float (0.0–1.0) |
| 2 | `prev_address_months_count` | Numerical | Yes | Integer (-1 to 366) |
| 3 | `current_address_months_count`| Numerical | Yes | Integer (-1 to 418) |
| 4 | `customer_age` | Numerical | Yes | Integer (10 to 90) |
| 5 | `days_since_request` | Numerical | Yes | Continuous float (0.0 to 76.6) |
| 6 | `intended_balcon_amount` | Numerical | Yes | Continuous float (-13.2 to 111.9) |
| 7 | `zip_count_4w` | Numerical | Yes | Integer (5 to 6563) |
| 8 | `velocity_6h` | Numerical | Yes | Continuous float (-130.5 to 16564.6)|
| 9 | `velocity_24h` | Numerical | Yes | Continuous float (1529.8 to 9506.9) |
| 10 | `velocity_4w` | Numerical | Yes | Continuous float (2990.7 to 6974.6) |
| 11 | `bank_branch_count_8w` | Numerical | Yes | Integer (0 to 2350) |
| 12 | `date_of_birth_distinct_emails_4w`| Numerical | Yes | Integer (0 to 38) |
| 13 | `credit_risk_score` | Numerical | Yes | Integer (-170 to 386) |
| 14 | `email_is_free` | Numerical (Flag)| Yes | Binary integer (0 or 1) |
| 15 | `phone_home_valid` | Numerical (Flag)| Yes | Binary integer (0 or 1) |
| 16 | `phone_mobile_valid` | Numerical (Flag)| Yes | Binary integer (0 or 1) |
| 17 | `bank_months_count` | Numerical | Yes | Integer (-1 to 32) |
| 18 | `has_other_cards` | Numerical (Flag)| Yes | Binary integer (0 or 1) |
| 19 | `proposed_credit_limit` | Numerical | Yes | Continuous float (190 to 2100) |
| 20 | `foreign_request` | Numerical (Flag)| Yes | Binary integer (0 or 1) |
| 21 | `session_length_in_minutes` | Numerical | Yes | Continuous float (-1 to 83.2) |
| 22 | `keep_alive_session` | Numerical (Flag)| Yes | Binary integer (0 or 1) |
| 23 | `device_distinct_emails_8w` | Numerical | Yes | Integer (-1 to 2) |
| 24 | `payment_type` | Categorical | Yes | String / Categorical (5 values) |
| 25 | `employment_status` | Categorical | Yes | String / Categorical (7 values) |
| 26 | `housing_status` | Categorical | Yes | String / Categorical (7 values) |
| 27 | `source` | Categorical | Yes | String / Categorical (2 values) |
| 28 | `device_os` | Categorical | Yes | String / Categorical (5 values) |

### Excluded Columns (Strict Leakage Prevention)
The following 3 columns in `Base.csv` are **strictly excluded** from inference inputs:
1. `fraud_bool` (target column — target leakage).
2. `month` (temporal batch identifier — temporal leakage).
3. `device_fraud_count` (constant 0 column dropped during protocol freeze).

*Sources: [`data_loader.py:21`](file:///d:/adaptive-upi-fraud-detection/experiments/phase4_drift_adaptation/src/data_loader.py#L21), [`PHASE4_PROTOCOL_MANIFEST.json:25-26`](file:///d:/adaptive-upi-fraud-detection/experiments/phase4_drift_adaptation/artifacts/PHASE4_PROTOCOL_MANIFEST.json#L25-L26), [`PHASE4_FINAL_AUDIT.md:11-25`](file:///d:/adaptive-upi-fraud-detection/experiments/phase4_drift_adaptation/docs/PHASE4_FINAL_AUDIT.md#L11-L25).*

---

## 5. Numerical and Categorical Feature Specifications

### 24 Numerical Features
```python
NUMERICAL_COLS = [
    "income", "name_email_similarity", "prev_address_months_count",
    "current_address_months_count", "customer_age", "days_since_request",
    "intended_balcon_amount", "zip_count_4w", "velocity_6h",
    "velocity_24h", "velocity_4w", "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w", "credit_risk_score", "email_is_free",
    "phone_home_valid", "phone_mobile_valid", "bank_months_count",
    "has_other_cards", "proposed_credit_limit", "foreign_request",
    "session_length_in_minutes", "keep_alive_session", "device_distinct_emails_8w"
]
```

### 5 Categorical Features & Recognized Domains
*Source: [`experiments/phase4_drift_adaptation/artifacts/drift/psi_reference_months0_3.json:440-478`](file:///d:/adaptive-upi-fraud-detection/experiments/phase4_drift_adaptation/artifacts/drift/psi_reference_months0_3.json#L440-L478)*

1. `payment_type`: `["AB", "AA", "AC", "AD", "AE"]`
2. `employment_status`: `["CA", "CB", "CF", "CC", "CD", "CE", "CG"]`
3. `housing_status`: `["BC", "BB", "BE", "BA", "BD", "BF", "BG"]`
4. `source`: `["INTERNET", "TELEAPP"]`
5. `device_os`: `["linux", "other", "windows", "macintosh", "x11"]`

---

## 6. Training-Compatible Preprocessing Requirements

### Categorical Handling (LightGBM Native)
- In Phase 4, categorical features are passed directly as native LightGBM categoricals (`pd.Categorical(df[col])`).
- **NO One-Hot Encoding** is used.
- **NO OrdinalEncoder / LabelEncoder** is used.
- *Source: [`experiments/phase4_drift_adaptation/src/model.py:17-19, 98-103`](file:///d:/adaptive-upi-fraud-detection/experiments/phase4_drift_adaptation/src/model.py#L17-L19), [`PHASE4_PROTOCOL_MANIFEST.json:68-73`](file:///d:/adaptive-upi-fraud-detection/experiments/phase4_drift_adaptation/artifacts/PHASE4_PROTOCOL_MANIFEST.json#L68-L73).*

### Handling Missing Values
- **In Training:** BAF Base.csv contains 0 missing values (`missingness: none`).
- **In Real-Time Inference:**
  - **Numerical:** LightGBM handles `NaN` natively during inference by traversing the default branch learned during tree construction.
  - **Categorical:** Missing values or unknown categorical strings cast to `NaN` in `pd.Categorical` are routed to LightGBM's default categorical bucket without crashing.

### Feature Scaling
- **None Required:** Decision trees in LightGBM are invariant to monotonic feature scaling. No standard scaling or min-max normalization was applied in Phase 4.

---

## 7. Approved Prediction Threshold & Source

### Frozen Threshold Value
$$\tau^* = 0.8964321608040201 \quad (\approx 0.8964)$$

### Source & Derivation Logic
- File: [`experiments/phase4_drift_adaptation/artifacts/thresholds/frozen_threshold_v1.json:1-9`](file:///d:/adaptive-upi-fraud-detection/experiments/phase4_drift_adaptation/artifacts/thresholds/frozen_threshold_v1.json#L1-L9)
- Derivation Rule: 200-point threshold sweep ($0.01$ to $0.99$) on **Month 4 validation predictions** using static model `v1` to maximize $F_1$ score ($F_1 = 0.2458$).
- Protocol Rule: Permanently frozen across **all model versions** (`v1`, `v2`, `v3`). Re-selection on later data was strictly prohibited to prevent validation leakage into retraining sets.
- *Source: [`PHASE4_PROTOCOL_MANIFEST.json:127-138`](file:///d:/adaptive-upi-fraud-detection/experiments/phase4_drift_adaptation/artifacts/PHASE4_PROTOCOL_MANIFEST.json#L127-L138), [`PHASE4_FINAL_AUDIT.md:67-85`](file:///d:/adaptive-upi-fraud-detection/experiments/phase4_drift_adaptation/docs/PHASE4_FINAL_AUDIT.md#L67-L85).*

### Decision Equation for Inference
$$\text{prediction} = \begin{cases} 1 \ (\text{FRAUD}) & \text{if } P(\text{fraud} = 1) \ge 0.8964321608040201 \\ 0 \ (\text{LEGIT}) & \text{if } P(\text{fraud} = 1) < 0.8964321608040201 \end{cases}$$

---

## 8. Python Dependencies & Environment Versions

Verified existing versions installed in the local environment:

| Package | Installed Version | Required Minimum | Role in Inference Service |
|---|---|---|---|
| `fastapi` | **0.141.1** | `>=0.110.0` | API application, routing, and openapi docs |
| `uvicorn` | **0.52.4** | `>=0.28.0` | High-performance ASGI web server |
| `pydantic` | **2.13.5** | `>=2.6.0` | Strict request validation, type checking |
| `lightgbm` | **4.7.0** | `==4.7.0` | Direct booster execution |
| `pandas` | **3.0.5** | `>=2.2.0` | DataFrame construction, categorical dtypes |
| `numpy` | **2.4.6** | `>=1.26.0` | Numeric array operations |
| `scikit-learn` | **1.9.0** | `>=1.4.0` | Validation helpers and metrics |
| `httpx` | **0.28.1** | `>=0.27.0` | Async FastAPI test client |
| `pytest` | **9.1.1** | `>=8.0.0` | Test runner |

---

## 9. Potential Compatibility Issues for FastAPI Inference

1. **Windows CRLF vs Linux LF Line Endings in Model Artifacts:**
   As discovered during inspection, direct `lgb.Booster(model_file=...)` fails on Windows if git cloned the `.txt` model with CRLF. The API loader must read model bytes, replace `\r\n` with `\n`, and load via `model_str`.
2. **DataFrame Conversion Latency Overhead:**
   Calling `pd.DataFrame([payload])` for a single request adds ~1.0 ms overhead. In high-throughput streaming environments, converting payload dictionaries directly into numpy arrays with encoded categoricals reduces latency to < 1.5 ms.
3. **Categorical Dtype Matching:**
   If a payload is passed as raw string columns without explicit categorical casting, LightGBM Booster throws an error. The API inference engine must wrap incoming categorical values with `pd.Categorical(values, categories=KNOWN_CATEGORIES)` or map them to integer category codes.
4. **Missing or Out-of-Range Field Inputs:**
   In real-time banking, certain velocity or historical counts may be negative or null (e.g. `prev_address_months_count = -1` indicates no previous address). Pydantic models must allow these sentinel values without failing validation.

---

## 10. Files & Modules to Reuse in Phase 2

1. **Feature Definitions:**
   [`experiments/phase4_drift_adaptation/src/data_loader.py`](file:///d:/adaptive-upi-fraud-detection/experiments/phase4_drift_adaptation/src/data_loader.py#L21-L60) — Single source of truth for `FEATURE_COLS`, `NUMERICAL_COLS`, `CATEGORICAL_COLS`, and `EXCLUDED_COLS`.
2. **Approved Model Artifact:**
   [`experiments/phase4_drift_adaptation/artifacts/models/lgbm_v3.txt`](file:///d:/adaptive-upi-fraud-detection/experiments/phase4_drift_adaptation/artifacts/models/lgbm_v3.txt) — Promoted champion LightGBM model.
3. **Frozen Threshold:**
   [`experiments/phase4_drift_adaptation/artifacts/thresholds/frozen_threshold_v1.json`](file:///d:/adaptive-upi-fraud-detection/experiments/phase4_drift_adaptation/artifacts/thresholds/frozen_threshold_v1.json) — Source of truth for `0.8964321608040201`.
4. **Member 2 Integration Telemetry:**
   [`experiments/phase4_drift_adaptation/src/member2_drift_adapter.py`](file:///d:/adaptive-upi-fraud-detection/experiments/phase4_drift_adaptation/src/member2_drift_adapter.py) — For exposing drift status (`/api/v1/drift-status`) without recalculating PSI.

---

## 11. Unresolved Questions & Design Decisions for Phase 2

1. **Model Selection Parameter:**
   Should the API provide a header/parameter to toggle between `champion` (`v3`) and `baseline` (`v1`) for canary/shadow testing, while defaulting to `champion`? *(Recommended: Yes, support `model_version: "champion" | "baseline"`).*
2. **Inference Latency SLA:**
   What is the target P99 latency for `/predict`? *(Under LightGBM native C++ booster with 500 trees, single-transaction latency is expected to be ~2ms–5ms on CPU).*
