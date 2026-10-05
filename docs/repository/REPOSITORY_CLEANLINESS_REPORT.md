# REPOSITORY CLEANLINESS & HYGIENE REPORT

**Repository:** `adaptive-upi-fraud-detection`  
**Current Branch:** `Upto_Phase-4` (`c78494e`)  
**Audit Scope:** Cleanliness scan across tracked and untracked files.

---

## 1. Cleanliness Audit Findings

| Category | Scan Target | Result | Action Taken |
| :--- | :--- | :---: | :--- |
| **Duplicate Research Trees** | `spark/experiments/`, `spark/src/`, `spark/phase2_gru/` | **CLEAN** | Excluded during Stage B merge; zero duplicate research files remain. |
| **Secrets & Credentials** | `.env`, `credentials.json`, `*.key`, `*.pem` | **CLEAN** | 0 secrets found in git index or workspace. |
| **Raw Datasets** | `Datasets/`, `train_transaction.csv`, `Base.csv` | **CLEAN** | All raw data is excluded by `.gitignore`; 0 raw datasets committed. |
| **Bytecode & Build Artifacts**| `__pycache__`, `*.pyc`, `*.pyo`, `.pytest_cache` | **CLEAN** | Fully ignored by `.gitignore`; 0 compiled binaries committed. |
| **IDE Configurations** | `.vscode/settings.json`, `.idea/` | **CLEAN** | Excluded from merge; zero machine-specific IDE files tracked. |
| **Backup / Temp Files** | `*.backup`, `*.before_*`, `*.orig`, `*.tmp` | **CLEAN** | 0 temporary/backup files tracked. |
| **Hardcoded Paths** | Developer usernames (`HADASSAH KIRAN`, etc.) | **CLEAN** | All paths converted to dynamic `pathlib.Path` relative to repo root. |

---

## 2. Working Tree Verification

- **Git Status:** `nothing to commit, working tree clean`
- **Tracked Large Files (> 1MB):** Limited strictly to legitimate model weights (`model.txt`, `gru_best.pt`, `lgbm_v[1-3].txt`), test predictions (`predictions.parquet`), and necessary connector JARs (`kafka-clients-3.8.1.jar`).
- **Verdict:** 🟢 **READY FOR RESEARCH REPRODUCTION AND ENGINEERING COLLABORATION** (Not a live production fraud-detection deployment)
