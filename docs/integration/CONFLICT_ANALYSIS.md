# Conflict Analysis: Member 1 & Member 2 Integration

**Target Repository:** `adaptive-upi-fraud-detection` (Branch `Upto_Phase-4`)  
**Source Repository:** `HarikaReddy440/Kafka-spark-flink` (HEAD `f74042f`)  
**Audit Date:** October 2026  
**Auditor:** Senior Software & Research Integration Engineer

---

## 1. Executive Summary

This conflict analysis methodically audits 23 potential friction points between the existing Phase 1–4 research baseline and the incoming Member 1 (Kafka) and Member 2 (Spark/Flink) streaming and batch data processing code.

The primary finding is that **the canonical research pipeline (Phases 1–4) and the real-time infrastructure operate on distinct, decoupled concerns**. The real-time system acts as an ingestion and feature processing pipeline that prepares data for downstream scoring, while the research track defines the model architectures, training protocols, and statistical baselines.

Because Member 2 nested their work inside `spark/spark/` and `spark/flink/` while carrying an older duplicate of the research repo, the most significant risk is accidental overwriting of research artifacts. This risk is 100% mitigated by unpacking only the unique components into top-level `spark/` and `flink/` while excluding the duplicate research directories.

---

## 2. Comprehensive 23-Point Conflict Matrix

### 1. `requirements.txt`
- **Existing Target Behavior:** Minimal root requirements file (16 lines) containing core research dependencies: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `pyarrow`, `shap`, `scipy`, `joblib`, `pyyaml`, `matplotlib`, `seaborn`.
- **Member 1/2 Behavior:** Source has component-specific requirement files (`spark/spark/requirements.txt`, `spark/flink/requirements.txt`) containing `pyspark>=3.5,<4`, `apache-flink==2.3.0`, `kafka-python>=2.0`, `psutil>=5.9`.
- **Risk:** Blindly overwriting root `requirements.txt` would either break research environments or introduce heavyweight PySpark/PyFlink dependencies into lightweight ML setups.
- **Recommended Resolution:** Keep the root `requirements.txt` clean and dedicated to the core research models. Maintain dedicated requirement files in `kafka/requirements.txt`, `spark/requirements.txt`, and `flink/requirements.txt`. Add an optional `requirements-all.txt` or documented extras.
- **Safe to Modify?** Yes (keep root `requirements.txt` intact; add modular component requirement files).

---

### 2. `.gitignore`
- **Existing Target Behavior:** Comprehensive rules ignoring `Datasets/`, `datasets/`, `__pycache__`, virtual environments, `experiments/E1_encoding_comparison/`, `reports/*.pdf`, and `experiments/E3_hybrid/`. Explicitly tracks `model.txt`, `preprocessing.joblib`, `predictions.parquet`.
- **Member 1/2 Behavior:** Source ignores parquet benchmarks, `.vscode`, `.idea`, python cache.
- **Risk:** If root `.gitignore` is overwritten blindly, large raw datasets or generated parquet outputs could accidentally get tracked in Git.
- **Recommended Resolution:** Merge new rules into target `.gitignore` without removing existing research exclusions. Add rules for:
  - `kafka/data/` (Kafka KRaft broker log directories)
  - `spark/output/` & `flink/output/` (Streaming checkpoints and parquet sinks)
  - `*.jar` exclusions if any exceed GitHub limits (current jars are 464KB and 8.4MB, safe to track or ignore).
- **Safe to Modify?** Yes (append new infrastructure ignore patterns safely).

---

### 3. `README.md`
- **Existing Target Behavior:** Complete multi-phase research guide detailing Phase 1–4, teammate guidance, model comparison table, and BAF dataset instructions.
- **Member 1/2 Behavior:** Contains Member 1 Kafka README, Member 2 Spark README, and Flink README.
- **Risk:** Overwriting target `README.md` would erase the entire research documentation and teammate guidance.
- **Recommended Resolution:** Keep existing research documentation intact. Add a dedicated section: *"Real-Time Infrastructure Track (Member 1: Kafka & Member 2: Spark/Flink)"* with architectural diagrams, execution commands, and quickstart instructions.
- **Safe to Modify?** Yes (additive integration).

---

### 4. `configs/`
- **Existing Target Behavior:** `configs/phase1_lightgbm.yaml` contains Phase 1 hyperparameters.
- **Member 1/2 Behavior:** `spark/spark/configs/spark_config.yaml`, `spark/flink/configs/flink_config.yaml`, `spark/flink/configs/kafka_config.yaml`, and `kafka/config/config.py`.
- **Risk:** Path collisions or namespace ambiguity.
- **Recommended Resolution:** Store infrastructure configs inside their component directories:
  - `kafka/config/config.py`
  - `spark/configs/spark_config.yaml`
  - `flink/configs/flink_config.yaml`
  - `flink/configs/kafka_config.yaml`  
  Keep research configs in `configs/phase1_lightgbm.yaml`.
- **Safe to Modify?** Yes (zero filename collision).

---

### 5. `tests/`
- **Existing Target Behavior:** `tests/test_phase1.py` (10 unit tests for Phase 1).
- **Member 1/2 Behavior:**
  - `kafka/tests/` (2 test suites)
  - `spark/spark/tests/` (4 test suites)
  - `spark/flink/tests/` (13 test suites)
  - Root: `run_full_e2e_integration_test.py` (14-step integration test)
- **Risk:** Overwriting `tests/test_phase1.py` or mixing unit and end-to-end integration tests.
- **Recommended Resolution:** Keep unit tests inside their respective modules (`kafka/tests/`, `spark/tests/`, `flink/tests/`). Create `tests/integration/` for multi-process integration tests (`run_full_e2e_integration_test.py`). Preserve `tests/test_phase1.py`.
- **Safe to Modify?** Yes.

---

### 6. `reports/`
- **Existing Target Behavior:** `reports/phase1/`, `reports/phase2/`, `DATASET_FORENSICS.json`, `RESEARCH_STATUS_COMPLETE_REVIEW.md`.
- **Member 1/2 Behavior:** Source has `MEMBER1_FINAL_DELIVERABLES.md`, `SPARK_AND_FLINK_IMPLEMENTATION_AND_RESULTS.md`, `COMPREHENSIVE_SYSTEM_AND_BENCHMARK_REPORT.md`, `WSL_BENCHMARK_RESULTS.md`, `FLINK_RESEARCH_REPORT.md`, `member1_*_results.csv`.
- **Risk:** Report pollution or overwriting canonical research reports.
- **Recommended Resolution:** Organize infrastructure reports under `docs/integration/` and module-specific report subdirectories (`kafka/reports/`, `spark/reports/`, `flink/reports/`). Leave research `reports/` completely untouched.
- **Safe to Modify?** Yes.

---

### 7. `scripts/` & `analysis_scripts/`
- **Existing Target Behavior:** Target has `analysis_scripts/` with 7 exploratory dataset analysis scripts.
- **Member 1/2 Behavior:** Source carried a duplicate copy of `analysis_scripts/`.
- **Risk:** Overwriting target scripts.
- **Recommended Resolution:** Exclude `spark/analysis_scripts/` from the merge. Keep target `analysis_scripts/` untouched.
- **Safe to Modify?** Yes (exclusion avoids conflict).

---

### 8. `experiments/`
- **Existing Target Behavior:** `experiments/E1_lightgbm/`, `E3_hybrid/`, `phase4_drift_adaptation/`, `run_phase1.py`.
- **Member 1/2 Behavior:** Source carried an older duplicate of `experiments/` inside `spark/experiments/`.
- **Risk:** **CRITICAL RISK:** Overwriting frozen model checkpoints, predictions, or metrics.
- **Recommended Resolution:** **STRICT EXCLUSION.** Completely exclude `spark/experiments/` from the merge.
- **Safe to Modify?** Excluded. Target `experiments/` remains immutable.

---

### 9. `src/`
- **Existing Target Behavior:** Canonical research source code (`audit/`, `data/`, `evaluation/`, `features/`, `models/`, `utils/`).
- **Member 1/2 Behavior:** Source carried duplicate `spark/src/`.
- **Risk:** Accidental regression of research feature engineering or data loaders.
- **Recommended Resolution:** **STRICT EXCLUSION.** Completely exclude `spark/src/`. Target `src/` remains untouched.
- **Safe to Modify?** Excluded. Target `src/` remains immutable.

---

### 10. Dataset Paths
- **Existing Target Behavior:** Datasets stored under `Datasets/` (e.g. `Datasets/IEEE CIS-.../IEEE CIS/train_transaction.csv`, `Datasets/BAF/Base.csv`).
- **Member 1/2 Behavior:** Some scripts in source had hardcoded paths such as `/mnt/c/Users/HADASSAH KIRAN/Downloads/Datasets/...`.
- **Risk:** `FileNotFoundError` on any machine other than the original developer's.
- **Recommended Resolution:** Refactor hardcoded paths to use flexible resolution: check `Datasets/IEEE CIS-...`, `Datasets/raw/`, `Datasets/train_transaction.csv`, or an optional `--dataset` CLI argument.
- **Safe to Modify?** Yes.

---

### 11. Environment Variables
- **Existing Target Behavior:** Standard Python environment variables.
- **Member 1/2 Behavior:** `spark/spark/streaming/stream_processor.py` sets:
  ```python
  os.environ["PYSPARK_PYTHON"] = sys.executable
  os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
  ```
- **Risk:** Minor on non-Windows; beneficial on Windows to ensure driver and worker use the same Python binary.
- **Recommended Resolution:** Preserve these PySpark environment variable guards in Spark scripts.
- **Safe to Modify?** Yes (safe and recommended for Windows environments).

---

### 12. Python Versions
- **Existing Target Behavior:** Python 3.13.13 in active Conda environment.
- **Member 1/2 Behavior:** Member 1 developed on Python 3.10/3.11. Member 2 Flink was tested in WSL Ubuntu with Python 3.11 (since PyFlink has limited wheels on Python 3.13).
- **Risk:** PyFlink may not run natively on Windows with Python 3.13 without WSL or Docker.
- **Recommended Resolution:** Pure Python components (Kafka producer/consumer, reference features, unit tests) run seamlessly on Python 3.13. Document that full PyFlink and PySpark distributed cluster execution can be run either natively or inside WSL2/Ubuntu with Python 3.11/Java 17 as benchmarked by Member 2.
- **Safe to Modify?** Yes (document compatibility transparently).

---

### 13. Java Versions
- **Existing Target Behavior:** Research track requires no Java.
- **Member 1/2 Behavior:** Spark 3.5.9 and Flink 2.2 require Java 11 or 17 (OpenJDK).
- **Risk:** If Java is missing, native Spark/Flink invocations will fail. Standalone Python fallback exists in `spark_streaming_consumer.py`.
- **Recommended Resolution:** Document Java 17 prerequisite for distributed Spark/Flink jobs. Retain standalone Python simulation modes for environments lacking JVM runtimes.
- **Safe to Modify?** Yes.

---

### 14. Spark Versions
- **Existing Target Behavior:** None.
- **Member 1/2 Behavior:** Apache Spark 3.5.9 (`pyspark==3.5.9`, `spark-sql-kafka-0-10_2.12:3.5.9`).
- **Risk:** Version mismatch with Kafka connector.
- **Recommended Resolution:** Lock PySpark requirement to `pyspark>=3.5,<4` in `spark/requirements.txt`.
- **Safe to Modify?** Yes.

---

### 15. Flink Versions
- **Existing Target Behavior:** None.
- **Member 1/2 Behavior:** Apache Flink 2.2 / 2.3 (`apache-flink==2.3.0`, bundled connector jar `flink-connector-kafka-3.3.0-1.20.jar`).
- **Risk:** Connector compatibility with Kafka brokers.
- **Recommended Resolution:** Keep the tested connector JARs in `flink/jars/` and lock PyFlink version in `flink/requirements.txt`.
- **Safe to Modify?** Yes.

---

### 16. Kafka Versions
- **Existing Target Behavior:** None.
- **Member 1/2 Behavior:** Apache Kafka 3.x+ (KRaft mode, Kafka protocol 2.0+ client `kafka-python>=2.0`).
- **Risk:** None observed.
- **Recommended Resolution:** Standardize on `kafka-python>=2.0`.
- **Safe to Modify?** Yes.

---

### 17. Network Ports
- **Existing Target Behavior:** None.
- **Member 1/2 Behavior:**
  - Kafka Broker: `9092`
  - Multi-node KRaft Brokers: `9092`, `9094`, `9096`
  - Multi-node KRaft Quorum Controllers: `9093`, `9095`, `9097`
  - Flink Web UI: `8081`
  - Spark Web UI: `4040`
- **Risk:** Port collisions if local services are already running on ports 9092 or 8081.
- **Recommended Resolution:** Make broker ports configurable via CLI (`--broker localhost:9092`) and configuration files (`kafka/config/config.py`, `spark/configs/spark_config.yaml`).
- **Safe to Modify?** Yes.

---

### 18. Kafka Broker Addresses
- **Existing Target Behavior:** None.
- **Member 1/2 Behavior:** Default is `localhost:9092` across all scripts.
- **Risk:** Hardcoded IP addresses in scripts.
- **Recommended Resolution:** Consistently pass `--broker` parameter across all CLI runners and defaults.
- **Safe to Modify?** Yes.

---

### 19. Kafka Topic Names
- **Existing Target Behavior:** None.
- **Member 1/2 Behavior:**
  - Raw / Replay Streams: `transactions`, `ieee_cis_transactions`, `fraud-transactions`.
  - Feature Sinks: `fraud-features`.
  - Partition Scale Tests: `transactions_p1`, `transactions_p3`, `transactions_p6`, `transactions_p12`.
- **Risk:** Topic mismatch between producer and consumer.
- **Recommended Resolution:** Document canonical topics:
  - Canonical Ingestion Topic: `fraud-transactions` (alias: `transactions`, `ieee_cis_transactions`).
  - Canonical Feature Sink: `fraud-features`.
- **Safe to Modify?** Yes.

---

### 20. Package / Module Names
- **Existing Target Behavior:** Root packages: `src`, `phase2_gru`.
- **Member 1/2 Behavior:**
  - Source imports: `from spark.batch... import ...`, `from flink.features... import ...`.
  - In source, these were nested under `spark/spark` and `spark/flink`.
- **Risk:** Broken imports if placed inside nested directories.
- **Recommended Resolution:** Place Spark at root `spark/` and Flink at root `flink/`. This aligns with the import statements (`import spark...`, `import flink...`) and eliminates any import path hacking.
- **Safe to Modify?** Yes.

---

### 21. Relative Paths & Root Resolution
- **Existing Target Behavior:** Research scripts resolve `ROOT = Path(__file__).resolve().parent.parent`.
- **Member 1/2 Behavior:** Several scripts use `Path(__file__).resolve().parents[2]` or hardcoded paths.
- **Recommended Resolution:** Standardize path resolution in all integrated scripts to dynamically find the workspace root.
- **Safe to Modify?** Yes.

---

### 22. Output Directories
- **Existing Target Behavior:** Research outputs in `experiments/`, `reports/`.
- **Member 1/2 Behavior:** Spark writes to `spark/output/`, Flink writes to `flink/output/`.
- **Recommended Resolution:** Maintain output directory isolation:
  - Spark outputs: `spark/output/`
  - Flink outputs: `flink/output/`
  - Ensure both are ignored by `.gitignore`.
- **Safe to Modify?** Yes.

---

### 23. Logging Configuration
- **Existing Target Behavior:** Research logging uses standard Python `logging` configured in `src/utils/logging_config.py`.
- **Member 1/2 Behavior:** Spark uses Log4j (`spark.sparkContext.setLogLevel("WARN")`); Kafka scripts print formatted status banners and flush stdout.
- **Recommended Resolution:** Retain component-specific logging. No global logging conflict exists.
- **Safe to Modify?** Yes.
