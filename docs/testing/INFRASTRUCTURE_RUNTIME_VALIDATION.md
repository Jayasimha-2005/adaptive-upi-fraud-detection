# INFRASTRUCTURE RUNTIME VALIDATION GUIDE

**Repository:** `adaptive-upi-fraud-detection`  
**Purpose:** Guide for validating the 9 environmentally skipped tests and running live distributed benchmarks in dedicated, isolated environments without contaminating the host research conda environment.

---

## 1. Prerequisites

- Java 21 LTS (or Java 11/17) with `JAVA_HOME` configured.
- Apache Kafka 3.8 binaries (for KRaft cluster commands).
- Python 3.10+ virtual environments.

---

## 2. Validating Member 2 (Apache Spark) Skips

To execute `test_spark_batch.py` and `test_previous_transaction_stateful.py`:

```powershell
# 1. Create dedicated Spark virtual environment
python -m venv .venv-spark
.\.venv-spark\Scripts\Activate.ps1

# 2. Install isolated Spark dependencies
pip install -r spark/requirements.txt

# 3. Run all Spark unit and distributed tests
pytest spark/tests/ -v

# 4. Run Parquet conversion and streaming benchmarks
python spark/benchmarks/convert_to_parquet.py
python spark/benchmarks/run_benchmarks.py

# 5. Clean up
deactivate
```

---

## 3. Validating Member 2 (Apache Flink) Skips

To execute the 7 PyFlink streaming window tests:

```powershell
# 1. Create dedicated Flink virtual environment
python -m venv .venv-flink
.\.venv-flink\Scripts\Activate.ps1

# 2. Install isolated Flink dependencies
pip install -r flink/requirements.txt

# 3. Run all Flink tests with PyFlink runtime enabled
pytest flink/tests/ -v

# 4. Clean up
deactivate
```

---

## 4. Validating Member 1 (Apache Kafka) Live Cluster & Multi-Process Tests

To start the 3-node KRaft broker cluster and run the end-to-end integration test:

```powershell
# 1. Format and start KRaft brokers (in separate terminal windows):
kafka-storage.bat format -t <CLUSTER_UUID> -c cluster/server-1.properties
kafka-server-start.bat cluster/server-1.properties
kafka-server-start.bat cluster/server-2.properties
kafka-server-start.bat cluster/server-3.properties

# 2. In isolated Kafka venv:
python -m venv .venv-kafka
.\.venv-kafka\Scripts\Activate.ps1
pip install -r kafka/requirements.txt

# 3. Execute Kafka connection and delivery tests:
pytest kafka/tests/ -v

# 4. Run full multi-process end-to-end integration test:
python tests/integration/run_full_e2e_integration_test.py

# 5. Run live scale benchmark optimizations:
python tests/integration/test_scale_benchmark_optimizations.py

deactivate
```
