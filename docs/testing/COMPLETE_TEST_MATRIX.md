# COMPLETE TEST MATRIX & VERIFICATION INVENTORY

**Repository:** `adaptive-upi-fraud-detection`  
**Current Branch:** `Upto_Phase-4` (`c78494e`)  
**Audit Date:** October 2026  
**Auditor Mode:** Read-Only Verification  

---

## 1. Authoritative Test Matrix

| Layer / Track | Test File / Directory | Tests | Passed | Skipped | Failed | Purpose & Invariant Verified | Runtime Dependency |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Phase 1 (E1)** | `tests/test_phase1.py` | 10 | 10 | 0 | 0 | E1 baseline integrity, splits, determinism, zero feature leakage | Python 3 + LightGBM + scikit-learn |
| **Phase 2 (E2b)** | `phase2_gru/tests/` | 80 | 80 | 0 | 0 | Temporal GRU sequences, causal ordering, LT3–LT9 leakage invariants | Python 3 + PyTorch |
| **Phase 3 (E3)** | `experiments/E3_hybrid/tests/` | 20 | 20 | 0 | 0 | E3 leakage protections, common test IDs, E1/E2b artifact hashes | Python 3 + PyTorch + LightGBM |
| **Phase 4 (BAF)** | `experiments/phase4_drift_adaptation/tests/integrity_tests.py` | 52 | 52 | 0 | 0 | Monthly temporal splits, PSI triggers, adaptive v1/v2/v3, M7 bootstrap | Python 3 + scikit-learn |
| **Member 1 (Kafka)**| `kafka/tests/test_spark_flink_integration.py` | 3 | 3 | 0 | 0 | Normalization schemas, credit card schemas, Flink window contract | Pure Python |
| **Member 2 (Spark)**| `spark/tests/` | 5 | 3 | 2 | 0 | Event-time sequence order, watermark boundaries, amount ratios | Python 3 (PySpark skipped via `importorskip`) |
| **Member 2 (Flink)**| `flink/tests/` | 12 | 5 | 7 | 0 | Batch cleaner, dataset joiner, amount bucketing, transaction schemas | Python 3 (PyFlink skipped via `importorskip`) |
| **Total Discovered**| **Repository Full Suite** | **182** | **173** | **9** | **0** | **173 / 173 applicable tests passed (100% of applicable tests; 9 environmental skips)** | **Valid & Clean** |

---

## 2. Explanation of the 9 Environmental Skips

1. **Spark Skips (2 tests):**
   - `spark/tests/test_spark_batch.py`: Requires active distributed PySpark JVM engine to test DataFrame transformations.
   - `spark/tests/test_previous_transaction_stateful.py`: Requires active PySpark driver to execute `mapGroupsWithState`.
   - **Reason:** PySpark is isolated in `spark/requirements.txt` to prevent contaminating the core research conda environment. Tests cleanly skipped via `pytest.importorskip("pyspark")`.
2. **Flink Skips (7 tests):**
   - `flink/tests/test_window_features.py`: Requires active PyFlink runtime to evaluate streaming window triggers.
   - `flink/tests/test_watermark.py`, `test_state_manager.py`, `hello_flink.py`, `map_test.py`, `java_streaming_test.py`: Require PyFlink engine and JAR bindings.
   - **Reason:** PyFlink is isolated in `flink/requirements.txt`. Tests cleanly skipped via `pytest.importorskip("pyflink")`.

---

## 3. Standalone Benchmark & Live Integration Scripts

The repository also includes standalone live-process test scripts in `kafka/producer/`, `kafka/tests/test_kafka_connection.py`, and `tests/integration/`:
- `kafka/tests/test_kafka_connection.py`: Connects to `localhost:9092` via `KafkaAdminClient`.
- `tests/integration/run_full_e2e_integration_test.py`: 14-step integration test orchestrating live subprocesses.
- `tests/integration/test_scale_benchmark_optimizations.py`: Scale benchmark script.
- **Classification:** Standalone infrastructure scripts that require an active multi-node Kafka broker and virtual environment with `kafka-python-ng`. They are not unit tests and skip when running without cluster infrastructure.
