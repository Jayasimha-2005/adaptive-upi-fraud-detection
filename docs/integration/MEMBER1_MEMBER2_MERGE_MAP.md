# File-by-File Merge Map: Member 1 & Member 2 Integration

**Source:** `https://github.com/HarikaReddy440/Kafka-spark-flink` (HEAD `f74042f`)  
**Target:** `https://github.com/Jayasimha-2005/adaptive-upi-fraud-detection` (Branch `integration/member1-member2`)  
**Status:** Audit & Merge Map Specification  
**Classification Actions:** `KEEP`, `COPY`, `MOVE`, `MODIFY`, `MERGE`, `EXCLUDE`, `REPLACE`, `REVIEW`

---

## 1. Executive Summary & Mapping Principles

1. **Strict Research Immutability:** All existing target research files in `src/`, `experiments/`, `phase2_gru/`, `reports/`, and `configs/` are classified as `KEEP`. No source file is permitted to overwrite them.
2. **Namespace Unpacking:**
   - Member 1 files from `kafka/` and `cluster/` are mapped directly to target `kafka/` and `cluster/`.
   - Member 2 Spark files from `spark/spark/` are mapped to top-level target `spark/`.
   - Member 2 Flink files from `spark/flink/` are mapped to top-level target `flink/`.
3. **Exclusion of Duplicate Research Trees:** The entire nested duplicate research tree inside `spark/` (`spark/experiments/`, `spark/phase2_gru/`, `spark/dataset_analysis/`, `spark/analysis_scripts/`, `spark/src/`, `spark/tests/test_phase1.py`) is classified as `EXCLUDE`.
4. **Exclusion of Transient / Backup Artifacts:** All `__pycache__`, `.pyc`, `.vscode/settings.json`, and backup files (`*.backup`, `*.before_sink`) are classified as `EXCLUDE`.

---

## 2. File-by-File Merge Mapping Table

### A. Member 1: Apache Kafka Layer (`kafka/` & `cluster/`)

| Source Path | Target Path | Action | Reason | Conflict? | Dependencies | Validation Required |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `cluster/server-[1-3].properties` | `cluster/server-[1-3].properties` | **COPY** | 3-node KRaft multi-broker Kafka configuration | None (new dir) | Apache Kafka 3.x | Config syntax check |
| `kafka/config/config.py` | `kafka/config/config.py` | **COPY** | Kafka connection & topic defaults | None (new dir) | None | Python import test |
| `kafka/transaction_generator/dataset_loader.py` | `kafka/transaction_generator/dataset_loader.py` | **COPY** | Universal CSV parser and normalizer | None | `pathlib`, `csv` | Normalization unit test |
| `kafka/transaction_generator/transaction_generator.py` | `kafka/transaction_generator/transaction_generator.py` | **COPY** | Synthetic event generator | None | `random`, `uuid` | Generator smoke test |
| `kafka/producer/ieee_cis_replay_producer.py` | `kafka/producer/ieee_cis_replay_producer.py` | **COPY** | Replays IEEE-CIS transactions with key affinity | None | `kafka-python` | Producer dry-run test |
| `kafka/producer/high_throughput_producer_benchmark.py` | `kafka/producer/high_throughput_producer_benchmark.py` | **COPY** | Benchmarks producer throughput scaling | None | `kafka-python`, `multiprocessing` | Smoke run |
| `kafka/producer/batching_compression_benchmark.py` | `kafka/producer/batching_compression_benchmark.py` | **COPY** | Benchmarks compression codecs & batch sizes | None | `kafka-python`, `lz4`, `snappy` | Smoke run |
| `kafka/producer/acks_test_producer.py` | `kafka/producer/acks_test_producer.py` | **COPY** | Tests delivery durability (`acks=0,1,all`) | None | `kafka-python` | Durability verification |
| `kafka/producer/ordering_key_producer.py` | `kafka/producer/ordering_key_producer.py` | **COPY** | Verifies per-card partition affinity | None | `kafka-python` | Ordering check |
| `kafka/producer/idempotence_test_producer.py` | `kafka/producer/idempotence_test_producer.py` | **COPY** | Tests exactly-once producer idempotence | None | `kafka-python` | Idempotence test |
| `kafka/producer/rebalance_producer.py` | `kafka/producer/rebalance_producer.py` | **COPY** | Produces load during consumer rebalancing | None | `kafka-python` | Rebalance test |
| `kafka/producer/retry_behavior_producer.py` | `kafka/producer/retry_behavior_producer.py` | **COPY** | Tests network retry backoff | None | `kafka-python` | Retry test |
| `kafka/producer/producer.py` | `kafka/producer/producer.py` | **COPY** | Standard baseline Kafka producer | None | `kafka-python` | Producer test |
| `kafka/consumer/fraud_detection_consumer.py` | `kafka/consumer/fraud_detection_consumer.py` | **COPY** | Real-time rule-based fraud detection consumer | None | `kafka-python`, `json` | Rule engine test |
| `kafka/consumer/spark_streaming_consumer.py` | `kafka/consumer/spark_streaming_consumer.py` | **MODIFY** | Spark Structured Streaming consumer adapter (clarify heuristic scoring vs canonical E1) | None | `kafka-python`, optional `pyspark` | Standalone streaming test |
| `kafka/consumer/flink_cep_consumer.py` | `kafka/consumer/flink_cep_consumer.py` | **COPY** | Flink CEP sliding velocity window consumer | None | `kafka-python`, `deque` | CEP window test |
| `kafka/consumer/multiprocess_parallel_consumer_benchmark.py` | `kafka/consumer/multiprocess_parallel_consumer_benchmark.py` | **COPY** | Consumer group scaling across partitions | None | `kafka-python`, `multiprocessing` | Benchmark test |
| `kafka/consumer/at_least_once_consumer.py` | `kafka/consumer/at_least_once_consumer.py` | **COPY** | Manual offset commits and crash recovery | None | `kafka-python` | Offset test |
| `kafka/consumer/recovery_consumer.py` | `kafka/consumer/recovery_consumer.py` | **COPY** | Rebalance fault recovery consumer | None | `kafka-python` | Recovery test |
| `kafka/consumer/consumer.py` | `kafka/consumer/consumer.py` | **COPY** | Standard baseline Kafka consumer | None | `kafka-python` | Consumer test |
| `kafka/consumer/ieee_cis_consumer.py` | `kafka/consumer/ieee_cis_consumer.py` | **COPY** | IEEE-CIS specific topic consumer | None | `kafka-python` | Consumer test |
| `kafka/tests/test_kafka_connection.py` | `kafka/tests/test_kafka_connection.py` | **COPY** | Broker ping & connection unit test | None | `kafka-python` | Unit test execution |
| `kafka/tests/test_spark_flink_integration.py` | `kafka/tests/test_spark_flink_integration.py` | **COPY** | Normalization & CEP contract test | None | `unittest` | Unit test execution (3 tests) |
| `kafka/run_member1_dataset_experiments.py` | `kafka/run_member1_dataset_experiments.py` | **COPY** | Automated benchmark runner for Member 1 | None | `kafka-python` | Script validation |
| `kafka/README.md` | `kafka/README.md` | **MODIFY** | Member 1 documentation; sanitize machine-specific paths | None | Markdown | Link verification |
| `kafka/MEMBER1_RESEARCH_REPORT.md` | `kafka/reports/MEMBER1_RESEARCH_REPORT.md` | **MOVE** | Member 1 research findings report | None | Markdown | Report review |
| `member1_*_results.csv` (4 files) | `kafka/reports/benchmarks/member1_*_results.csv` | **MOVE** | Validated experimental benchmark CSV metrics | None | CSV | Data integrity check |
| `member1_results_summary.py` | `kafka/reports/benchmarks/member1_results_summary.py` | **MOVE** | Benchmark results aggregator | None | Python | Execution test |
| `MEMBER1_FINAL_DELIVERABLES.md` | `kafka/reports/MEMBER1_FINAL_DELIVERABLES.md` | **MODIFY** | Member 1 comprehensive deliverables; fix local paths | None | Markdown | Path sanitization |
| `MEMBER1_FINAL_DELIVERABLES.pdf` | `kafka/reports/MEMBER1_FINAL_DELIVERABLES.pdf` | **COPY** | PDF export of Member 1 deliverables | None | Binary PDF | Integrity check |

---

### B. Member 2: Apache Spark Layer (from `spark/spark/`)

| Source Path | Target Path | Action | Reason | Conflict? | Dependencies | Validation Required |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `spark/spark/__init__.py` | `spark/__init__.py` | **COPY** | Spark module initialization | None | None | Import test |
| `spark/spark/configs/spark_config.yaml` | `spark/configs/spark_config.yaml` | **COPY** | Central Spark configuration | None | `pyyaml` | YAML parsing test |
| `spark/spark/requirements.txt` | `spark/requirements.txt` | **COPY** | Component-specific Spark requirements | None | None | Dependency review |
| `spark/spark/batch/historical_pipeline.py` | `spark/batch/historical_pipeline.py` | **MODIFY** | PySpark batch cleaning and window feature engineering; make dataset paths portable | None | `pyspark` | Batch import & dry-run test |
| `spark/spark/benchmarks/convert_to_parquet.py` | `spark/benchmarks/convert_to_parquet.py` | **COPY** | CSV to Parquet conversion utility | None | `pyspark` | Script verification |
| `spark/spark/benchmarks/run_benchmarks.py` | `spark/benchmarks/run_benchmarks.py` | **COPY** | 3-trial CSV vs Parquet speedup benchmark | None | `pyspark` | Benchmark dry-run |
| `spark/spark/benchmarks/WSL_BENCHMARK_RESULTS.md` | `spark/reports/WSL_BENCHMARK_RESULTS.md` | **MOVE** | Spark benchmark findings documentation | None | Markdown | Documentation check |
| `spark/spark/features/realtime_features.py` | `spark/features/realtime_features.py` | **COPY** | Reference implementation of 5m/10m window features | None | Pure Python | Unit test execution |
| `spark/spark/features/feature_definitions.py` | `spark/features/feature_definitions.py` | **COPY** | Canonical streaming feature lists | None | Pure Python | Import test |
| `spark/spark/kafka_connector/consumer_sink.py` | `spark/kafka_connector/consumer_sink.py` | **COPY** | Kafka consumer connector for Spark | None | `pyspark` | Import test |
| `spark/spark/kafka_connector/producer_simulator.py` | `spark/kafka_connector/producer_simulator.py` | **COPY** | Producer simulator for Spark tests | None | `kafka-python` | Import test |
| `spark/spark/streaming/stream_processor.py` | `spark/streaming/stream_processor.py` | **COPY** | PySpark Structured Streaming consumer | None | `pyspark` | Streaming test |
| `spark/spark/streaming/state_manager.py` | `spark/streaming/state_manager.py` | **COPY** | Spark streaming state management | None | `pyspark` | State test |
| `spark/spark/experiments/batch_vs_streaming.py` | `spark/experiments/batch_vs_streaming.py` | **COPY** | Architectural comparison benchmark | None | `pyspark` | Script test |
| `spark/spark/experiments/event_time_experiment.py` | `spark/experiments/event_time_experiment.py` | **COPY** | Watermark and event-time experiment | None | `pyspark` | Script test |
| `spark/spark/reports/SPARK_RESEARCH_REPORT.md` | `spark/reports/SPARK_RESEARCH_REPORT.md` | **MODIFY** | Member 2 Spark research report; sanitize local paths | None | Markdown | Link verification |
| `spark/spark/tests/test_spark_batch.py` | `spark/tests/test_spark_batch.py` | **COPY** | PySpark batch pipeline import & function test | None | `pytest`, `pyspark` | Pytest run |
| `spark/spark/tests/test_spark_streaming.py` | `spark/tests/test_spark_streaming.py` | **COPY** | Pure Python streaming feature test | None | `pytest` | Pytest run (all pass) |
| `spark/spark/tests/test_event_time.py` | `spark/tests/test_event_time.py` | **COPY** | Event-time ordering test | None | `pytest` | Pytest run |
| `spark/spark/tests/test_previous_transaction_stateful.py` | `spark/tests/test_previous_transaction_stateful.py` | **COPY** | Stateful previous transaction test | None | `pytest` | Pytest run |

---

### C. Member 2: Apache Flink Layer (from `spark/flink/`)

| Source Path | Target Path | Action | Reason | Conflict? | Dependencies | Validation Required |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `spark/flink/__init__.py` | `flink/__init__.py` | **COPY** | Flink module initialization | None | None | Import test |
| `spark/flink/README.md` | `flink/README.md` | **MODIFY** | Flink setup & execution guide; fix local paths | None | Markdown | Documentation check |
| `spark/flink/requirements.txt` | `flink/requirements.txt` | **COPY** | Component-specific Flink requirements | None | None | Dependency review |
| `spark/flink/configs/flink_config.yaml` | `flink/configs/flink_config.yaml` | **COPY** | Flink job configuration | None | `pyyaml` | YAML parsing test |
| `spark/flink/configs/kafka_config.yaml` | `flink/configs/kafka_config.yaml` | **COPY** | Flink Kafka connector configuration | None | `pyyaml` | YAML parsing test |
| `spark/flink/jars/*.jar` (2 jars) | `flink/jars/*.jar` | **COPY** | Bundled Kafka connector JARs for Flink (under 10MB) | None | Java / Flink | File integrity & size check |
| `spark/flink/streaming/stream_processor_flink22_test.py` | `flink/streaming/stream_processor.py` | **COPY** | Canonical PyFlink streaming processor | None | PyFlink, Java 17 | Syntax & dry run check |
| `spark/flink/streaming/kafka_source.py` | `flink/streaming/kafka_source.py` | **COPY** | Kafka source builder for Flink | None | PyFlink | Import test |
| `spark/flink/streaming/watermark.py` | `flink/streaming/watermark.py` | **COPY** | Bounded-out-of-orderness watermark strategy | None | PyFlink | Import test |
| `spark/flink/streaming/transaction_schema.py` | `flink/streaming/transaction_schema.py` | **COPY** | Flink streaming schema definitions | None | PyFlink | Import test |
| `spark/flink/streaming/window_features.py` | `flink/streaming/window_features.py` | **COPY** | Window feature calculation | None | Pure Python | Unit test execution |
| `spark/flink/features/velocity_features.py` | `flink/features/velocity_features.py` | **COPY** | Stateful `UserVelocityState` deque | None | Pure Python | Unit test execution |
| `spark/flink/features/aggregation_features.py` | `flink/features/aggregation_features.py` | **COPY** | Amount and count aggregation logic | None | Pure Python | Unit test execution |
| `spark/flink/features/transaction_features.py` | `flink/features/transaction_features.py` | **COPY** | Amount bucket categorization | None | Pure Python | Unit test execution |
| `spark/flink/batch/historical_pipeline.py` | `flink/batch/historical_pipeline.py` | **MODIFY** | Flink batch pipeline; make dataset paths portable | None | Pandas / PyArrow | Path refactoring test |
| `spark/flink/batch/data_loader.py` | `flink/batch/data_loader.py` | **COPY** | Chunked dataset loader | None | Pandas | Import test |
| `spark/flink/batch/data_cleaner.py` | `flink/batch/data_cleaner.py` | **COPY** | Transaction cleaning utility | None | Pandas | Import test |
| `spark/flink/batch/dataset_joiner.py` | `flink/batch/dataset_joiner.py` | **COPY** | Transaction and identity joiner | None | Pandas | Import test |
| `spark/flink/batch/feature_engineering.py` | `flink/batch/feature_engineering.py` | **COPY** | Batch feature engineering utility | None | Pandas | Import test |
| `spark/flink/batch/output_writer.py` | `flink/batch/output_writer.py` | **COPY** | Parquet output writer | None | PyArrow | Import test |
| `spark/flink/benchmarks/streaming_benchmark_producer.py` | `flink/benchmarks/streaming_benchmark_producer.py` | **COPY** | Rate-controlled benchmark producer | None | `kafka-python` | Benchmark smoke test |
| `spark/flink/benchmarks/flink_benchmark.py` | `flink/benchmarks/flink_benchmark.py` | **COPY** | Flink throughput benchmark runner | None | PyFlink | Benchmark check |
| `spark/flink/reports/FLINK_RESEARCH_REPORT.md` | `flink/reports/FLINK_RESEARCH_REPORT.md` | **COPY** | Member 2 Flink research report | None | Markdown | Report check |
| `spark/flink/reports/FLINK_ARCHITECTURE.md` | `flink/reports/FLINK_ARCHITECTURE.md` | **COPY** | Flink streaming architecture design | None | Markdown | Design check |
| `spark/flink/tests/*.py` (13 test files) | `flink/tests/*.py` | **COPY** | Flink feature, window, state, and pipeline unit tests | None | `pytest` | Pytest run (pure Python tests pass) |

---

### D. Top-Level Integration Scripts & Synthesis Reports

| Source Path | Target Path | Action | Reason | Conflict? | Dependencies | Validation Required |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `run_kafka_spark_flink_pipeline.py` | `run_kafka_spark_flink_pipeline.py` | **MODIFY** | End-to-end multi-process pipeline runner (Producer + Spark + Flink); adapt import paths to new top-level dirs | None | `multiprocessing`, `kafka-python` | Smoke run `--records 10` |
| `run_full_e2e_integration_test.py` | `tests/integration/run_full_e2e_integration_test.py` | **MODIFY** | 14-step integration test suite; place under `tests/integration/` | None | `kafka-python` | Dry-run / mock broker test |
| `test_scale_benchmark_optimizations.py` | `tests/integration/test_scale_benchmark_optimizations.py` | **MODIFY** | Scaled batch test suite; place under `tests/integration/` | None | `kafka-python` | Execution test |
| `CODEBASE_INPUT_OUTPUT.md` | `docs/integration/CODEBASE_INPUT_OUTPUT.md` | **COPY** | Complete input/output specification | None | Markdown | Documentation check |
| `CODEBASE_INPUT_OUTPUT.pdf` | `docs/integration/CODEBASE_INPUT_OUTPUT.pdf` | **COPY** | PDF export of input/output specification | None | Binary PDF | Integrity check |
| `COMPLETE_CODEBASE_GUIDE.md` | `docs/integration/COMPLETE_CODEBASE_GUIDE.md` | **COPY** | Architecture and implementation guide | None | Markdown | Documentation check |
| `COMPLETE_CODEBASE_GUIDE.pdf` | `docs/integration/COMPLETE_CODEBASE_GUIDE.pdf` | **COPY** | PDF export of codebase guide | None | Binary PDF | Integrity check |
| `spark/SPARK_AND_FLINK_IMPLEMENTATION_AND_RESULTS.md` | `docs/integration/SPARK_AND_FLINK_IMPLEMENTATION_AND_RESULTS.md` | **MODIFY** | Synthesis report for Spark & Flink; sanitize local paths | None | Markdown | Path sanitization |
| `spark/SPARK_AND_FLINK_IMPLEMENTATION_AND_RESULTS.pdf` | `docs/integration/SPARK_AND_FLINK_IMPLEMENTATION_AND_RESULTS.pdf` | **COPY** | PDF export of Spark & Flink report | None | Binary PDF | Integrity check |
| `spark/COMPREHENSIVE_SYSTEM_AND_BENCHMARK_REPORT.md` | `docs/integration/COMPREHENSIVE_SYSTEM_AND_BENCHMARK_REPORT.md` | **MODIFY** | System benchmark synthesis report; sanitize local paths | None | Markdown | Path sanitization |
| `spark/INPUT_OUTPUT.md` & `.pdf` | `docs/integration/SPARK_FLINK_INPUT_OUTPUT.md` & `.pdf` | **MOVE** | Spark & Flink input/output report | None | Markdown / PDF | Reference check |

---

### E. Explicit Exclusions (`EXCLUDE`)

| Source Path | Action | Exact Rationale for Exclusion |
| :--- | :--- | :--- |
| `spark/experiments/` (entire tree) | **EXCLUDE** | Duplicate research tree from earlier clone. Must not overwrite canonical Phase 1-4 experiment artifacts. |
| `spark/phase2_gru/` (entire tree) | **EXCLUDE** | Duplicate of Phase 2 GRU research. Target repo already contains authoritative, tested Phase 2 code. |
| `spark/dataset_analysis/` (entire tree) | **EXCLUDE** | Duplicate EDA files. Target repo already contains authoritative dataset analysis. |
| `spark/analysis_scripts/` (entire tree) | **EXCLUDE** | Duplicate scripts. Target repo already contains authoritative scripts. |
| `spark/src/` (entire tree) | **EXCLUDE** | Duplicate core ML source code. Target repo has canonical `src/`. |
| `spark/tests/` (root duplicates) | **EXCLUDE** | Duplicate of `test_phase1.py`. Target repo already has clean `tests/test_phase1.py`. |
| `spark/download_dataset.py` | **EXCLUDE** | Duplicate of root `download_dataset.py`. |
| `spark/configs/` (root duplicates) | **EXCLUDE** | Duplicate of `configs/phase1_lightgbm.yaml`. |
| `spark/reports/` (root duplicates) | **EXCLUDE** | Duplicate research completion reports and PDFs. Target repo reports are authoritative. |
| `spark/requirements.txt` | **EXCLUDE** | Exact duplicate of root `requirements.txt`. |
| `spark/README.md` | **EXCLUDE** | Outdated duplicate of root README. |
| `spark/.gitignore` | **EXCLUDE** | Nested gitignore from duplicate clone. |
| `__pycache__/` & `*.pyc` (all locations) | **EXCLUDE** | Compiled Python bytecode. Must never be committed to Git. |
| `.vscode/settings.json` | **EXCLUDE** | Local developer IDE configuration. |
| `spark/flink/streaming/*.backup` | **EXCLUDE** | Developer scratch backup files. |
| `spark/flink/streaming/*.before_*` | **EXCLUDE** | Intermediate development backup files. |
| `spark/spark/benchmarks/data/` | **EXCLUDE** | Generated parquet files and marker directories. |
