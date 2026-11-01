"""
phase4_api/tests/test_benchmarks.py
===================================
Automated Unit & Integration Tests for Phase 8 Benchmarking and Parity Suite.

Part of Member 3 — API Serving Layer for Adaptive UPI / BAF Fraud Detection.
Phase 8: Performance Testing, Benchmarking & Parity Validation.

Tests cover:
  1. BenchmarkConfig validation and boundaries
  2. SystemMonitor environment inspection and fallback handling
  3. BAFDatasetLoader read-only ingestion, validation, ID separation, and synthetic fallback
  4. BenchmarkApiClient with mocked endpoints (single, batch, health, model info)
  5. SingleBenchmarkRunner warm-up exclusion and percentile calculations
  6. BatchBenchmarkRunner count/order/ID preservation and throughput metrics
  7. ConcurrencyBenchmarkRunner bounded threading and error resilience
  8. OfflineOnlineParityChecker precision tolerance, agreement, and mismatch diagnostics
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure workspace root is in sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase4_api.benchmarks.api_client import (
    BatchRequestResult,
    BenchmarkApiClient,
    SingleRequestResult,
)
from phase4_api.benchmarks.batch_benchmark import BatchBenchmarkRunner
from phase4_api.benchmarks.benchmark_config import BenchmarkConfig
from phase4_api.benchmarks.concurrency_benchmark import ConcurrencyBenchmarkRunner
from phase4_api.benchmarks.dataset_loader import (
    BAFDatasetLoader,
    DatasetValidationResult,
    ValidatedTransactionRecord,
)
from phase4_api.benchmarks.parity_checker import (
    OfflineOnlineParityChecker,
    ParitySummary,
)
from phase4_api.benchmarks.single_benchmark import SingleBenchmarkRunner
from phase4_api.benchmarks.system_monitor import SystemMonitor
from phase4_api.src.batch_inference_engine import DEFAULT_MAX_BATCH_SIZE
from phase4_api.src.inference_engine import PredictionResponse
from phase4_api.src.risk_classifier import ReasonCode


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_features() -> dict:
    """A single canonical 29-feature transaction dictionary."""
    return {
        "income": 0.5,
        "name_email_similarity": 0.85,
        "prev_address_months_count": 12.0,
        "current_address_months_count": 24.0,
        "customer_age": 35.0,
        "days_since_request": 0.05,
        "intended_balcon_amount": 50.0,
        "zip_count_4w": 500.0,
        "velocity_6h": 1200.0,
        "velocity_24h": 3400.0,
        "velocity_4w": 4500.0,
        "bank_branch_count_8w": 2.0,
        "date_of_birth_distinct_emails_4w": 1.0,
        "credit_risk_score": 140.0,
        "email_is_free": 1.0,
        "phone_home_valid": 1.0,
        "phone_mobile_valid": 1.0,
        "bank_months_count": 10.0,
        "has_other_cards": 0.0,
        "proposed_credit_limit": 1000.0,
        "foreign_request": 0.0,
        "session_length_in_minutes": 5.5,
        "keep_alive_session": 1.0,
        "device_distinct_emails_8w": 1.0,
        "payment_type": "AA",
        "employment_status": "CA",
        "housing_status": "BA",
        "source": "INTERNET",
        "device_os": "windows",
    }


@pytest.fixture
def sample_records(sample_features) -> list[ValidatedTransactionRecord]:
    """List of 5 validated transaction records."""
    records = []
    for i in range(5):
        feat = dict(sample_features)
        feat["income"] = round(0.2 + (i * 0.1), 2)
        records.append(
            ValidatedTransactionRecord(
                transaction_id=f"tx_test_{i:03d}",
                features=feat,
                ground_truth_label=0,
                is_synthetic=True,
                row_index=i,
            )
        )
    return records


# ---------------------------------------------------------------------------
# 1. BenchmarkConfig Tests
# ---------------------------------------------------------------------------
class TestBenchmarkConfig:
    def test_default_config_is_valid(self):
        cfg = BenchmarkConfig()
        assert cfg.limit == 100
        assert cfg.batch_sizes == [1, 10, 50, 100]
        assert cfg.concurrency_levels == [1, 5, 10, 20]
        assert cfg.tolerance == 1e-8

    def test_invalid_limit_raises_error(self):
        with pytest.raises(ValueError, match="limit must be positive"):
            BenchmarkConfig(limit=0)

    def test_invalid_base_url_raises_error(self):
        with pytest.raises(ValueError, match="base_url must start with http"):
            BenchmarkConfig(base_url="ftp://invalid")

    def test_batch_size_exceeding_limit_raises_error(self):
        with pytest.raises(ValueError, match="exceeds API maximum batch limit"):
            BenchmarkConfig(batch_sizes=[DEFAULT_MAX_BATCH_SIZE + 1])

    def test_negative_warmup_raises_error(self):
        with pytest.raises(ValueError, match="warmup_requests cannot be negative"):
            BenchmarkConfig(warmup_requests=-1)


# ---------------------------------------------------------------------------
# 2. SystemMonitor Tests
# ---------------------------------------------------------------------------
class TestSystemMonitor:
    def test_environment_info_structure(self):
        env = SystemMonitor.get_environment_info()
        assert "os" in env
        assert "python_version" in env
        assert "library_versions" in env
        assert "lightgbm" in env["library_versions"]
        assert "fastapi" in env["library_versions"]

    def test_capture_snapshot(self):
        monitor = SystemMonitor()
        snapshot = monitor.capture_snapshot()
        assert "monitoring_enabled" in snapshot


# ---------------------------------------------------------------------------
# 3. BAFDatasetLoader Tests
# ---------------------------------------------------------------------------
class TestBAFDatasetLoader:
    def test_generate_synthetic_dataset(self):
        loader = BAFDatasetLoader()
        res = loader.generate_synthetic_dataset(limit=10)
        assert res.is_synthetic is True
        assert res.valid_count == 10
        assert res.invalid_count == 0
        assert len(res.records) == 10
        for rec in res.records:
            assert len(rec.features) == 29
            assert rec.transaction_id.startswith("synth_tx_")

    def test_missing_dataset_path_without_synthetic_flag(self):
        loader = BAFDatasetLoader()
        with pytest.raises(ValueError, match="Dataset path was not provided"):
            loader.load_dataset(dataset_path=None, allow_synthetic_fallback=False)

    def test_nonexistent_dataset_file_raises_error(self):
        loader = BAFDatasetLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_dataset(dataset_path="nonexistent_baf_file.csv")

    def test_dataset_loader_with_synthetic_fallback(self):
        loader = BAFDatasetLoader()
        res = loader.load_dataset(dataset_path=None, limit=5, allow_synthetic_fallback=True)
        assert res.is_synthetic is True
        assert len(res.records) == 5

    def test_loader_detects_missing_features(self, tmp_path):
        bad_csv = tmp_path / "bad_baf.csv"
        bad_csv.write_text("fraud_bool,income,customer_age\n0,0.5,35\n")
        loader = BAFDatasetLoader()
        res = loader.load_dataset(str(bad_csv), limit=5)
        assert res.valid_count == 0
        assert len(res.missing_columns) > 0


# ---------------------------------------------------------------------------
# 4. BenchmarkApiClient Tests (Mocked)
# ---------------------------------------------------------------------------
class TestBenchmarkApiClient:
    @patch("httpx.Client.get")
    def test_check_health_healthy(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ready": True, "status": "healthy"}
        mock_get.return_value = mock_resp

        client = BenchmarkApiClient(base_url="http://127.0.0.1:8000")
        health = client.check_health()
        assert health["is_ready"] is True
        assert health["status_code"] == 200

    @patch("httpx.Client.get")
    def test_check_health_degraded(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_get.return_value = mock_resp

        client = BenchmarkApiClient(base_url="http://127.0.0.1:8000")
        health = client.check_health()
        assert health["is_ready"] is False
        assert health["status_code"] == 503

    @patch("httpx.Client.get")
    def test_get_model_info_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"model_version": "v3", "feature_count": 29}
        mock_get.return_value = mock_resp

        client = BenchmarkApiClient(base_url="http://127.0.0.1:8000")
        info = client.get_model_info()
        assert info["model_version"] == "v3"
        assert info["feature_count"] == 29

    @patch("httpx.Client.post")
    def test_predict_single_success(self, mock_post, sample_features):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "SUCCESS",
            "transaction_id": "tx_001",
            "prediction": "LEGIT",
            "fraud_probability": 0.052,
            "risk_level": "LOW",
            "execution_time_ms": 1.25,
            "reason_codes": [],
        }
        mock_post.return_value = mock_resp

        client = BenchmarkApiClient(base_url="http://127.0.0.1:8000")
        res = client.predict_single("tx_001", sample_features)
        assert res.is_success is True
        assert res.status_code == 200
        assert res.prediction == "LEGIT"
        assert res.fraud_probability == 0.052
        assert res.server_execution_time_ms == 1.25
        assert res.client_latency_ms >= 0.0

    @patch("httpx.Client.post")
    def test_predict_batch_success(self, mock_post, sample_features):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "batch_status": "all_success",
            "total_transactions": 2,
            "successful_transactions": 2,
            "failed_transactions": 0,
            "execution_time_ms": 2.5,
            "results": [
                {
                    "input_index": 0,
                    "transaction_id": "tx_0",
                    "status": "success",
                    "prediction": "LEGIT",
                    "fraud_probability": 0.02,
                    "risk_level": "LOW",
                },
                {
                    "input_index": 1,
                    "transaction_id": "tx_1",
                    "status": "success",
                    "prediction": "FRAUD",
                    "fraud_probability": 0.95,
                    "risk_level": "CRITICAL",
                },
            ],
        }
        mock_post.return_value = mock_resp

        client = BenchmarkApiClient(base_url="http://127.0.0.1:8000")
        res = client.predict_batch([dict(sample_features), dict(sample_features)])
        assert res.is_success is True
        assert res.total_transactions == 2
        assert res.successful_transactions == 2
        assert len(res.records) == 2
        assert res.records[0].prediction == "LEGIT"
        assert res.records[1].prediction == "FRAUD"


# ---------------------------------------------------------------------------
# 5. SingleBenchmarkRunner Tests
# ---------------------------------------------------------------------------
class TestSingleBenchmarkRunner:
    def test_single_benchmark_calculates_percentiles(self, sample_records):
        mock_client = MagicMock()
        mock_client.predict_single.return_value = SingleRequestResult(
            transaction_id="tx_test_000",
            status_code=200,
            is_success=True,
            client_latency_ms=10.0,
            server_execution_time_ms=1.5,
            prediction="LEGIT",
            fraud_probability=0.04,
            risk_level="LOW",
        )

        runner = SingleBenchmarkRunner(mock_client)
        summary, details = runner.run(sample_records, warmup_count=2)

        assert summary.total_requests == 5
        assert summary.warmup_requests == 2
        assert summary.measured_requests == 3
        assert summary.successful_requests == 3
        assert summary.failed_requests == 0
        assert summary.error_rate == 0.0
        assert summary.client_latency_p50_ms == 10.0
        assert summary.server_time_p50_ms == 1.5
        assert len(details) == 3


# ---------------------------------------------------------------------------
# 6. BatchBenchmarkRunner Tests
# ---------------------------------------------------------------------------
class TestBatchBenchmarkRunner:
    def test_batch_benchmark_verifies_parity(self, sample_records):
        mock_client = MagicMock()
        mock_client.predict_batch.side_effect = lambda chunk: BatchRequestResult(
            batch_size=len(chunk),
            status_code=200,
            is_success=True,
            client_latency_ms=15.0,
            server_execution_time_ms=3.0,
            batch_status="all_success",
            total_transactions=len(chunk),
            successful_transactions=len(chunk),
            failed_transactions=0,
            records=[
                MagicMock(
                    input_index=i,
                    transaction_id=rec["transaction_id"],
                    status="success",
                    prediction="LEGIT",
                    fraud_probability=0.03,
                    risk_level="LOW",
                )
                for i, rec in enumerate(chunk)
            ],
        )

        runner = BatchBenchmarkRunner(mock_client)
        summary, details = runner.run_for_batch_size(sample_records, batch_size=2)

        assert summary.batch_size == 2
        assert summary.total_transactions_submitted == 5
        assert summary.successful_transactions == 5
        assert summary.count_parity_verified is True
        assert summary.order_parity_verified is True
        assert summary.id_preservation_verified is True


# ---------------------------------------------------------------------------
# 7. ConcurrencyBenchmarkRunner Tests
# ---------------------------------------------------------------------------
class TestConcurrencyBenchmarkRunner:
    @patch.object(ConcurrencyBenchmarkRunner, "_worker_single")
    def test_concurrency_runner_bounded_execution(self, mock_worker, sample_records):
        mock_worker.return_value = SingleRequestResult(
            transaction_id="tx_test",
            status_code=200,
            is_success=True,
            client_latency_ms=5.0,
            server_execution_time_ms=1.0,
            prediction="LEGIT",
            fraud_probability=0.01,
            risk_level="LOW",
        )

        runner = ConcurrencyBenchmarkRunner(base_url="http://127.0.0.1:8000")
        metrics_list, details = runner.run_single_concurrency(sample_records, concurrency_levels=[2])

        assert len(metrics_list) == 1
        metric = metrics_list[0]
        assert metric.concurrency_level == 2
        assert metric.total_requests == 5
        assert metric.successful_requests == 5
        assert metric.error_rate == 0.0
        assert metric.client_latency_p50_ms == 5.0


# ---------------------------------------------------------------------------
# 8. OfflineOnlineParityChecker Tests
# ---------------------------------------------------------------------------
class TestOfflineOnlineParityChecker:
    def test_parity_perfect_agreement(self, sample_records):
        mock_client = MagicMock()
        mock_client.predict_single.return_value = SingleRequestResult(
            transaction_id="tx_001",
            status_code=200,
            is_success=True,
            client_latency_ms=8.0,
            server_execution_time_ms=1.0,
            prediction="LEGIT",
            fraud_probability=0.05000000,
            risk_level="LOW",
            reason_codes=["LOW_RISK_SCORE"],
        )

        mock_offline = MagicMock()
        mock_offline.predict.return_value = PredictionResponse(
            status="SUCCESS",
            transaction_id="tx_001",
            prediction="LEGIT",
            fraud_probability=0.05000000,
            threshold=0.8964321608040201,
            risk_level="LOW",
            model_version="v3",
            model_status="ACTIVE_LOADED",
            feature_count=29,
            reason_codes=[ReasonCode(code="LOW_RISK_SCORE", description="Normal score", category="RISK_LEVEL")],
            execution_time_ms=0.9,
        )

        checker = OfflineOnlineParityChecker(
            api_client=mock_client,
            offline_engine=mock_offline,
            tolerance=1e-8,
        )
        summary, details = checker.compare_records([sample_records[0]])

        assert summary.total_evaluated == 1
        assert summary.reconciled_count == 1
        assert summary.reconciliation_rate == 1.0
        assert summary.probability_parity_percentage == 100.0
        assert summary.decision_agreement_percentage == 100.0
        assert summary.max_absolute_probability_difference == 0.0
        assert len(summary.mismatched_transaction_ids) == 0

    def test_parity_detects_mismatch(self, sample_records):
        mock_client = MagicMock()
        mock_client.predict_single.return_value = SingleRequestResult(
            transaction_id="tx_001",
            status_code=200,
            is_success=True,
            client_latency_ms=8.0,
            server_execution_time_ms=1.0,
            prediction="FRAUD",
            fraud_probability=0.92,
            risk_level="CRITICAL",
            reason_codes=[],
        )

        mock_offline = MagicMock()
        mock_offline.predict.return_value = PredictionResponse(
            status="SUCCESS",
            transaction_id="tx_001",
            prediction="LEGIT",
            fraud_probability=0.05,
            threshold=0.8964321608040201,
            risk_level="LOW",
            model_version="v3",
            model_status="ACTIVE_LOADED",
            feature_count=29,
            reason_codes=[],
            execution_time_ms=0.9,
        )

        checker = OfflineOnlineParityChecker(
            api_client=mock_client,
            offline_engine=mock_offline,
            tolerance=1e-8,
        )
        summary, details = checker.compare_records([sample_records[0]])

        assert summary.reconciled_count == 0
        assert summary.reconciliation_rate == 0.0
        assert summary.probability_parity_percentage == 0.0
        assert summary.decision_agreement_percentage == 0.0
        assert summary.max_absolute_probability_difference == pytest.approx(0.87, abs=1e-4)
        assert len(summary.mismatched_transaction_ids) == 1
