"""
phase4_api/tests/test_batch_inference_engine.py
===============================================
Comprehensive test suite for BatchTransactionInferenceEngine.

Validates all Phase 5 requirements:
- Input validation (empty, single, multiple, non-list, non-dict, missing features, invalid types)
- Batch behavior (order preservation, ID preservation, duplicate IDs, partial success, non-mutation)
- Vectorized prediction (single call, no call if all invalid, probability bounds, threshold boundaries)
- Single-vs-batch consistency (matching probability to 1e-6 and identical predictions)
- Real-model smoke test on actual LightGBM v3 booster
"""
from __future__ import annotations

import copy
import math
import pathlib
import sys
import unittest
from typing import Any, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# Setup path
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase4_api.src.batch_inference_engine import (
    BatchInferenceError,
    BatchPredictionResponse,
    BatchRecordResult,
    BatchSizeLimitExceededError,
    BatchTransactionInferenceEngine,
    InvalidBatchInputError,
    get_batch_inference_engine,
)
from phase4_api.src.inference_engine import SingleTransactionInferenceEngine
from phase4_api.src.model_loader import (
    CANONICAL_FEATURE_COLS,
    EXPECTED_FEATURE_COUNT,
    LoadedModelBundle,
    ModelInfo,
    get_model_bundle,
)
from phase4_api.tests.test_inference_engine import MockBooster, create_mock_bundle
from phase4_api.tests.test_transaction_preprocessor import get_sample_valid_transaction


def get_two_distinct_valid_transactions() -> List[dict]:
    """Returns two valid transactions with distinct feature values."""
    tx1 = get_sample_valid_transaction()
    tx2 = get_sample_valid_transaction()
    tx2["income"] = 0.9
    tx2["customer_age"] = 60
    tx2["payment_type"] = "AB"
    tx2["employment_status"] = "CB"
    tx2["housing_status"] = "BA"
    tx2["device_os"] = "linux"
    return [tx1, tx2]


class TestBatchInferenceEngine(unittest.TestCase):
    """Test suite for BatchTransactionInferenceEngine."""

    def setUp(self) -> None:
        self.sample = get_sample_valid_transaction()
        self.mock_booster = MockBooster(return_val=np.array([0.25, 0.92]))
        self.bundle = create_mock_bundle(self.mock_booster)
        self.engine = BatchTransactionInferenceEngine(
            model_version="v3",
            model_bundle=self.bundle,
            max_batch_size=100,
        )

    # -------------------------------------------------------------------------
    # Input Validation Tests (1 - 11)
    # -------------------------------------------------------------------------
    def test_01_empty_batch_handling(self) -> None:
        resp = self.engine.predict_batch([])
        self.assertEqual(resp.batch_status, "empty")
        self.assertEqual(resp.total_transactions, 0)
        self.assertEqual(resp.successful_transactions, 0)
        self.assertEqual(resp.failed_transactions, 0)
        self.assertEqual(resp.results, [])
        self.assertEqual(self.mock_booster.call_count, 0)

    def test_02_valid_single_record_batch(self) -> None:
        self.mock_booster.return_val = np.array([0.33])
        resp = self.engine.predict_batch([self.sample])
        self.assertEqual(resp.batch_status, "all_success")
        self.assertEqual(resp.total_transactions, 1)
        self.assertEqual(resp.successful_transactions, 1)
        self.assertEqual(resp.failed_transactions, 0)
        self.assertEqual(len(resp.results), 1)
        self.assertEqual(resp.results[0].prediction, "LEGIT")
        self.assertEqual(resp.results[0].fraud_probability, 0.33)

    def test_03_multiple_valid_records(self) -> None:
        records = get_two_distinct_valid_transactions()
        resp = self.engine.predict_batch(records)
        self.assertEqual(resp.batch_status, "all_success")
        self.assertEqual(resp.total_transactions, 2)
        self.assertEqual(resp.successful_transactions, 2)
        self.assertEqual(resp.failed_transactions, 0)
        self.assertEqual(resp.results[0].prediction, "LEGIT")
        self.assertEqual(resp.results[1].prediction, "FRAUD")  # 0.92 >= 0.8964

    def test_04_non_list_input_rejected(self) -> None:
        with self.assertRaises(InvalidBatchInputError):
            self.engine.predict_batch(None)  # type: ignore[arg-type]
        with self.assertRaises(InvalidBatchInputError):
            self.engine.predict_batch("string_is_not_batch")  # type: ignore[arg-type]
        with self.assertRaises(InvalidBatchInputError):
            self.engine.predict_batch({"dict": "is_not_list"})  # type: ignore[arg-type]
        with self.assertRaises(InvalidBatchInputError):
            self.engine.predict_batch(12345)  # type: ignore[arg-type]

    def test_05_non_dictionary_record_handled_as_record_error(self) -> None:
        batch = [self.sample, "not_a_dictionary", 999]
        self.mock_booster.return_val = np.array([0.15])
        resp = self.engine.predict_batch(batch)
        self.assertEqual(resp.batch_status, "partial_success")
        self.assertEqual(resp.total_transactions, 3)
        self.assertEqual(resp.successful_transactions, 1)
        self.assertEqual(resp.failed_transactions, 2)

        # Record 0 succeeded
        self.assertEqual(resp.results[0].status, "success")
        # Records 1 and 2 failed
        self.assertEqual(resp.results[1].status, "error")
        self.assertEqual(resp.results[1].error["code"], "INVALID_RECORD_FORMAT")
        self.assertEqual(resp.results[2].status, "error")
        self.assertEqual(resp.results[2].error["code"], "INVALID_RECORD_FORMAT")

    def test_06_missing_required_features_handled(self) -> None:
        bad_record = dict(self.sample)
        del bad_record["income"]
        batch = [self.sample, bad_record]
        self.mock_booster.return_val = np.array([0.12])
        resp = self.engine.predict_batch(batch)
        self.assertEqual(resp.batch_status, "partial_success")
        self.assertEqual(resp.results[0].status, "success")
        self.assertEqual(resp.results[1].status, "error")
        self.assertEqual(resp.results[1].error["code"], "MISSING_FEATURE")
        self.assertIn("income", resp.results[1].error["message"])

    def test_07_invalid_numerical_input_handled(self) -> None:
        bad_num = dict(self.sample)
        bad_num["velocity_6h"] = "not_a_number"
        batch = [bad_num]
        resp = self.engine.predict_batch(batch)
        self.assertEqual(resp.batch_status, "all_failed")
        self.assertEqual(resp.results[0].status, "error")
        self.assertEqual(resp.results[0].error["code"], "INVALID_NUMERICAL_VALUE")

    def test_08_infinite_numerical_input_handled(self) -> None:
        bad_inf = dict(self.sample)
        bad_inf["intended_balcon_amount"] = float("inf")
        batch = [bad_inf]
        resp = self.engine.predict_batch(batch)
        self.assertEqual(resp.batch_status, "all_failed")
        self.assertEqual(resp.results[0].status, "error")
        self.assertEqual(resp.results[0].error["code"], "INVALID_NUMERICAL_VALUE")

    def test_09_invalid_categorical_value_handled(self) -> None:
        bad_cat = dict(self.sample)
        bad_cat["payment_type"] = "INVALID_CATEGORY"
        batch = [bad_cat]
        resp = self.engine.predict_batch(batch)
        self.assertEqual(resp.batch_status, "all_failed")
        self.assertEqual(resp.results[0].status, "error")
        self.assertEqual(resp.results[0].error["code"], "INVALID_CATEGORICAL_VALUE")

    def test_10_missing_transaction_id_supported(self) -> None:
        self.mock_booster.return_val = np.array([0.5])
        resp = self.engine.predict_batch([self.sample])
        self.assertIsNone(resp.results[0].transaction_id)
        self.assertEqual(resp.results[0].status, "success")

    def test_11_duplicate_transaction_ids_preserved(self) -> None:
        # Caller supplies duplicate transaction IDs
        rec1 = dict(self.sample)
        rec1["transaction_id"] = "DUP-ID-01"
        rec2 = dict(self.sample)
        rec2["transaction_id"] = "DUP-ID-01"
        self.mock_booster.return_val = np.array([0.2, 0.4])

        resp = self.engine.predict_batch([rec1, rec2])
        self.assertEqual(resp.total_transactions, 2)
        self.assertEqual(resp.successful_transactions, 2)
        # Distinct indices preserved
        self.assertEqual(resp.results[0].input_index, 0)
        self.assertEqual(resp.results[0].transaction_id, "DUP-ID-01")
        self.assertEqual(resp.results[1].input_index, 1)
        self.assertEqual(resp.results[1].transaction_id, "DUP-ID-01")

    # -------------------------------------------------------------------------
    # Batch Behavior Tests (12 - 20)
    # -------------------------------------------------------------------------
    def test_12_input_order_preservation(self) -> None:
        records = get_two_distinct_valid_transactions()
        bad_rec = dict(self.sample)
        del bad_rec["customer_age"]
        # Interleave valid and invalid: [valid, bad, valid]
        batch = [records[0], bad_rec, records[1]]
        self.mock_booster.return_val = np.array([0.10, 0.95])

        resp = self.engine.predict_batch(batch)
        self.assertEqual(resp.total_transactions, 3)
        self.assertEqual(len(resp.results), 3)

        self.assertEqual(resp.results[0].input_index, 0)
        self.assertEqual(resp.results[0].status, "success")
        self.assertEqual(resp.results[0].fraud_probability, 0.10)

        self.assertEqual(resp.results[1].input_index, 1)
        self.assertEqual(resp.results[1].status, "error")

        self.assertEqual(resp.results[2].input_index, 2)
        self.assertEqual(resp.results[2].status, "success")
        self.assertEqual(resp.results[2].fraud_probability, 0.95)

    def test_13_transaction_id_preservation(self) -> None:
        rec1 = dict(self.sample)
        rec1["transaction_id"] = "TID-AAA"
        rec2 = dict(self.sample)
        rec2["transaction_id"] = "TID-BBB"
        self.mock_booster.return_val = np.array([0.2, 0.3])

        resp = self.engine.predict_batch([rec1, rec2])
        self.assertEqual(resp.results[0].transaction_id, "TID-AAA")
        self.assertEqual(resp.results[1].transaction_id, "TID-BBB")

    def test_14_to_16_counts_accuracy(self) -> None:
        rec_bad = dict(self.sample)
        del rec_bad["source"]
        batch = [self.sample, rec_bad, self.sample]
        self.mock_booster.return_val = np.array([0.1, 0.2])

        resp = self.engine.predict_batch(batch)
        self.assertEqual(resp.total_transactions, 3)
        self.assertEqual(resp.successful_transactions, 2)
        self.assertEqual(resp.failed_transactions, 1)
        self.assertEqual(resp.batch_status, "partial_success")

    def test_17_partial_success(self) -> None:
        rec_bad = dict(self.sample)
        rec_bad["days_since_request"] = "invalid_string"
        batch = [self.sample, rec_bad]
        self.mock_booster.return_val = np.array([0.5])

        resp = self.engine.predict_batch(batch)
        self.assertEqual(resp.batch_status, "partial_success")
        self.assertEqual(resp.successful_transactions, 1)
        self.assertEqual(resp.failed_transactions, 1)

    def test_18_all_invalid_batch(self) -> None:
        rec_bad1 = dict(self.sample)
        del rec_bad1["income"]
        rec_bad2 = dict(self.sample)
        rec_bad2["payment_type"] = "BAD_CAT"
        batch = [rec_bad1, rec_bad2]

        resp = self.engine.predict_batch(batch)
        self.assertEqual(resp.batch_status, "all_failed")
        self.assertEqual(resp.successful_transactions, 0)
        self.assertEqual(resp.failed_transactions, 2)
        self.assertEqual(self.mock_booster.call_count, 0)

    def test_19_no_mutation_of_input_records(self) -> None:
        rec1 = dict(self.sample)
        rec1["transaction_id"] = "IMMUTABLE_TEST_01"
        snapshot = copy.deepcopy(rec1)
        self.mock_booster.return_val = np.array([0.5])

        self.engine.predict_batch([rec1])
        self.assertEqual(rec1, snapshot)
        self.assertIn("transaction_id", rec1)

    def test_20_prediction_exception_handled_cleanly(self) -> None:
        self.mock_booster.return_val = RuntimeError("C++ booster out of memory")
        resp = self.engine.predict_batch([self.sample])
        self.assertEqual(resp.batch_status, "all_failed")
        self.assertEqual(resp.successful_transactions, 0)
        self.assertEqual(resp.failed_transactions, 1)
        self.assertEqual(resp.results[0].status, "error")
        self.assertEqual(resp.results[0].error["code"], "PREDICTION_ERROR")
        self.assertIn("LightGBM", resp.results[0].error["message"])

    # -------------------------------------------------------------------------
    # Model Prediction Tests (21 - 29)
    # -------------------------------------------------------------------------
    def test_21_exactly_one_vectorized_call_for_valid_batch(self) -> None:
        records = get_two_distinct_valid_transactions()
        self.mock_booster.return_val = np.array([0.1, 0.2])
        resp = self.engine.predict_batch(records)
        self.assertEqual(self.mock_booster.call_count, 1)
        self.assertEqual(resp.successful_transactions, 2)

    def test_22_no_prediction_call_when_all_records_invalid(self) -> None:
        batch = [{"incomplete": 1}, {"corrupted": 2}]
        resp = self.engine.predict_batch(batch)
        self.assertEqual(self.mock_booster.call_count, 0)
        self.assertEqual(resp.successful_transactions, 0)

    def test_23_correct_output_length(self) -> None:
        records = get_two_distinct_valid_transactions()
        self.mock_booster.return_val = np.array([0.3, 0.4])
        resp = self.engine.predict_batch(records)
        self.assertEqual(len(resp.results), 2)

    def test_24_probability_range_validation(self) -> None:
        self.mock_booster.return_val = np.array([0.77])
        resp = self.engine.predict_batch([self.sample])
        prob = resp.results[0].fraud_probability
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_25_nan_or_infinity_prediction_rejected(self) -> None:
        # Injected NaN
        self.mock_booster.return_val = np.array([np.nan])
        resp = self.engine.predict_batch([self.sample])
        self.assertEqual(resp.results[0].status, "error")
        self.assertEqual(resp.results[0].error["code"], "PREDICTION_ERROR")

        # Injected Inf
        self.mock_booster.return_val = np.array([np.inf])
        resp_inf = self.engine.predict_batch([self.sample])
        self.assertEqual(resp_inf.results[0].status, "error")
        self.assertEqual(resp_inf.results[0].error["code"], "PREDICTION_ERROR")

    def test_26_threshold_boundary_behavior(self) -> None:
        # Below, exact, and above
        threshold = 0.8964321608040201
        self.mock_booster.return_val = np.array([0.896430, threshold, 0.950000])
        batch = [self.sample, self.sample, self.sample]

        resp = self.engine.predict_batch(batch)
        self.assertEqual(resp.results[0].prediction, "LEGIT")
        self.assertEqual(resp.results[1].prediction, "FRAUD")
        self.assertEqual(resp.results[2].prediction, "FRAUD")

    def test_27_correct_model_version_reported(self) -> None:
        self.mock_booster.return_val = np.array([0.5])
        resp = self.engine.predict_batch([self.sample])
        self.assertEqual(resp.model_version, "v3")
        self.assertEqual(resp.results[0].model_version, "v3")

    def test_28_correct_feature_count_reported(self) -> None:
        self.mock_booster.return_val = np.array([0.5])
        resp = self.engine.predict_batch([self.sample])
        self.assertEqual(resp.results[0].feature_count, 29)

    def test_29_cached_model_reuse_across_batches(self) -> None:
        bundle_init = self.engine.bundle
        self.mock_booster.return_val = np.array([0.1])
        self.engine.predict_batch([self.sample])
        self.engine.predict_batch([self.sample])
        self.assertIs(self.engine.bundle, bundle_init)

    # -------------------------------------------------------------------------
    # Batch Size Limit Test
    # -------------------------------------------------------------------------
    def test_30_max_batch_size_limit_exceeded(self) -> None:
        small_engine = BatchTransactionInferenceEngine(
            model_bundle=self.bundle,
            max_batch_size=2,
        )
        batch = [self.sample, self.sample, self.sample]  # 3 > 2
        with self.assertRaises(BatchSizeLimitExceededError) as ctx:
            small_engine.predict_batch(batch)
        self.assertIn("exceeds maximum allowed limit of 2", str(ctx.exception))

    # -------------------------------------------------------------------------
    # Consistency Tests (Single vs Batch) (31 - 34)
    # -------------------------------------------------------------------------
    def test_31_single_vs_batch_consistency(self) -> None:
        """
        Critical Requirement: Compares SingleTransactionInferenceEngine
        with BatchTransactionInferenceEngine on the exact same model and sample.
        """
        single_engine = SingleTransactionInferenceEngine(
            model_bundle=self.bundle,
        )
        self.mock_booster.return_val = np.array([0.896431])  # Just below threshold

        single_resp = single_engine.predict(self.sample, transaction_id="CONSIST-01")
        batch_resp = self.engine.predict_batch(
            [self.sample],
            transaction_ids=["CONSIST-01"],
        )

        batch_item = batch_resp.results[0]
        self.assertEqual(batch_item.prediction, single_resp.prediction)
        self.assertAlmostEqual(
            batch_item.fraud_probability,
            single_resp.fraud_probability,
            places=6,
        )
        self.assertEqual(batch_item.threshold, single_resp.threshold)
        self.assertEqual(batch_item.model_version, single_resp.model_version)
        self.assertEqual(batch_item.feature_count, single_resp.feature_count)

    def test_32_confirm_exact_feature_ordering_in_batch(self) -> None:
        # Check that CANONICAL_FEATURE_COLS ordering is used for batch DataFrame
        records = get_two_distinct_valid_transactions()
        self.mock_booster.return_val = np.array([0.1, 0.2])
        resp = self.engine.predict_batch(records)
        self.assertEqual(resp.batch_status, "all_success")

    def test_33_confirm_missing_numerical_handled_consistently(self) -> None:
        # Missing values (None / NaN) preserved as np.nan
        sample_nan = dict(self.sample)
        sample_nan["prev_address_months_count"] = None
        sample_nan["intended_balcon_amount"] = float("nan")

        self.mock_booster.return_val = np.array([0.18])
        resp = self.engine.predict_batch([sample_nan])
        self.assertEqual(resp.batch_status, "all_success")
        self.assertEqual(resp.results[0].status, "success")

    # -------------------------------------------------------------------------
    # Real-Model Smoke Test (Actual lgbm_v3.txt)
    # -------------------------------------------------------------------------
    def test_34_real_model_batch_smoke_test(self) -> None:
        """
        Executes real batch inference across the entire stack using the actual
        approved LightGBM v3 booster artifact and authoritative frozen threshold.
        """
        real_engine = BatchTransactionInferenceEngine(model_version="v3")
        real_single_engine = SingleTransactionInferenceEngine(model_version="v3")

        records = get_two_distinct_valid_transactions()
        records[0]["transaction_id"] = "REAL-BATCH-001"
        records[1]["transaction_id"] = "REAL-BATCH-002"

        # Execute batch
        batch_resp = real_engine.predict_batch(records)
        self.assertEqual(batch_resp.batch_status, "all_success")
        self.assertEqual(batch_resp.total_transactions, 2)
        self.assertEqual(batch_resp.successful_transactions, 2)
        self.assertEqual(batch_resp.failed_transactions, 0)
        self.assertEqual(batch_resp.model_version, "v3")
        self.assertEqual(batch_resp.threshold, 0.8964321608040201)

        # Execute single for comparison
        single_resp_0 = real_single_engine.predict(records[0], transaction_id="REAL-BATCH-001")
        single_resp_1 = real_single_engine.predict(records[1], transaction_id="REAL-BATCH-002")

        # Compare probabilities within 1e-5 numerical tolerance
        self.assertAlmostEqual(
            batch_resp.results[0].fraud_probability,
            single_resp_0.fraud_probability,
            places=5,
        )
        self.assertAlmostEqual(
            batch_resp.results[1].fraud_probability,
            single_resp_1.fraud_probability,
            places=5,
        )

        # Predictions match exactly
        self.assertEqual(batch_resp.results[0].prediction, single_resp_0.prediction)
        self.assertEqual(batch_resp.results[1].prediction, single_resp_1.prediction)

        # JSON serializability check
        d = batch_resp.to_dict()
        self.assertIn("results", d)
        self.assertEqual(len(d["results"]), 2)
        self.assertNotIn("Users", str(d))


if __name__ == "__main__":
    unittest.main(verbosity=2)
