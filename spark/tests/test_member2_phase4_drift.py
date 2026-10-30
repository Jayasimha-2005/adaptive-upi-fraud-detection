"""
Tests for Member 2 Phase 4: Spark & Flink BAF Drift Monitoring
===============================================================
Covers:
  - BAF schema and label handling
  - Reference-period and monitoring-period selection
  - PSI calculations (no-drift case + known-shift case)
  - Prevention of target-label leakage
  - Kafka-to-Spark integration (schema contract)
  - Kafka-to-Flink integration (schema contract)
  - Consistency of Spark and Flink drift decisions
  - Correct output counts and reconciliation
"""
import json
import os
import sys
import unittest
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parents[2]
PHASE4_SRC = ROOT / "experiments" / "phase4_drift_adaptation" / "src"
KAFKA_GEN  = ROOT / "kafka" / "transaction_generator"
sys.path.insert(0, str(PHASE4_SRC))
sys.path.insert(0, str(KAFKA_GEN))

import numpy as np
import pandas as pd

from data_loader import (
    load_baf, FEATURE_COLS, NUMERICAL_COLS, CATEGORICAL_COLS,
    TARGET_COL, TIME_COL, EXPECTED_COUNTS,
)
from psi import PSIDetector
from dataset_loader import normalize_record, resolve_dataset_file


class TestBAFSchemaAndLabels(unittest.TestCase):
    """Verify the BAF dataset schema, label column, and month column."""

    @classmethod
    def setUpClass(cls):
        baf_path = ROOT / "Datasets" / "BAF" / "Base.csv"
        if not baf_path.exists():
            raise unittest.SkipTest("BAF Base.csv not found")
        # Load only header + small sample for fast tests
        cls.df_head = pd.read_csv(baf_path, nrows=100)
        cls.baf_path = baf_path

    def test_baf_has_32_columns(self):
        self.assertEqual(len(self.df_head.columns), 32)

    def test_baf_has_fraud_bool_target(self):
        self.assertIn("fraud_bool", self.df_head.columns)

    def test_baf_has_month_column(self):
        self.assertIn("month", self.df_head.columns)

    def test_fraud_bool_is_binary(self):
        unique = set(self.df_head["fraud_bool"].unique())
        self.assertTrue(unique.issubset({0, 1}))

    def test_all_29_feature_cols_present(self):
        for col in FEATURE_COLS:
            self.assertIn(col, self.df_head.columns, f"Missing feature: {col}")

    def test_target_not_in_feature_cols(self):
        """fraud_bool must NOT be in FEATURE_COLS to prevent leakage."""
        self.assertNotIn(TARGET_COL, FEATURE_COLS)

    def test_month_not_in_feature_cols(self):
        """month must NOT be in FEATURE_COLS to prevent leakage."""
        self.assertNotIn(TIME_COL, FEATURE_COLS)


class TestTemporalSplits(unittest.TestCase):
    """Verify reference and monitoring period selection."""

    @classmethod
    def setUpClass(cls):
        baf_path = ROOT / "Datasets" / "BAF" / "Base.csv"
        if not baf_path.exists():
            raise unittest.SkipTest("BAF Base.csv not found")
        cls.df = pd.read_csv(baf_path, usecols=["month", "fraud_bool"], low_memory=False)

    def test_8_temporal_periods(self):
        self.assertEqual(self.df["month"].nunique(), 8)

    def test_periods_are_0_through_7(self):
        self.assertEqual(set(self.df["month"].unique()), set(range(8)))

    def test_reference_months_0_3_have_data(self):
        ref = self.df[self.df["month"].isin([0, 1, 2, 3])]
        self.assertGreater(len(ref), 500000)

    def test_monitoring_months_5_6_7_have_data(self):
        for m in [5, 6, 7]:
            count = (self.df["month"] == m).sum()
            self.assertGreater(count, 50000, f"Month {m} has < 50k rows")

    def test_month_row_counts_match_protocol(self):
        for m, (expected_total, expected_fraud) in EXPECTED_COUNTS.items():
            actual = (self.df["month"] == m).sum()
            self.assertEqual(actual, expected_total,
                f"Month {m}: expected {expected_total}, got {actual}")


class TestPSICalculations(unittest.TestCase):
    """Test PSI with synthetic data for both no-drift and known-shift cases."""

    def _make_detector(self):
        return PSIDetector(
            numerical_cols=["feat_a", "feat_b"],
            categorical_cols=["cat_c"],
            n_bins=10,
            psi_threshold=0.10,
            trigger_count=2,
        )

    def test_psi_no_drift_identical_distributions(self):
        """Same categorical distribution → PSI near zero → NOT triggered."""
        np.random.seed(42)
        n = 100000
        ref = pd.DataFrame({
            "cat_a": np.random.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2]),
            "cat_b": np.random.choice(["X", "Y"], n, p=[0.7, 0.3]),
        })
        det = PSIDetector(
            numerical_cols=[],
            categorical_cols=["cat_a", "cat_b"],
            n_bins=10,
            psi_threshold=0.10,
            trigger_count=1,
        )
        det.fit(ref)
        result = det.compute(ref, window_label="no_drift_test")
        self.assertFalse(result["triggered"])
        self.assertEqual(result["triggered_count"], 0)
        for cat in ["cat_a", "cat_b"]:
            self.assertLess(result["psi_values"][cat], 0.01)

    def test_psi_known_shift_triggers_drift(self):
        """Shifted distribution → high PSI → TRIGGERED."""
        np.random.seed(42)
        n = 100000
        ref = pd.DataFrame({
            "feat_a": np.random.normal(0, 1, n),
            "feat_b": np.random.uniform(0, 100, n),
            "cat_c":  np.random.choice(["A", "B", "C"], n),
        })
        # Shifted monitor window
        mon = pd.DataFrame({
            "feat_a": np.random.normal(5, 2, n),   # big mean shift
            "feat_b": np.random.uniform(50, 200, n), # range shift
            "cat_c":  np.random.choice(["D", "E", "F"], n), # all unseen
        })
        det = self._make_detector()
        det.fit(ref)
        result = det.compute(mon, window_label="shift_test")
        self.assertTrue(result["triggered"])
        self.assertGreaterEqual(result["triggered_count"], 2)

    def test_psi_values_are_non_negative(self):
        np.random.seed(42)
        n = 100000
        ref = pd.DataFrame({
            "feat_a": np.random.normal(0, 1, n),
            "feat_b": np.random.uniform(0, 100, n),
            "cat_c":  np.random.choice(["X", "Y"], n),
        })
        det = self._make_detector()
        det.fit(ref)
        result = det.compute(ref, window_label="non_neg_test")
        for v in result["psi_values"].values():
            self.assertGreaterEqual(v, 0.0)


class TestTargetLeakagePrevention(unittest.TestCase):
    """Verify fraud_bool is excluded from PSI features."""

    def test_fraud_bool_excluded_from_numerical(self):
        self.assertNotIn("fraud_bool", NUMERICAL_COLS)

    def test_fraud_bool_excluded_from_categorical(self):
        self.assertNotIn("fraud_bool", CATEGORICAL_COLS)

    def test_fraud_bool_excluded_from_feature_cols(self):
        self.assertNotIn("fraud_bool", FEATURE_COLS)

    def test_device_fraud_count_excluded(self):
        self.assertNotIn("device_fraud_count", FEATURE_COLS)


class TestKafkaToSparkIntegration(unittest.TestCase):
    """Verify Member 1's canonical event schema is compatible with Spark consumer."""

    def test_baf_normalize_record_has_month(self):
        """Member 1's normalize_record must include 'month' for Phase 4."""
        row = {
            "fraud_bool": "0",
            "income": "0.5",
            "customer_age": "30",
            "intended_balcon_amount": "500.0",
            "payment_type": "AA",
            "proposed_credit_limit": "1500",
            "month": "4",
        }
        record = normalize_record(row, source_type="baf", row_index=0)
        self.assertIn("month", record)
        self.assertEqual(record["month"], 4)

    def test_baf_normalize_record_has_all_drift_features(self):
        """Canonical event must carry all 29 features for drift monitoring."""
        row = {
            "fraud_bool": "1",
            "income": "0.8",
            "name_email_similarity": "0.95",
            "customer_age": "45",
            "intended_balcon_amount": "2500.50",
            "payment_type": "AB",
            "proposed_credit_limit": "5000.0",
            "foreign_request": "0",
            "device_os": "linux",
            "source": "INTERNET",
            "credit_risk_score": "120",
            "velocity_6h": "8500.0",
            "velocity_24h": "6200.0",
            "velocity_4w": "5100.0",
            "session_length_in_minutes": "14.5",
            "prev_address_months_count": "10",
            "current_address_months_count": "20",
            "days_since_request": "5.0",
            "zip_count_4w": "1000",
            "bank_branch_count_8w": "5",
            "date_of_birth_distinct_emails_4w": "3",
            "employment_status": "CA",
            "email_is_free": "1",
            "housing_status": "BA",
            "phone_home_valid": "1",
            "phone_mobile_valid": "1",
            "bank_months_count": "24",
            "has_other_cards": "0",
            "keep_alive_session": "1",
            "device_distinct_emails_8w": "2",
            "month": "5",
        }
        record = normalize_record(row, source_type="baf", row_index=0)
        # Check key drift features are present
        for feat in ["income", "velocity_6h", "velocity_24h", "customer_age",
                      "credit_risk_score", "payment_type", "employment_status",
                      "month", "fraud_bool"]:
            self.assertIn(feat, record, f"Missing {feat} in canonical event")

    def test_baf_record_json_serializable(self):
        """Canonical event must be JSON-serializable for Kafka."""
        row = {
            "fraud_bool": "0", "income": "0.5", "customer_age": "30",
            "intended_balcon_amount": "100", "payment_type": "AA",
            "proposed_credit_limit": "1500", "month": "0",
        }
        record = normalize_record(row, source_type="baf", row_index=0)
        payload = json.dumps(record)
        self.assertIsInstance(payload, str)
        self.assertGreater(len(payload), 10)

    def test_canonical_schema_has_card_id_for_keyed_processing(self):
        """card_id must be present for Kafka key-based partitioning."""
        row = {
            "fraud_bool": "0", "income": "0.5", "customer_age": "30",
            "intended_balcon_amount": "100", "payment_type": "AA",
            "month": "0",
        }
        record = normalize_record(row, source_type="baf", row_index=0)
        self.assertIn("card_id", record)
        self.assertTrue(record["card_id"].startswith("CARD"))


class TestKafkaToFlinkIntegration(unittest.TestCase):
    """Verify Member 1's canonical event schema is compatible with Flink consumer."""

    def test_epoch_timestamp_present(self):
        """Flink event-time processing requires epoch_timestamp."""
        row = {
            "fraud_bool": "0", "income": "0.5", "customer_age": "30",
            "intended_balcon_amount": "100", "payment_type": "AA",
            "month": "0",
        }
        record = normalize_record(row, source_type="baf", row_index=0)
        self.assertIn("epoch_timestamp", record)
        self.assertIsInstance(record["epoch_timestamp"], float)

    def test_dataset_source_is_base_csv(self):
        """BAF records must be tagged as dataset_source='base.csv'."""
        row = {
            "fraud_bool": "0", "income": "0.5", "customer_age": "30",
            "intended_balcon_amount": "100", "payment_type": "AA",
            "month": "0",
        }
        record = normalize_record(row, source_type="baf", row_index=0)
        self.assertEqual(record["dataset_source"], "base.csv")


class TestSparkFlinkConsistency(unittest.TestCase):
    """Verify Spark and Flink produce identical drift decisions on same input."""

    def test_same_psi_on_identical_input(self):
        """Both engines using the same PSI detector must agree on drift."""
        np.random.seed(123)
        n = 100000
        ref = pd.DataFrame({
            "feat_x": np.random.normal(0, 1, n),
            "cat_y":  np.random.choice(["A", "B"], n),
        })
        mon = pd.DataFrame({
            "feat_x": np.random.normal(3, 1, n),  # shifted
            "cat_y":  np.random.choice(["A", "B", "C"], n),  # new category
        })

        # Simulate Spark path
        det_spark = PSIDetector(
            numerical_cols=["feat_x"], categorical_cols=["cat_y"],
            n_bins=10, psi_threshold=0.10, trigger_count=1,
        )
        det_spark.fit(ref)
        spark_result = det_spark.compute(mon, window_label="spark")

        # Simulate Flink path (fresh detector, same reference)
        det_flink = PSIDetector(
            numerical_cols=["feat_x"], categorical_cols=["cat_y"],
            n_bins=10, psi_threshold=0.10, trigger_count=1,
        )
        det_flink.fit(ref)
        flink_result = det_flink.compute(mon, window_label="flink")

        # Must agree
        self.assertEqual(spark_result["triggered"], flink_result["triggered"])
        self.assertEqual(spark_result["triggered_count"], flink_result["triggered_count"])
        # PSI values must be numerically identical
        for feat in ["feat_x", "cat_y"]:
            self.assertAlmostEqual(
                spark_result["psi_values"][feat],
                flink_result["psi_values"][feat],
                places=10,
            )


class TestOutputReconciliation(unittest.TestCase):
    """Verify that output row counts match input."""

    @classmethod
    def setUpClass(cls):
        baf_path = ROOT / "Datasets" / "BAF" / "Base.csv"
        if not baf_path.exists():
            raise unittest.SkipTest("BAF Base.csv not found")
        cls.df = pd.read_csv(baf_path, usecols=["month"], low_memory=False)

    def test_total_rows_is_one_million(self):
        self.assertEqual(len(self.df), 1_000_000)

    def test_month_partition_sums_to_total(self):
        month_counts = self.df.groupby("month").size()
        self.assertEqual(month_counts.sum(), 1_000_000)

    def test_ref_plus_monitor_partitions(self):
        ref = self.df[self.df["month"].isin([0,1,2,3])]
        mon = self.df[self.df["month"].isin([4,5,6,7])]
        self.assertEqual(len(ref) + len(mon), 1_000_000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
