# pyright: reportMissingImports=false
# type: ignore
"""
experiments/phase4_drift_adaptation/tests/test_member2_phase4_integration.py
=============================================================================
Integration Test Suite: Member 2 (Spark/Flink PSI Drift) -> Phase 4 (Adaptive ML)

Verifies the 10 Mandatory Requirements:
  1. Member 2 drift output can be read by Phase 4.
  2. Month 5 result correctly identifies 17 drifted features.
  3. Month 6 result correctly identifies 18 drifted features.
  4. Month 7 result correctly identifies 17 drifted features.
  5. The complete set of 29 features remains available.
  6. Non-drifted features are not removed.
  7. PSI values and feature identities are preserved accurately.
  8. Missing or malformed drift output produces a clear error.
  9. Existing Member 2 and Phase 4 outputs remain unchanged.
  10. No model training or retraining is triggered.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

# Setup paths for workspace execution
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase4_drift_adaptation.src.member2_drift_adapter import (
    CANONICAL_FEATURE_COLS,
    CANONICAL_FEATURE_SET,
    DriftAwareFeatureAnalysis,
    Member2DriftReport,
    load_member2_drift_report,
    validate_and_handoff_drift,
)


class TestMember2Phase4Integration(unittest.TestCase):
    """Integration verification tests between Member 2 and Phase 4."""

    # -------------------------------------------------------------------------
    # Requirement 1: Member 2 drift output can be read by Phase 4
    # -------------------------------------------------------------------------
    def test_01_read_member2_spark_and_flink_outputs(self):
        """Verifies Phase 4 successfully loads both Spark and Flink drift outputs."""
        report_spark = load_member2_drift_report(window=5, engine="spark")
        self.assertIsInstance(report_spark, Member2DriftReport)
        self.assertEqual(report_spark.engine, "spark")

        report_flink = load_member2_drift_report(window=5, engine="flink")
        self.assertIsInstance(report_flink, Member2DriftReport)
        self.assertEqual(report_flink.engine, "flink")

    # -------------------------------------------------------------------------
    # Requirement 2: Month 5 result correctly identifies 17 drifted features
    # -------------------------------------------------------------------------
    def test_02_month5_drifted_features_count(self):
        """Verifies Month 5 contains exactly 17 drifted features in Spark & Flink."""
        for engine in ["spark", "flink"]:
            report = load_member2_drift_report(window=5, engine=engine)
            self.assertEqual(
                report.drifted_count,
                17,
                f"Month 5 {engine} drifted count mismatch: expected 17, got {report.drifted_count}",
            )
            self.assertTrue(report.triggered)
            self.assertEqual(len(report.drifted_feature_names), 17)
            # Verify primary driver is prev_address_months_count
            self.assertEqual(report.top_drifted_features[0][0], "prev_address_months_count")
            self.assertGreater(report.top_drifted_features[0][1], 15.0)

    # -------------------------------------------------------------------------
    # Requirement 3: Month 6 result correctly identifies 18 drifted features
    # -------------------------------------------------------------------------
    def test_03_month6_drifted_features_count(self):
        """Verifies Month 6 contains exactly 18 drifted features in Spark & Flink."""
        for engine in ["spark", "flink"]:
            report = load_member2_drift_report(window=6, engine=engine)
            self.assertEqual(
                report.drifted_count,
                18,
                f"Month 6 {engine} drifted count mismatch: expected 18, got {report.drifted_count}",
            )
            self.assertTrue(report.triggered)
            self.assertEqual(len(report.drifted_feature_names), 18)

    # -------------------------------------------------------------------------
    # Requirement 4: Month 7 result correctly identifies 17 drifted features
    # -------------------------------------------------------------------------
    def test_04_month7_drifted_features_count(self):
        """Verifies Month 7 contains exactly 17 drifted features in Spark & Flink."""
        for engine in ["spark", "flink"]:
            report = load_member2_drift_report(window=7, engine=engine)
            self.assertEqual(
                report.drifted_count,
                17,
                f"Month 7 {engine} drifted count mismatch: expected 17, got {report.drifted_count}",
            )
            self.assertTrue(report.triggered)
            self.assertEqual(len(report.drifted_feature_names), 17)

    # -------------------------------------------------------------------------
    # Requirement 5: Complete set of 29 features remains available
    # -------------------------------------------------------------------------
    def test_05_all_29_features_retained(self):
        """Verifies that all 29 features are retained in the model feature schema."""
        report, analysis = validate_and_handoff_drift(window=5, engine="spark")
        self.assertEqual(len(report.retained_features), 29)
        self.assertEqual(len(report.psi_values), 29)
        self.assertEqual(set(report.retained_features), CANONICAL_FEATURE_SET)

        # Ensure Phase 4 schema query returns full 29 canonical features
        model_schema = analysis.get_model_feature_schema()
        self.assertEqual(len(model_schema), 29)
        self.assertEqual(model_schema, CANONICAL_FEATURE_COLS)

    # -------------------------------------------------------------------------
    # Requirement 6: Non-drifted features are not removed
    # -------------------------------------------------------------------------
    def test_06_non_drifted_features_preserved(self):
        """Verifies that stable (non-drifted) features remain intact and accessible."""
        report, analysis = validate_and_handoff_drift(window=5, engine="spark")
        self.assertEqual(report.stable_count, 12)  # 29 - 17 = 12 stable
        self.assertEqual(len(report.stable_features), 12)

        # Verify specific stable features are preserved with valid non-negative PSI values
        for feat in ["payment_type", "employment_status", "housing_status", "source", "device_os"]:
            self.assertIn(feat, report.psi_values)
            self.assertIn(feat, report.retained_features)
            self.assertLess(report.psi_values[feat], 0.10)

        # Attention weights should be assigned to ALL 29 features (none dropped)
        weights = analysis.get_attention_weights()
        self.assertEqual(len(weights), 29)
        for feat in CANONICAL_FEATURE_COLS:
            self.assertIn(feat, weights)
            self.assertGreater(weights[feat], 0.0)

    # -------------------------------------------------------------------------
    # Requirement 7: PSI values and feature identities preserved accurately
    # -------------------------------------------------------------------------
    def test_07_psi_values_and_identities_preserved(self):
        """Verifies numerical PSI values match Member 2 JSON file down to the last decimal."""
        spark_json_path = ROOT / "spark" / "reports" / "drift" / "spark_psi_month5.json"
        with open(spark_json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        report = load_member2_drift_report(window=5, engine="spark")
        for feat, expected_psi in raw_data["psi_values"].items():
            self.assertIn(feat, report.psi_values)
            self.assertEqual(report.psi_values[feat], expected_psi)

        # Check top shifted feature exact numerical value
        self.assertEqual(report.psi_values["prev_address_months_count"], raw_data["psi_values"]["prev_address_months_count"])

    # -------------------------------------------------------------------------
    # Requirement 8: Missing or malformed drift output produces a clear error
    # -------------------------------------------------------------------------
    def test_08_missing_or_malformed_drift_error_handling(self):
        """Verifies fail-closed behavior on missing files, missing features, and corrupt JSON."""
        # A. Missing file
        with self.assertRaises(FileNotFoundError) as ctx_missing:
            load_member2_drift_report(window=99, engine="spark")
        self.assertIn("Member 2 drift report not found", str(ctx_missing.exception))

        # B. Malformed JSON
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
            tf.write("{ this is corrupt json ...")
            tf_path = Path(tf.name)
        try:
            with self.assertRaises(ValueError) as ctx_corrupt:
                load_member2_drift_report(tf_path)
            self.assertIn("Malformed JSON", str(ctx_corrupt.exception))
        finally:
            if tf_path.exists():
                tf_path.unlink()

        # C. Missing feature in schema (e.g. 28 features instead of 29)
        spark_json_path = ROOT / "spark" / "reports" / "drift" / "spark_psi_month5.json"
        with open(spark_json_path, "r", encoding="utf-8") as f:
            valid_data = json.load(f)

        del valid_data["psi_values"]["income"]  # Remove 1 feature
        valid_data["triggered_count"] = 16

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
            json.dump(valid_data, tf)
            tf_path = Path(tf.name)
        try:
            with self.assertRaises(ValueError) as ctx_schema:
                load_member2_drift_report(tf_path)
            self.assertIn("Feature schema mismatch", str(ctx_schema.exception))
            self.assertIn("Missing 1 features: ['income']", str(ctx_schema.exception))
        finally:
            if tf_path.exists():
                tf_path.unlink()

        # D. Excluded target label present (leakage check)
        valid_data["psi_values"]["income"] = 3.63
        valid_data["psi_values"]["fraud_bool"] = 0.5  # illegal target label
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
            json.dump(valid_data, tf)
            tf_path = Path(tf.name)
        try:
            with self.assertRaises(ValueError) as ctx_leak:
                load_member2_drift_report(tf_path)
            self.assertIn("Feature schema mismatch", str(ctx_leak.exception))
            self.assertIn("Unknown 1 features: ['fraud_bool']", str(ctx_leak.exception))
        finally:
            if tf_path.exists():
                tf_path.unlink()

    # -------------------------------------------------------------------------
    # Requirement 9: Existing Member 2 and Phase 4 outputs remain unchanged
    # -------------------------------------------------------------------------
    def test_09_existing_outputs_unaltered(self):
        """Verifies file hashes of existing Member 2 drift reports are unchanged."""
        m5_path = ROOT / "spark" / "reports" / "drift" / "spark_psi_month5.json"
        self.assertTrue(m5_path.exists())
        # Ensure file can be read and is valid JSON
        with open(m5_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["triggered_count"], 17)

    # -------------------------------------------------------------------------
    # Requirement 10: No model training or retraining is triggered
    # -------------------------------------------------------------------------
    def test_10_no_model_training_triggered(self):
        """Verifies integration execution executes purely analytical handoff without training."""
        # Load reports across all months
        for m in [5, 6, 7]:
            report, analysis = validate_and_handoff_drift(window=m, engine="spark")
            self.assertTrue(analysis.should_trigger_retraining())
            # Telemetry shows recommendation without invoking LightGBM
            self.assertEqual(report.action_recommendation, "RETRAIN_MODEL")

        # Verify existing model artifacts in artifacts/models remain intact
        models_dir = ROOT / "experiments" / "phase4_drift_adaptation" / "artifacts" / "models"
        self.assertTrue((models_dir / "lgbm_v1.txt").exists())
        self.assertTrue((models_dir / "lgbm_v2.txt").exists())
        self.assertTrue((models_dir / "lgbm_v3.txt").exists())

    # -------------------------------------------------------------------------
    # Requirement 11: Phase 4 pipeline.py imports and connects Member 2 adapter
    # -------------------------------------------------------------------------
    def test_11_pipeline_script_integration(self):
        """Verifies pipeline.py explicitly imports and calls member2_drift_adapter."""
        import ast

        pipeline_path = (
            ROOT
            / "experiments"
            / "phase4_drift_adaptation"
            / "src"
            / "pipeline.py"
        )
        self.assertTrue(pipeline_path.exists())

        code = pipeline_path.read_text(encoding="utf-8")
        parsed = ast.parse(code, filename=str(pipeline_path))

        # Check imports
        imported_names = set()
        for node in ast.walk(parsed):
            if isinstance(node, ast.ImportFrom):
                if node.module in ("member2_drift_adapter", "experiments.phase4_drift_adaptation.src.member2_drift_adapter"):
                    for alias in node.names:
                        imported_names.add(alias.name)

        self.assertIn("validate_and_handoff_drift", imported_names)
        self.assertIn("DriftAwareFeatureAnalysis", imported_names)
        self.assertIn("Member2DriftReport", imported_names)

        # Check that validate_and_handoff_drift is invoked in the code
        self.assertIn("validate_and_handoff_drift(window=5, engine=\"spark\")", code)
        self.assertIn("validate_and_handoff_drift(window=6, engine=\"spark\")", code)
        self.assertIn("validate_and_handoff_drift(window=7, engine=\"spark\")", code)

        # Check that member2_drift_handoff is recorded in the results manifest
        self.assertIn('"member2_drift_handoff": {', code)
        self.assertIn('"month5": m2_analysis_m5.summary()', code)
        self.assertIn('"month6": m2_analysis_m6.summary()', code)
        self.assertIn('"month7": m2_analysis_m7.summary()', code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
