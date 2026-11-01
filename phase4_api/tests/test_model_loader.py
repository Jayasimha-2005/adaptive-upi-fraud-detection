"""
phase4_api/tests/test_model_loader.py
=====================================
Unit test suite for Phase 4 Model Loader.

Validates the 12 requirements:
  1. Approved v3 model loads successfully.
  2. Loaded model has exactly 29 features.
  3. Feature names match the original schema.
  4. Feature ordering is preserved.
  5. Model version is correctly identified.
  6. Frozen threshold is loaded correctly.
  7. Missing model produces a clear error.
  8. Corrupted model produces a clear error.
  9. Invalid feature count is rejected.
  10. Invalid threshold is rejected.
  11. Missing metadata is handled correctly.
  12. Windows CRLF normalization works.
  13. Caching / singleton prevents redundant loading.
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

# Ensure workspace root is in sys.path
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase4_api.src.model_loader import (
    APPROVED_DEFAULT_MODEL_VERSION,
    CANONICAL_CATEGORICAL_COLS,
    CANONICAL_FEATURE_COLS,
    CANONICAL_NUMERICAL_COLS,
    EXPECTED_FEATURE_COUNT,
    FeatureSchemaMismatchError,
    LoadedModelBundle,
    ModelArtifactNotFoundError,
    ModelCorruptedError,
    ModelInfo,
    ModelMetadataError,
    Phase4ModelLoader,
    ThresholdConfigurationError,
    UnsupportedModelVersionError,
    clear_model_cache,
    find_project_root,
    get_model_bundle,
    get_model_info,
)


class TestModelLoader(unittest.TestCase):
    """Test suite for Phase4ModelLoader and associated utilities."""

    def setUp(self) -> None:
        clear_model_cache()
        self.project_root = find_project_root()
        self.loader = Phase4ModelLoader(
            project_root=self.project_root,
            model_version=APPROVED_DEFAULT_MODEL_VERSION,
        )

    def tearDown(self) -> None:
        clear_model_cache()

    # -------------------------------------------------------------------------
    # Requirement 1: Approved v3 model loads successfully
    # -------------------------------------------------------------------------
    def test_01_approved_v3_model_loads_successfully(self) -> None:
        bundle = self.loader.load()
        self.assertIsInstance(bundle, LoadedModelBundle)
        self.assertIsNotNone(bundle.booster)
        self.assertEqual(bundle.info.model_loading_status, "ACTIVE_LOADED")

    # -------------------------------------------------------------------------
    # Requirement 2: Loaded model has exactly 29 features
    # -------------------------------------------------------------------------
    def test_02_loaded_model_feature_count(self) -> None:
        bundle = self.loader.load()
        self.assertEqual(bundle.booster.num_feature(), EXPECTED_FEATURE_COUNT)
        self.assertEqual(bundle.info.num_features, 29)
        self.assertEqual(len(bundle.info.feature_names), 29)

    # -------------------------------------------------------------------------
    # Requirement 3: Feature names match original schema
    # -------------------------------------------------------------------------
    def test_03_feature_names_match_canonical_schema(self) -> None:
        bundle = self.loader.load()
        actual_names = bundle.booster.feature_name()
        self.assertEqual(set(actual_names), set(CANONICAL_FEATURE_COLS))

        # Check breakdown
        for num_col in CANONICAL_NUMERICAL_COLS:
            self.assertIn(num_col, actual_names)
        for cat_col in CANONICAL_CATEGORICAL_COLS:
            self.assertIn(cat_col, actual_names)

    # -------------------------------------------------------------------------
    # Requirement 4: Feature ordering is preserved
    # -------------------------------------------------------------------------
    def test_04_feature_ordering_is_strictly_preserved(self) -> None:
        bundle = self.loader.load()
        actual_names = bundle.booster.feature_name()
        self.assertEqual(list(actual_names), list(CANONICAL_FEATURE_COLS))

        # Verify numerical precede categorical
        for i in range(24):
            self.assertEqual(actual_names[i], CANONICAL_NUMERICAL_COLS[i])
        for j in range(5):
            self.assertEqual(actual_names[24 + j], CANONICAL_CATEGORICAL_COLS[j])

    # -------------------------------------------------------------------------
    # Requirement 5: Model version is correctly identified
    # -------------------------------------------------------------------------
    def test_05_model_version_identified(self) -> None:
        bundle = self.loader.load()
        self.assertEqual(bundle.info.model_version, "v3")
        self.assertEqual(bundle.metadata.get("version_label"), "v3")
        self.assertEqual(bundle.metadata.get("training_months"), [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(bundle.info.num_trees, 500)

    # -------------------------------------------------------------------------
    # Requirement 6: Frozen threshold is loaded correctly
    # -------------------------------------------------------------------------
    def test_06_frozen_threshold_loaded_correctly(self) -> None:
        threshold = self.loader.load_frozen_threshold()
        self.assertAlmostEqual(threshold, 0.8964321608040201, places=6)
        self.assertGreater(threshold, 0.0)
        self.assertLess(threshold, 1.0)

        # Check bundle propagation
        bundle = self.loader.load()
        self.assertEqual(bundle.frozen_threshold, threshold)
        self.assertEqual(bundle.info.frozen_threshold, threshold)

    # -------------------------------------------------------------------------
    # Requirement 7: Missing model produces a clear error
    # -------------------------------------------------------------------------
    def test_07_missing_model_produces_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = pathlib.Path(tmp_dir)
            fake_loader = Phase4ModelLoader(
                project_root=tmp_root,
                model_version="v3",
            )
            # Threshold missing first, or if we mock threshold, model missing
            with self.assertRaises(ThresholdConfigurationError):
                fake_loader.load()

            # Now test load_booster directly with non-existent path
            non_existent = tmp_root / "lgbm_v3.txt"
            with self.assertRaises(ModelArtifactNotFoundError) as ctx:
                fake_loader.load_booster(artifact_path=non_existent)
            self.assertIn("not found", str(ctx.exception).lower())

    # -------------------------------------------------------------------------
    # Requirement 8: Corrupted model produces a clear error
    # -------------------------------------------------------------------------
    def test_08_corrupted_model_produces_error(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tf:
            tf.write(b"NOT_A_VALID_LIGHTGBM_MODEL_CORRUPTED_BYTES_123456")
            tf_path = pathlib.Path(tf.name)
        try:
            with self.assertRaises(ModelCorruptedError) as ctx:
                self.loader.load_booster(artifact_path=tf_path)
            self.assertIn("initialization failed", str(ctx.exception).lower())
        finally:
            if tf_path.exists():
                tf_path.unlink()

    # -------------------------------------------------------------------------
    # Requirement 9: Invalid feature count is rejected
    # -------------------------------------------------------------------------
    def test_09_invalid_feature_count_rejected(self) -> None:
        # Create a mock booster with num_feature != 29
        class MockBooster:
            def num_feature(self) -> int:
                return 15

            def feature_name(self) -> list[str]:
                return ["col"] * 15

        mock = MockBooster()
        with self.assertRaises(FeatureSchemaMismatchError) as ctx:
            self.loader.validate_features(mock)  # type: ignore[arg-type]
        self.assertIn("Feature count mismatch: model has 15 features, expected exactly 29", str(ctx.exception))

    # -------------------------------------------------------------------------
    # Requirement 10: Invalid threshold is rejected
    # -------------------------------------------------------------------------
    def test_10_invalid_threshold_rejected(self) -> None:
        # Case A: threshold > 1.0
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = pathlib.Path(tmp_dir)
            t_dir = tmp_root / "experiments" / "phase4_drift_adaptation" / "artifacts" / "thresholds"
            t_dir.mkdir(parents=True)
            t_file = t_dir / "frozen_threshold_v1.json"
            t_file.write_text(json.dumps({"threshold": 1.5}))

            loader = Phase4ModelLoader(project_root=tmp_root, model_version="v3")
            with self.assertRaises(ThresholdConfigurationError) as ctx:
                loader.load_frozen_threshold()
            self.assertIn("outside valid probability interval", str(ctx.exception))

            # Case B: non-numeric threshold
            t_file.write_text(json.dumps({"threshold": "not_a_number"}))
            with self.assertRaises(ThresholdConfigurationError) as ctx_nan:
                loader.load_frozen_threshold()
            self.assertIn("not a valid float", str(ctx_nan.exception))

            # Case C: missing threshold key
            t_file.write_text(json.dumps({"other_key": 0.5}))
            with self.assertRaises(ThresholdConfigurationError) as ctx_key:
                loader.load_frozen_threshold()
            self.assertIn("missing 'threshold' key", str(ctx_key.exception))

    # -------------------------------------------------------------------------
    # Requirement 11: Missing metadata is handled correctly
    # -------------------------------------------------------------------------
    def test_11_missing_metadata_handled_correctly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = pathlib.Path(tmp_dir)
            m_dir = tmp_root / "experiments" / "phase4_drift_adaptation" / "artifacts" / "models"
            m_dir.mkdir(parents=True)
            # No metadata file created
            loader = Phase4ModelLoader(project_root=tmp_root, model_version="v3")
            with self.assertRaises(ModelMetadataError) as ctx:
                loader.load_metadata()
            self.assertIn("missing at", str(ctx.exception).lower())

            # Corrupted metadata JSON
            meta_path = m_dir / "lgbm_v3_meta.json"
            meta_path.write_text("{broken_json: true")
            with self.assertRaises(ModelMetadataError) as ctx_corrupt:
                loader.load_metadata()
            self.assertIn("failed to parse", str(ctx_corrupt.exception).lower())

    # -------------------------------------------------------------------------
    # Requirement 12: Windows CRLF normalization works
    # -------------------------------------------------------------------------
    def test_12_windows_crlf_normalization(self) -> None:
        # Read the real v3 model artifact bytes
        real_path = self.loader.model_artifact_path
        self.assertTrue(real_path.exists())
        real_bytes = real_path.read_bytes()

        # Artificially inject CRLF across all newlines
        crlf_bytes = real_bytes.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        self.assertIn(b"\r\n", crlf_bytes)

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tf:
            tf.write(crlf_bytes)
            tf_path = pathlib.Path(tf.name)

        try:
            # Load with CRLF normalization
            booster = self.loader.load_booster(artifact_path=tf_path)
            self.assertEqual(booster.num_feature(), EXPECTED_FEATURE_COUNT)
            self.assertEqual(booster.num_trees(), 500)
        finally:
            if tf_path.exists():
                tf_path.unlink()

    # -------------------------------------------------------------------------
    # Requirement 13: Caching / singleton prevents redundant loading
    # -------------------------------------------------------------------------
    def test_13_caching_and_singleton_behavior(self) -> None:
        bundle1 = get_model_bundle("v3")
        bundle2 = get_model_bundle("v3")
        # Exact same in-memory object instance
        self.assertIs(bundle1, bundle2)
        self.assertIs(bundle1.booster, bundle2.booster)

        # Force reload returns new object
        bundle3 = get_model_bundle("v3", force_reload=True)
        self.assertIsNot(bundle1, bundle3)

        # Public info serialization
        info = get_model_info("v3")
        self.assertIsInstance(info, ModelInfo)
        public_dict = info.to_public_dict()
        self.assertEqual(public_dict["model_version"], "v3")
        self.assertEqual(public_dict["model_artifact_path"], "lgbm_v3.txt")
        self.assertEqual(public_dict["num_features"], 29)
        self.assertNotIn("Users", public_dict["model_artifact_path"])

    # -------------------------------------------------------------------------
    # Requirement 14: Unsupported version rejected
    # -------------------------------------------------------------------------
    def test_14_unsupported_model_version_rejected(self) -> None:
        with self.assertRaises(UnsupportedModelVersionError):
            Phase4ModelLoader(model_version="v999")


if __name__ == "__main__":
    unittest.main(verbosity=2)
