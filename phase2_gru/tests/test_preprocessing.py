"""
Phase 2 — E2b Preprocessing Tests (PT1–PT8)
=============================================
Verifies that SequenceStandardScaler:
  PT1 — Scaler statistics fitted only on training data
  PT2 — Validation data does not affect fitted statistics
  PT3 — Test data does not affect fitted statistics
  PT4 — Feature count remains 406 after transform
  PT5 — Feature ordering is unchanged
  PT6 — No NaN/Inf after transformation
  PT7 — Correct tensor shapes maintained
  PT8 — Transformation is deterministic

Run:
    python -m pytest phase2_gru/tests/test_preprocessing.py -v
"""

import sys
import pathlib
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from phase2_gru.src.data.preprocessing import SequenceStandardScaler

# ── Deterministic toy data ────────────────────────────────────────────────────
RNG = np.random.default_rng(42)

N_TRAIN = 200
N_VAL   = 50
N_TEST  = 40
SEQ_LEN = 4
N_FEAT  = 406

def _make_data():
    """Generate deterministic toy train/val/test arrays [N, 4, 406]."""
    X_train = RNG.normal(loc=10.0, scale=100.0,
                         size=(N_TRAIN, SEQ_LEN, N_FEAT)).astype(np.float32)
    X_val   = RNG.normal(loc=10.0, scale=100.0,
                         size=(N_VAL,   SEQ_LEN, N_FEAT)).astype(np.float32)
    X_test  = RNG.normal(loc=10.0, scale=100.0,
                         size=(N_TEST,  SEQ_LEN, N_FEAT)).astype(np.float32)
    return X_train, X_val, X_test


# ─────────────────────────────────────────────────────────────────────────────
# PT1 — Statistics fitted only on training data
# ─────────────────────────────────────────────────────────────────────────────
class TestPT1_FitOnTrainOnly:
    """PT1 — Scaler mean/scale computed from X_train only."""

    def test_mean_matches_train_flat(self):
        X_train, _, _ = _make_data()
        scaler = SequenceStandardScaler()
        scaler.fit(X_train)
        expected_mean = X_train.reshape(-1, N_FEAT).astype(np.float64).mean(axis=0).astype(np.float32)
        assert np.allclose(scaler.mean_, expected_mean, rtol=1e-5), \
            "PT1 FAIL: fitted mean does not match train flat mean"

    def test_scale_matches_train_flat(self):
        X_train, _, _ = _make_data()
        scaler = SequenceStandardScaler()
        scaler.fit(X_train)
        expected_std = X_train.reshape(-1, N_FEAT).astype(np.float64).std(axis=0).astype(np.float32)
        expected_scale = np.where(expected_std < 1e-8, np.float32(1.0), expected_std)
        assert np.allclose(scaler.scale_, expected_scale, rtol=1e-5), \
            "PT1 FAIL: fitted scale does not match train std"

    def test_is_fitted_true_after_fit(self):
        X_train, _, _ = _make_data()
        scaler = SequenceStandardScaler()
        assert not scaler.is_fitted
        scaler.fit(X_train)
        assert scaler.is_fitted, "PT1 FAIL: is_fitted should be True after fit()"

    def test_transform_before_fit_raises(self):
        _, X_val, _ = _make_data()
        scaler = SequenceStandardScaler()
        with pytest.raises(RuntimeError, match="fit"):
            scaler.transform(X_val)


# ─────────────────────────────────────────────────────────────────────────────
# PT2 — Validation does not affect fitted statistics
# ─────────────────────────────────────────────────────────────────────────────
class TestPT2_ValDoesNotAffectScaler:
    """PT2 — Transforming validation data does not change the fitted statistics."""

    def test_mean_unchanged_after_val_transform(self):
        X_train, X_val, _ = _make_data()
        scaler = SequenceStandardScaler()
        scaler.fit(X_train)
        mean_before = scaler.mean_.copy()
        scaler.transform(X_val)   # apply to val
        assert np.array_equal(scaler.mean_, mean_before), \
            "PT2 FAIL: scaler.mean_ changed after val transform"

    def test_scale_unchanged_after_val_transform(self):
        X_train, X_val, _ = _make_data()
        scaler = SequenceStandardScaler()
        scaler.fit(X_train)
        scale_before = scaler.scale_.copy()
        scaler.transform(X_val)
        assert np.array_equal(scaler.scale_, scale_before), \
            "PT2 FAIL: scaler.scale_ changed after val transform"

    def test_val_transform_uses_train_stats(self):
        """Manually verify: val output = (X_val - train_mean) / train_std."""
        X_train, X_val, _ = _make_data()
        scaler = SequenceStandardScaler()
        scaler.fit(X_train)
        X_val_scaled = scaler.transform(X_val)
        expected = (X_val - scaler.mean_) / scaler.scale_
        assert np.allclose(X_val_scaled, expected, atol=1e-5), \
            "PT2 FAIL: val transform does not use train statistics"


# ─────────────────────────────────────────────────────────────────────────────
# PT3 — Test data does not affect fitted statistics
# ─────────────────────────────────────────────────────────────────────────────
class TestPT3_TestDoesNotAffectScaler:
    """PT3 — Transforming test data does not change fitted statistics."""

    def test_mean_unchanged_after_test_transform(self):
        X_train, _, X_test = _make_data()
        scaler = SequenceStandardScaler()
        scaler.fit(X_train)
        mean_before = scaler.mean_.copy()
        scaler.transform(X_test)
        assert np.array_equal(scaler.mean_, mean_before), \
            "PT3 FAIL: scaler.mean_ changed after test transform"

    def test_scale_unchanged_after_test_transform(self):
        X_train, _, X_test = _make_data()
        scaler = SequenceStandardScaler()
        scaler.fit(X_train)
        scale_before = scaler.scale_.copy()
        scaler.transform(X_test)
        assert np.array_equal(scaler.scale_, scale_before), \
            "PT3 FAIL: scaler.scale_ changed after test transform"


# ─────────────────────────────────────────────────────────────────────────────
# PT4 — Feature count remains 406
# ─────────────────────────────────────────────────────────────────────────────
class TestPT4_FeatureCount:
    """PT4 — Transform preserves feature dimension == 406."""

    def test_train_feature_count_after_transform(self):
        X_train, _, _ = _make_data()
        scaler = SequenceStandardScaler()
        X_out = scaler.fit_transform(X_train)
        assert X_out.shape[2] == N_FEAT, \
            f"PT4 FAIL: feature dim after transform = {X_out.shape[2]}, expected {N_FEAT}"

    def test_val_feature_count_after_transform(self):
        X_train, X_val, _ = _make_data()
        scaler = SequenceStandardScaler()
        scaler.fit(X_train)
        X_out = scaler.transform(X_val)
        assert X_out.shape[2] == N_FEAT, \
            f"PT4 FAIL: val feature dim = {X_out.shape[2]}, expected {N_FEAT}"


# ─────────────────────────────────────────────────────────────────────────────
# PT5 — Feature ordering unchanged
# ─────────────────────────────────────────────────────────────────────────────
class TestPT5_FeatureOrdering:
    """PT5 — Each feature transforms independently; ordering is preserved."""

    def test_feature_order_preserved(self):
        """
        Verify that feature i of the output is derived from feature i of the input.
        Insert a sentinel column (feature 200) with known mean/std.
        """
        X_train, X_val, _ = _make_data()
        # Overwrite feature 200 in train with constant=5.0
        X_train_mod = X_train.copy()
        X_train_mod[:, :, 200] = 5.0
        scaler = SequenceStandardScaler()
        scaler.fit(X_train_mod)
        # mean at position 200 should be 5.0; scale should be set to 1.0 (zero var)
        assert abs(scaler.mean_[200] - 5.0) < 1e-3, \
            f"PT5 FAIL: feature ordering wrong at index 200. mean={scaler.mean_[200]}"

    def test_transform_per_feature_independence(self):
        """
        After transform, each output feature depends ONLY on its input feature,
        not on any other feature.
        """
        X_train, _, _ = _make_data()
        X_train2 = X_train.copy()
        # Swap features 0 and 1 in a copy
        X_train2[:, :, [0, 1]] = X_train2[:, :, [1, 0]]
        s1 = SequenceStandardScaler().fit(X_train)
        s2 = SequenceStandardScaler().fit(X_train2)
        # s2.mean_[0] should equal s1.mean_[1] and vice versa
        assert abs(s2.mean_[0] - s1.mean_[1]) < 1e-3, "PT5 FAIL: feature ordering not independent"
        assert abs(s2.mean_[1] - s1.mean_[0]) < 1e-3, "PT5 FAIL: feature ordering not independent"


# ─────────────────────────────────────────────────────────────────────────────
# PT6 — No NaN/Inf after transformation
# ─────────────────────────────────────────────────────────────────────────────
class TestPT6_NoNaNInf:
    """PT6 — Transformed output contains no NaN or Inf values."""

    def test_no_nan_after_train_transform(self):
        X_train, _, _ = _make_data()
        scaler = SequenceStandardScaler()
        X_out = scaler.fit_transform(X_train)
        assert not np.any(np.isnan(X_out)), "PT6 FAIL: NaN in train transform output"
        assert not np.any(np.isinf(X_out)), "PT6 FAIL: Inf in train transform output"

    def test_no_nan_after_val_transform(self):
        X_train, X_val, _ = _make_data()
        scaler = SequenceStandardScaler()
        scaler.fit(X_train)
        X_out = scaler.transform(X_val)
        assert not np.any(np.isnan(X_out)), "PT6 FAIL: NaN in val transform output"
        assert not np.any(np.isinf(X_out)), "PT6 FAIL: Inf in val transform output"

    def test_zero_variance_feature_handled(self):
        """A constant feature (zero variance) must not produce NaN or Inf."""
        X_train, _, _ = _make_data()
        X_train_mod = X_train.copy()
        X_train_mod[:, :, 300] = 42.0  # constant feature
        scaler = SequenceStandardScaler()
        X_out = scaler.fit_transform(X_train_mod)
        assert not np.any(np.isnan(X_out[:, :, 300])), \
            "PT6 FAIL: zero-variance feature produced NaN"
        assert not np.any(np.isinf(X_out[:, :, 300])), \
            "PT6 FAIL: zero-variance feature produced Inf"
        # After centering (mean=42) and dividing by 1.0 → output should be near 0
        assert np.allclose(X_out[:, :, 300], 0.0, atol=1e-5), \
            "PT6 FAIL: zero-variance feature should be ~0 after transform"


# ─────────────────────────────────────────────────────────────────────────────
# PT7 — Correct tensor shapes
# ─────────────────────────────────────────────────────────────────────────────
class TestPT7_TensorShapes:
    """PT7 — Output shapes exactly match input shapes."""

    def test_train_shape_preserved(self):
        X_train, _, _ = _make_data()
        scaler = SequenceStandardScaler()
        X_out = scaler.fit_transform(X_train)
        assert X_out.shape == X_train.shape, \
            f"PT7 FAIL: train shape {X_out.shape} != {X_train.shape}"

    def test_val_shape_preserved(self):
        X_train, X_val, _ = _make_data()
        scaler = SequenceStandardScaler()
        scaler.fit(X_train)
        X_out = scaler.transform(X_val)
        assert X_out.shape == X_val.shape, \
            f"PT7 FAIL: val shape {X_out.shape} != {X_val.shape}"

    def test_output_dtype_is_float32(self):
        X_train, X_val, _ = _make_data()
        scaler = SequenceStandardScaler()
        scaler.fit(X_train)
        X_out_train = scaler.transform(X_train)
        X_out_val   = scaler.transform(X_val)
        assert X_out_train.dtype == np.float32, f"PT7 FAIL: dtype={X_out_train.dtype}"
        assert X_out_val.dtype   == np.float32, f"PT7 FAIL: dtype={X_out_val.dtype}"


# ─────────────────────────────────────────────────────────────────────────────
# PT8 — Deterministic transformation
# ─────────────────────────────────────────────────────────────────────────────
class TestPT8_Deterministic:
    """PT8 — Same input always produces same output."""

    def test_same_input_same_output(self):
        X_train, X_val, _ = _make_data()
        scaler = SequenceStandardScaler()
        scaler.fit(X_train)
        out_a = scaler.transform(X_val)
        out_b = scaler.transform(X_val)
        assert np.array_equal(out_a, out_b), \
            "PT8 FAIL: two transforms of same input gave different outputs"

    def test_two_fitted_scalers_identical(self):
        """Two scalers fitted on the same train data must be identical."""
        X_train, X_val, _ = _make_data()
        s1 = SequenceStandardScaler().fit(X_train)
        s2 = SequenceStandardScaler().fit(X_train)
        assert np.array_equal(s1.mean_,  s2.mean_),  "PT8 FAIL: means differ"
        assert np.array_equal(s1.scale_, s2.scale_), "PT8 FAIL: scales differ"
        assert np.array_equal(s1.transform(X_val), s2.transform(X_val)), \
            "PT8 FAIL: outputs from two identical scalers differ"

    def test_save_load_roundtrip(self, tmp_path):
        """Saved and reloaded scaler produces identical output."""
        X_train, X_val, _ = _make_data()
        scaler = SequenceStandardScaler()
        scaler.fit(X_train)
        out_before = scaler.transform(X_val)
        # Save and reload
        save_path = tmp_path / "test_scaler.pkl"
        scaler.save(save_path)
        loaded = SequenceStandardScaler.load(save_path)
        out_after = loaded.transform(X_val)
        assert np.array_equal(out_before, out_after), \
            "PT8 FAIL: save/load roundtrip produced different output"
