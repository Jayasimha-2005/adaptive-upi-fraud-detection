"""
Phase 2 — STEP 7 GRU Architecture Tests
==========================================
15 locked architecture tests. No training. No data loading.

Test map:
  AT1  — Model accepts [batch, 4, 406]
  AT2  — Output shape is [batch, 1]
  AT3  — Output is finite
  AT4  — Output is logits (not bounded to [0,1])
  AT5  — sigmoid(logits) in [0,1]
  AT6  — GRU input_size == 406
  AT7  — hidden_size == 64
  AT8  — num_layers == 1
  AT9  — sequence length == 4 (seq_len in config)
  AT10 — Final output layer is Linear(64 → 1)
  AT11 — BCEWithLogitsLoss consumes model output
  AT12 — pos_weight ≈ 27.47
  AT13 — Seed configuration is deterministic
  AT14 — Two models with seed=42 have identical initial parameters
  AT15 — Forward pass is reproducible under same seed+input

Run:
    python -m pytest phase2_gru/tests/test_gru_architecture.py -v
"""

import sys
import pathlib
import math

import pytest
import torch
import torch.nn as nn

# Add project root so imports work without installation
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from phase2_gru.src.models.gru_model import FraudGRU
from phase2_gru.src.training.training_config import TrainingConfig, CONFIG
from phase2_gru.src.utils.reproducibility import set_seed, get_device, SEED

# ── Constants (locked) ───────────────────────────────────────────────────────
BATCH_SIZE  = 4          # small deterministic batch for smoke tests
SEQ_LEN     = 4          # history length
INPUT_SIZE  = 406        # E2 features
HIDDEN_SIZE = 64
NUM_LAYERS  = 1
DROPOUT     = 0.2
POS_WEIGHT  = 27.47


def _make_model(seed: int = SEED) -> FraudGRU:
    """Create a fresh FraudGRU with seed-controlled initialisation."""
    set_seed(seed)
    return FraudGRU(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )


def _make_batch(seed: int = SEED) -> torch.Tensor:
    """Return a deterministic float32 batch [BATCH, SEQ_LEN, INPUT_SIZE]."""
    set_seed(seed)
    return torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# AT1 — Model accepts correct input shape
# ─────────────────────────────────────────────────────────────────────────────
class TestAT1_InputShape:
    """AT1 — Model accepts [batch, 4, 406] without error."""

    def test_model_accepts_correct_input(self):
        model = _make_model()
        model.eval()
        x = _make_batch()
        assert x.shape == (BATCH_SIZE, SEQ_LEN, INPUT_SIZE), \
            f"Batch shape wrong: {x.shape}"
        with torch.no_grad():
            out = model(x)
        assert out is not None
        print(f"AT1: PASS — input {x.shape} accepted cleanly")


# ─────────────────────────────────────────────────────────────────────────────
# AT2 — Output shape is [batch, 1]
# ─────────────────────────────────────────────────────────────────────────────
class TestAT2_OutputShape:
    """AT2 — Forward pass output shape is [batch_size, 1]."""

    def test_output_shape(self):
        model = _make_model()
        model.eval()
        x = _make_batch()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH_SIZE, 1), \
            f"AT2 FAIL: expected ({BATCH_SIZE}, 1), got {out.shape}"
        print(f"AT2: PASS — output shape {out.shape} ✅")


# ─────────────────────────────────────────────────────────────────────────────
# AT3 — Output is finite (no NaN / Inf)
# ─────────────────────────────────────────────────────────────────────────────
class TestAT3_OutputFinite:
    """AT3 — Forward pass produces finite logits."""

    def test_output_has_no_nan(self):
        model = _make_model()
        model.eval()
        x = _make_batch()
        with torch.no_grad():
            out = model(x)
        assert not torch.any(torch.isnan(out)), "AT3 FAIL: NaN in output"
        assert not torch.any(torch.isinf(out)), "AT3 FAIL: Inf in output"
        print(f"AT3: PASS — output finite, range [{out.min():.4f}, {out.max():.4f}]")


# ─────────────────────────────────────────────────────────────────────────────
# AT4 — Output is logits (NOT constrained to [0, 1])
# ─────────────────────────────────────────────────────────────────────────────
class TestAT4_LogitsNotProbabilities:
    """AT4 — Model returns raw logits, not sigmoid-clamped probabilities."""

    def test_logits_can_exceed_unit_interval(self):
        """
        With random weights and random inputs, logits will almost certainly
        fall outside [0, 1]. If they were all in [0, 1] it would indicate
        Sigmoid was applied inside the model (a bug per our spec).
        """
        set_seed(SEED)
        model = FraudGRU(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                         num_layers=NUM_LAYERS, dropout=0.0)
        model.eval()
        # Use a larger batch to increase chance of values outside [0,1]
        x = torch.randn(64, SEQ_LEN, INPUT_SIZE)
        with torch.no_grad():
            out = model(x)
        has_outside_unit = bool(
            torch.any(out < 0) or torch.any(out > 1)
        )
        assert has_outside_unit, \
            "AT4 FAIL: all outputs in [0,1] — Sigmoid may be applied inside model"
        print(f"AT4: PASS — logits in [{out.min():.3f}, {out.max():.3f}] (not bounded)")

    def test_no_sigmoid_in_forward_path(self):
        """Verify no Sigmoid module exists in the model's named modules."""
        model = _make_model()
        sigmoid_layers = [
            name for name, m in model.named_modules()
            if isinstance(m, nn.Sigmoid)
        ]
        assert len(sigmoid_layers) == 0, \
            f"AT4 FAIL: Sigmoid found in model: {sigmoid_layers}"
        print("AT4: PASS — no Sigmoid layer in model ✅")


# ─────────────────────────────────────────────────────────────────────────────
# AT5 — sigmoid(logits) in [0, 1]
# ─────────────────────────────────────────────────────────────────────────────
class TestAT5_SigmoidProducesProbabilities:
    """AT5 — sigmoid(logits) produces valid probabilities in [0, 1]."""

    def test_sigmoid_output_in_unit_interval(self):
        model = _make_model()
        model.eval()
        x = _make_batch()
        with torch.no_grad():
            logits = model(x)
            proba  = torch.sigmoid(logits)
        assert torch.all(proba >= 0.0) and torch.all(proba <= 1.0), \
            f"AT5 FAIL: sigmoid output out of [0,1]: min={proba.min()}, max={proba.max()}"
        print(f"AT5: PASS — sigmoid probabilities in [{proba.min():.4f}, {proba.max():.4f}]")

    def test_predict_proba_matches_sigmoid_logits(self):
        """predict_proba() must match sigmoid(forward())."""
        model = _make_model()
        model.eval()
        x = _make_batch()
        with torch.no_grad():
            logits     = model(x)
            proba_api  = model.predict_proba(x)
            proba_manual = torch.sigmoid(logits)
        assert torch.allclose(proba_api, proba_manual, atol=1e-6), \
            "AT5 FAIL: predict_proba() disagrees with sigmoid(forward())"
        print("AT5: PASS — predict_proba() == sigmoid(logits) ✅")


# ─────────────────────────────────────────────────────────────────────────────
# AT6 — GRU input_size == 406
# ─────────────────────────────────────────────────────────────────────────────
class TestAT6_GRUInputSize:
    """AT6 — GRU layer has input_size == 406."""

    def test_gru_input_size(self):
        model = _make_model()
        assert model.gru.input_size == INPUT_SIZE, \
            f"AT6 FAIL: gru.input_size={model.gru.input_size}, expected {INPUT_SIZE}"
        print(f"AT6: PASS — gru.input_size == {INPUT_SIZE} ✅")


# ─────────────────────────────────────────────────────────────────────────────
# AT7 — hidden_size == 64
# ─────────────────────────────────────────────────────────────────────────────
class TestAT7_HiddenSize:
    """AT7 — GRU has hidden_size == 64."""

    def test_hidden_size(self):
        model = _make_model()
        assert model.gru.hidden_size == HIDDEN_SIZE, \
            f"AT7 FAIL: hidden_size={model.gru.hidden_size}, expected {HIDDEN_SIZE}"
        print(f"AT7: PASS — hidden_size == {HIDDEN_SIZE} ✅")


# ─────────────────────────────────────────────────────────────────────────────
# AT8 — num_layers == 1
# ─────────────────────────────────────────────────────────────────────────────
class TestAT8_NumLayers:
    """AT8 — GRU has num_layers == 1."""

    def test_num_layers(self):
        model = _make_model()
        assert model.gru.num_layers == NUM_LAYERS, \
            f"AT8 FAIL: num_layers={model.gru.num_layers}, expected {NUM_LAYERS}"
        print(f"AT8: PASS — num_layers == {NUM_LAYERS} ✅")


# ─────────────────────────────────────────────────────────────────────────────
# AT9 — Sequence length == 4 (from config)
# ─────────────────────────────────────────────────────────────────────────────
class TestAT9_SequenceLength:
    """AT9 — Locked sequence length is 4 in config and accepted by model."""

    def test_seq_len_in_config(self):
        assert CONFIG.seq_len == SEQ_LEN, \
            f"AT9 FAIL: CONFIG.seq_len={CONFIG.seq_len}, expected {SEQ_LEN}"
        print(f"AT9: PASS — CONFIG.seq_len == {SEQ_LEN} ✅")

    def test_model_accepts_seq_len_4(self):
        model = _make_model()
        model.eval()
        x = torch.randn(2, SEQ_LEN, INPUT_SIZE)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1)
        print(f"AT9: PASS — model accepts seq_len={SEQ_LEN} ✅")


# ─────────────────────────────────────────────────────────────────────────────
# AT10 — Final output layer is Linear(64 → 1)
# ─────────────────────────────────────────────────────────────────────────────
class TestAT10_OutputLayer:
    """AT10 — classifier is Linear(hidden_size=64, out_features=1)."""

    def test_classifier_is_linear(self):
        model = _make_model()
        assert isinstance(model.classifier, nn.Linear), \
            f"AT10 FAIL: classifier is {type(model.classifier)}, expected nn.Linear"
        print("AT10: PASS — classifier is nn.Linear ✅")

    def test_classifier_dimensions(self):
        model = _make_model()
        assert model.classifier.in_features == HIDDEN_SIZE, \
            f"AT10 FAIL: in_features={model.classifier.in_features}, expected {HIDDEN_SIZE}"
        assert model.classifier.out_features == 1, \
            f"AT10 FAIL: out_features={model.classifier.out_features}, expected 1"
        print(f"AT10: PASS — Linear({HIDDEN_SIZE} → 1) ✅")


# ─────────────────────────────────────────────────────────────────────────────
# AT11 — BCEWithLogitsLoss consumes model output
# ─────────────────────────────────────────────────────────────────────────────
class TestAT11_LossFunction:
    """AT11 — BCEWithLogitsLoss with pos_weight can consume model logits."""

    def test_loss_computes_without_error(self):
        model = _make_model()
        model.eval()
        x = _make_batch()
        labels = torch.randint(0, 2, (BATCH_SIZE, 1), dtype=torch.float32)
        pw = torch.tensor([POS_WEIGHT], dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        with torch.no_grad():
            logits = model(x)
            loss   = loss_fn(logits, labels)
        assert torch.isfinite(loss), f"AT11 FAIL: loss={loss} is not finite"
        assert loss.item() >= 0.0,   f"AT11 FAIL: loss={loss} is negative"
        print(f"AT11: PASS — BCEWithLogitsLoss computed: {loss.item():.4f} ✅")

    def test_loss_fn_from_config(self):
        """CONFIG.get_loss_fn() returns a working BCEWithLogitsLoss."""
        model = _make_model()
        model.eval()
        x      = _make_batch()
        labels = torch.zeros(BATCH_SIZE, 1)
        loss_fn = CONFIG.get_loss_fn()
        with torch.no_grad():
            logits = model(x)
            loss   = loss_fn(logits, labels)
        assert torch.isfinite(loss)
        print(f"AT11: PASS — CONFIG.get_loss_fn() works: loss={loss.item():.4f} ✅")


# ─────────────────────────────────────────────────────────────────────────────
# AT12 — pos_weight ≈ 27.47
# ─────────────────────────────────────────────────────────────────────────────
class TestAT12_PosWeight:
    """AT12 — Locked pos_weight is approximately 27.47."""

    def test_pos_weight_value(self):
        assert math.isclose(CONFIG.pos_weight, POS_WEIGHT, rel_tol=1e-3), \
            f"AT12 FAIL: pos_weight={CONFIG.pos_weight}, expected ≈{POS_WEIGHT}"
        print(f"AT12: PASS — pos_weight={CONFIG.pos_weight} ≈ {POS_WEIGHT} ✅")

    def test_pos_weight_tensor_shape(self):
        pw = CONFIG.get_pos_weight_tensor()
        assert pw.shape == (1,), f"AT12 FAIL: pos_weight tensor shape {pw.shape}"
        assert pw.dtype == torch.float32
        print(f"AT12: PASS — pos_weight tensor shape {pw.shape}, dtype {pw.dtype} ✅")


# ─────────────────────────────────────────────────────────────────────────────
# AT13 — Seed configuration is deterministic
# ─────────────────────────────────────────────────────────────────────────────
class TestAT13_SeedDeterminism:
    """AT13 — set_seed(42) produces the same random tensors each call."""

    def test_numpy_determinism(self):
        import numpy as np
        set_seed(42)
        a = np.random.randn(5)
        set_seed(42)
        b = np.random.randn(5)
        assert np.allclose(a, b), "AT13 FAIL: NumPy not deterministic after seed"
        print("AT13: PASS — NumPy seed=42 deterministic ✅")

    def test_torch_determinism(self):
        set_seed(42)
        a = torch.randn(5)
        set_seed(42)
        b = torch.randn(5)
        assert torch.allclose(a, b), "AT13 FAIL: PyTorch not deterministic after seed"
        print("AT13: PASS — PyTorch seed=42 deterministic ✅")

    def test_config_seed_is_42(self):
        assert CONFIG.seed == SEED, f"AT13 FAIL: CONFIG.seed={CONFIG.seed}"
        print(f"AT13: PASS — CONFIG.seed == {SEED} ✅")


# ─────────────────────────────────────────────────────────────────────────────
# AT14 — Two models with seed=42 have identical initial parameters
# ─────────────────────────────────────────────────────────────────────────────
class TestAT14_ModelParameterIdentity:
    """AT14 — Two FraudGRU instances initialised with seed=42 are identical."""

    def test_identical_initial_weights(self):
        model_a = _make_model(seed=42)
        model_b = _make_model(seed=42)
        for (name_a, p_a), (name_b, p_b) in zip(
            model_a.named_parameters(), model_b.named_parameters()
        ):
            assert name_a == name_b
            assert torch.allclose(p_a, p_b), \
                f"AT14 FAIL: param '{name_a}' differs between two seed=42 models"
        print("AT14: PASS — two seed=42 models have identical initial parameters ✅")

    def test_different_seeds_give_different_weights(self):
        model_42 = _make_model(seed=42)
        model_99 = _make_model(seed=99)
        any_different = any(
            not torch.allclose(p_a, p_b)
            for p_a, p_b in zip(
                model_42.parameters(), model_99.parameters()
            )
        )
        assert any_different, \
            "AT14 FAIL: seed=42 and seed=99 models have identical weights (suspicious)"
        print("AT14: PASS — seed=42 and seed=99 give different initial weights ✅")


# ─────────────────────────────────────────────────────────────────────────────
# AT15 — Forward pass is reproducible under same seed+input
# ─────────────────────────────────────────────────────────────────────────────
class TestAT15_ForwardPassReproducibility:
    """AT15 — Same seed + same input → identical logits across two forward passes."""

    def test_forward_reproducible(self):
        model = _make_model(seed=42)
        model.eval()

        set_seed(42)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE)
        with torch.no_grad():
            out_a = model(x)

        set_seed(42)
        x2 = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE)
        with torch.no_grad():
            out_b = model(x2)

        assert torch.allclose(x, x2), \
            "AT15 FAIL: same seed produced different inputs"
        assert torch.allclose(out_a, out_b, atol=1e-6), \
            f"AT15 FAIL: outputs differ for same seed+input: {out_a} vs {out_b}"
        print(f"AT15: PASS — forward pass reproducible under seed=42 ✅")

    def test_end_to_end_smoke(self):
        """
        Smoke test: dataset → tensor → GRU → logits → loss.
        No model fitting. No data loading. Just verifies the full pipeline.
        """
        set_seed(42)
        device = torch.device("cpu")

        # Simulate one mini-batch of sequences
        x_batch = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE, dtype=torch.float32)
        y_batch = torch.tensor([0., 1., 0., 0.], dtype=torch.float32).unsqueeze(1)

        model   = FraudGRU().to(device)
        loss_fn = CONFIG.get_loss_fn(device)
        model.eval()

        with torch.no_grad():
            logits = model(x_batch.to(device))
            loss   = loss_fn(logits, y_batch.to(device))
            proba  = torch.sigmoid(logits)

        assert logits.shape == (BATCH_SIZE, 1)
        assert torch.isfinite(loss)
        assert torch.all(proba >= 0) and torch.all(proba <= 1)
        print(
            f"AT15 SMOKE: PASS — batch={BATCH_SIZE}, "
            f"logits={logits.T.numpy().round(3)}, "
            f"loss={loss.item():.4f}, "
            f"proba={proba.T.numpy().round(3)} ✅"
        )
