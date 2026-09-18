"""
Phase 2 — Locked Training Configuration
=========================================
All hyperparameters are LOCKED per sequence_specification.md v1.1.
Do NOT change these values without a documented research decision.

pos_weight derivation:
    train legitimate : 383,915
    train fraud      :  14,397
    pos_weight       = 383,915 / 14,397 ≈ 26.67

    (Note: specification locked pos_weight ≈ 27.47 based on original
     train counts before Option A filtering: 418,924 / 15,252 = 27.47.
     The Option A filtering changed train counts slightly. The locked
     value 27.47 is used as specified; a sensitivity re-run can test 26.67.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import torch


@dataclass(frozen=True)
class TrainingConfig:
    """
    Frozen dataclass holding all locked training hyperparameters.
    frozen=True prevents accidental modification at runtime.
    """

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed: int = 42

    # ── Model architecture (mirrors FraudGRU defaults) ────────────────────────
    input_size:  int   = 406
    hidden_size: int   = 64
    num_layers:  int   = 1
    dropout:     float = 0.2
    seq_len:     int   = 4

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer:    str   = "adam"
    learning_rate: float = 1e-3

    # ── Training loop ─────────────────────────────────────────────────────────
    batch_size:       int = 128
    max_epochs:       int = 30
    early_stop_patience: int = 5
    early_stop_monitor: str = "val_pr_auc"  # validation PR-AUC

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_fn:    str   = "BCEWithLogitsLoss"
    # Locked per specification (based on pre-Option-A counts: 418924 / 15252)
    pos_weight: float = 27.47

    # ── Augmentation ──────────────────────────────────────────────────────────
    use_smote:              bool = False
    use_random_oversampling: bool = False

    def get_pos_weight_tensor(self, device: torch.device = None) -> torch.Tensor:
        """Return pos_weight as a 1-element FloatTensor for BCEWithLogitsLoss."""
        t = torch.tensor([self.pos_weight], dtype=torch.float32)
        if device is not None:
            t = t.to(device)
        return t

    def get_loss_fn(self, device: torch.device = None) -> torch.nn.BCEWithLogitsLoss:
        """Instantiate the locked loss function with pos_weight."""
        return torch.nn.BCEWithLogitsLoss(
            pos_weight=self.get_pos_weight_tensor(device)
        )

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "LOCKED TRAINING CONFIGURATION",
            "=" * 50,
            f"  seed              : {self.seed}",
            f"  input_size        : {self.input_size}",
            f"  hidden_size       : {self.hidden_size}",
            f"  num_layers        : {self.num_layers}",
            f"  dropout           : {self.dropout}",
            f"  seq_len           : {self.seq_len}",
            f"  optimizer         : {self.optimizer}",
            f"  learning_rate     : {self.learning_rate}",
            f"  batch_size        : {self.batch_size}",
            f"  max_epochs        : {self.max_epochs}",
            f"  early_stop        : patience={self.early_stop_patience}, "
            f"monitor={self.early_stop_monitor}",
            f"  loss              : {self.loss_fn}",
            f"  pos_weight        : {self.pos_weight}",
            f"  SMOTE             : {self.use_smote}",
            f"  random_oversample : {self.use_random_oversampling}",
            "=" * 50,
        ]
        return "\n".join(lines)


# Module-level singleton — import and use directly
CONFIG = TrainingConfig()
