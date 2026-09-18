"""
Phase 2 — E2 GRU Trainer
==========================
Training loop with:
- Adam optimizer (lr=1e-3)
- BCEWithLogitsLoss (pos_weight=27.47)
- Early stopping on validation PR-AUC (patience=5)
- Best checkpoint saved to phase2_gru/artifacts/model/gru_best.pt
- Training history CSV written each epoch
- NaN/Inf loss detection — stops immediately and reports

LOCKED hyperparameters: defined in training_config.py
DO NOT MODIFY these values without a documented research decision.
"""

from __future__ import annotations

import csv
import json
import logging
import pathlib
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from phase2_gru.src.models.gru_model import FraudGRU
from phase2_gru.src.training.training_config import TrainingConfig, CONFIG
from phase2_gru.src.utils.reproducibility import set_seed

log = logging.getLogger(__name__)

ARTIFACTS_DIR = pathlib.Path("phase2_gru/artifacts/model")
CHECKPOINT_PATH = ARTIFACTS_DIR / "gru_best.pt"
HISTORY_CSV = pathlib.Path("phase2_gru/artifacts/training_history.csv")
METADATA_JSON = pathlib.Path("phase2_gru/artifacts/run_metadata.json")


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    val_loss: float
    val_pr_auc: float
    epoch_time: float


class EarlyStopper:
    """Monitors validation PR-AUC and stops when no improvement for `patience` epochs."""

    def __init__(self, patience: int = 5) -> None:
        self.patience = patience
        self.best_score = -1.0
        self.best_epoch = 0
        self.counter = 0

    def step(self, score: float, epoch: int) -> tuple[bool, bool]:
        """
        Returns:
            (improved, should_stop)
        """
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return True, False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return False, True
            return False, False


def _preflight_check(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """
    Run a single mini-batch through the full pipeline:
    tensor → model → logits → loss → backward → optimizer step

    Verifies: logits finite, loss finite, gradients finite, step succeeds.
    Raises RuntimeError on any failure.
    """
    log.info("Running pre-flight check (1 mini-batch)...")
    model.train()
    x_batch, y_batch = next(iter(train_loader))
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    optimizer.zero_grad()
    logits = model(x_batch)
    loss = loss_fn(logits, y_batch)

    if not torch.isfinite(logits).all():
        raise RuntimeError(f"PREFLIGHT FAIL: logits not finite: {logits}")
    if not torch.isfinite(loss):
        raise RuntimeError(f"PREFLIGHT FAIL: loss not finite: {loss}")

    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            raise RuntimeError(f"PREFLIGHT FAIL: gradient for '{name}' not finite")

    optimizer.step()
    log.info("Pre-flight check PASSED — logits=%.4f, loss=%.4f", logits.mean().item(), loss.item())
    # Reset model state after preflight
    set_seed(CONFIG.seed)
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """One full training epoch. Returns mean train loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f"NaN/Inf loss at batch {n_batches}: {loss.item()}")
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def _eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate validation set. Returns (val_loss, val_pr_auc)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_proba = []
    all_labels = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)
        total_loss += loss.item()
        n_batches += 1
        proba = torch.sigmoid(logits).cpu().numpy()
        labels = y_batch.cpu().numpy()
        all_proba.append(proba)
        all_labels.append(labels)

    all_proba  = np.concatenate(all_proba).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    val_pr_auc = float(average_precision_score(all_labels, all_proba))
    return total_loss / n_batches, val_pr_auc


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: TrainingConfig = CONFIG,
) -> dict:
    """
    Full training loop.

    Returns dict with training summary:
        best_epoch, best_val_pr_auc, history, checkpoint_path
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)

    # ── Initialise model + optimizer + loss ───────────────────────────────────
    set_seed(cfg.seed)
    model = FraudGRU(
        input_size=cfg.input_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn   = cfg.get_loss_fn(device)
    stopper   = EarlyStopper(patience=cfg.early_stop_patience)

    log.info(cfg.summary())
    log.info("Device: %s", device)
    log.info("Train batches/epoch: %d", len(train_loader))
    log.info("Val   batches/epoch: %d", len(val_loader))

    # ── Pre-flight ────────────────────────────────────────────────────────────
    _preflight_check(model, train_loader, loss_fn, optimizer, device)

    # Reinitialise cleanly after preflight
    set_seed(cfg.seed)
    model = FraudGRU(
        input_size=cfg.input_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # ── Training loop ─────────────────────────────────────────────────────────
    history: list[EpochResult] = []
    header = ["epoch", "train_loss", "val_loss", "val_pr_auc", "epoch_time_s"]

    with open(HISTORY_CSV, "w", newline="") as f:
        csv.writer(f).writerow(header)

    total_start = time.time()

    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.time()

        train_loss = _train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_pr_auc = _eval_epoch(model, val_loader, loss_fn, device)
        epoch_time = time.time() - t0

        result = EpochResult(epoch, train_loss, val_loss, val_pr_auc, epoch_time)
        history.append(result)

        improved, should_stop = stopper.step(val_pr_auc, epoch)

        # Log epoch
        log.info(
            "Epoch %2d/%d | train_loss=%.4f | val_loss=%.4f | "
            "val_PR_AUC=%.4f | time=%.1fs %s",
            epoch, cfg.max_epochs, train_loss, val_loss, val_pr_auc,
            epoch_time, "<-- BEST" if improved else ""
        )

        # Append to CSV
        with open(HISTORY_CSV, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, round(train_loss, 6), round(val_loss, 6),
                round(val_pr_auc, 6), round(epoch_time, 2)
            ])

        # Save best checkpoint
        if improved:
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_pr_auc":   val_pr_auc,
                "val_loss":     val_loss,
                "config":       {
                    "input_size":  cfg.input_size,
                    "hidden_size": cfg.hidden_size,
                    "num_layers":  cfg.num_layers,
                    "dropout":     cfg.dropout,
                },
            }, CHECKPOINT_PATH)
            log.info("Checkpoint saved: %s", CHECKPOINT_PATH)

        if should_stop:
            log.info(
                "Early stopping at epoch %d (no improvement for %d epochs). "
                "Best epoch=%d, best val_PR_AUC=%.4f",
                epoch, cfg.early_stop_patience,
                stopper.best_epoch, stopper.best_score,
            )
            break

    total_time = time.time() - total_start

    summary = {
        "best_epoch":        stopper.best_epoch,
        "best_val_pr_auc":   stopper.best_score,
        "epochs_completed":  len(history),
        "early_stopped":     len(history) < cfg.max_epochs,
        "total_time_s":      round(total_time, 1),
        "checkpoint_path":   str(CHECKPOINT_PATH),
        "history_csv":       str(HISTORY_CSV),
    }

    log.info(
        "Training complete. Best epoch=%d | Best val_PR_AUC=%.4f | "
        "Epochs=%d | Time=%.1fs",
        summary["best_epoch"], summary["best_val_pr_auc"],
        summary["epochs_completed"], summary["total_time_s"],
    )

    return summary, history
