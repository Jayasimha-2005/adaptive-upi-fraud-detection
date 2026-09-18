"""
Phase 2 — STEP 8: E2 GRU Training Runner
==========================================
Entry point for training. Executes:
  1. Set seed + detect device
  2. Generate E2 sequences (sequence_builder)
  3. Build DataLoaders
  4. Pre-flight check
  5. Full training loop (max 30 epochs, early stop patience=5)
  6. Save best checkpoint + metadata
  7. Print complete training summary

STOP CONDITION: This script does NOT evaluate the test set.
                STEP 9 handles test evaluation.

Usage:
    python phase2_gru/src/training/run_training.py
"""

import sys
import pathlib
import json
import logging
import time
import torch

# Project root on path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger(__name__)

from phase2_gru.src.utils.reproducibility import configure, SEED
from phase2_gru.src.training.training_config import CONFIG
from phase2_gru.src.data.sequence_builder import load_all_splits, build_sequences
from phase2_gru.src.data.dataset import make_loader
from phase2_gru.src.training.trainer import train, CHECKPOINT_PATH, METADATA_JSON

# ── Separator helpers ─────────────────────────────────────────────────────────
SEP = "=" * 65

def sep(title=""):
    if title:
        log.info("%s", SEP)
        log.info("  %s", title)
        log.info("%s", SEP)
    else:
        log.info("%s", SEP)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    t_script_start = time.time()

    sep("STEP 8 - E2 GRU TRAINING")
    log.info("Seed: %d", SEED)

    # 1. Reproducibility + device
    device = configure(SEED)
    log.info("PyTorch version: %s", torch.__version__)
    log.info("Device: %s", device)

    # 2. Generate sequences
    sep("Generating E2 sequences (Option A, strict DT ordering)")
    log.info("Loading all parquets...")
    df = load_all_splits()
    log.info("Rows loaded: %d", len(df))

    log.info("Building sequences (this takes ~17 min on CPU)...")
    seqs = build_sequences(df, smoke_test_n=None, verbose=True)
    n_excl = seqs["_meta"]["n_excluded_dup_dt"]
    log.info("Sequences built. Excluded (duplicate DT): %d", n_excl)

    # 3. Extract arrays
    sep("Data summary")
    split_info = {}
    for split in ["train", "validation", "test"]:
        d = seqs[split]
        X, y = d["X"], d["y"]
        n_fraud = int(y.sum())
        n = len(y)
        log.info(
            "[%s] sequences=%d  fraud=%d (%.2f%%)  X.shape=%s",
            split, n, n_fraud, 100*n_fraud/n if n else 0, X.shape
        )
        split_info[split] = {"n": n, "n_fraud": n_fraud}

    X_train, y_train = seqs["train"]["X"], seqs["train"]["y"]
    X_val,   y_val   = seqs["validation"]["X"], seqs["validation"]["y"]
    # X_test intentionally NOT used in this script (STEP 9 only)

    # 4. DataLoaders
    sep("Building DataLoaders")
    train_loader = make_loader(X_train, y_train, batch_size=CONFIG.batch_size, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   batch_size=CONFIG.batch_size, shuffle=False)
    log.info("Train loader: %d batches", len(train_loader))
    log.info("Val   loader: %d batches", len(val_loader))

    # 5 + 6. Train (includes preflight inside)
    sep("Training (max %d epochs, early stop patience=%d)" % (CONFIG.max_epochs, CONFIG.early_stop_patience))
    summary, history = train(train_loader, val_loader, device, CONFIG)

    # 7. Save metadata
    METADATA_JSON.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "step":               "STEP 8 - E2 GRU Training",
        "seed":               SEED,
        "device":             str(device),
        "torch_version":      torch.__version__,
        "architecture": {
            "input_size":     CONFIG.input_size,
            "hidden_size":    CONFIG.hidden_size,
            "num_layers":     CONFIG.num_layers,
            "dropout":        CONFIG.dropout,
            "seq_len":        CONFIG.seq_len,
        },
        "training": {
            "optimizer":      CONFIG.optimizer,
            "learning_rate":  CONFIG.learning_rate,
            "batch_size":     CONFIG.batch_size,
            "max_epochs":     CONFIG.max_epochs,
            "early_stop_patience": CONFIG.early_stop_patience,
            "monitor":        CONFIG.early_stop_monitor,
            "loss":           CONFIG.loss_fn,
            "pos_weight":     CONFIG.pos_weight,
            "smote":          CONFIG.use_smote,
            "random_oversample": CONFIG.use_random_oversampling,
        },
        "dataset": split_info,
        "n_excluded_dup_dt": n_excl,
        "results": {
            "best_epoch":       summary["best_epoch"],
            "best_val_pr_auc":  summary["best_val_pr_auc"],
            "epochs_completed": summary["epochs_completed"],
            "early_stopped":    summary["early_stopped"],
            "total_time_s":     summary["total_time_s"],
        },
        "test_evaluated": False,
        "e1_modified":    False,
    }
    with open(METADATA_JSON, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("Metadata saved: %s", METADATA_JSON)

    # 8. Final printed summary
    sep("TRAINING COMPLETE — STEP 8 RESULTS")
    log.info("")
    log.info("  Epoch history:")
    log.info("  %-6s  %-12s  %-12s  %-12s  %-10s", "Epoch", "Train Loss", "Val Loss", "Val PR-AUC", "Time(s)")
    log.info("  " + "-"*58)
    for r in history:
        best_marker = " <-- BEST" if r.epoch == summary["best_epoch"] else ""
        log.info(
            "  %-6d  %-12.4f  %-12.4f  %-12.4f  %-10.1f%s",
            r.epoch, r.train_loss, r.val_loss, r.val_pr_auc, r.epoch_time, best_marker
        )
    log.info("")
    log.info("  Best epoch          : %d", summary["best_epoch"])
    log.info("  Best val PR-AUC     : %.4f", summary["best_val_pr_auc"])
    log.info("  Epochs completed    : %d / %d", summary["epochs_completed"], CONFIG.max_epochs)
    log.info("  Early stopped       : %s", summary["early_stopped"])
    log.info("  Total training time : %.1f s (%.1f min)", summary["total_time_s"], summary["total_time_s"]/60)
    log.info("  Checkpoint saved    : %s", CHECKPOINT_PATH)
    log.info("")
    log.info("  TEST SET: NOT evaluated (reserved for STEP 9)")
    log.info("  E1 artifacts: NOT modified")
    log.info("")
    log.info("STEP 8 COMPLETE. STOP. Awaiting STEP 9 approval.")
    sep()

    total_elapsed = time.time() - t_script_start
    log.info("Total script time: %.1f min", total_elapsed / 60)


if __name__ == "__main__":
    main()
