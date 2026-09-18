"""
Phase 2 — STEP 8 E2b: Scaled GRU Training Runner
===================================================
E2b = same E2 sequences as E2a + train-fitted StandardScaler preprocessing.

E2b is a separately documented preprocessing variant.
It does NOT change:
  - sequence construction or sequence population
  - 406-feature contract or feature ordering
  - GRU architecture or hyperparameters
  - optimizer, lr, batch_size, loss, pos_weight, seed, early stopping

Purpose: determine whether appropriate numerical conditioning enables
the same locked GRU architecture to learn useful temporal patterns.

E2a result (reference):
  Best validation PR-AUC = 0.0345  (near-random baseline ~0.0339)

STOP CONDITION:
  This script does NOT evaluate the test set.
  STEP 9 handles test evaluation after research decision.
"""

import sys
import pathlib
import json
import logging
import time
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger(__name__)

from phase2_gru.src.utils.reproducibility import configure, SEED
from phase2_gru.src.training.training_config import CONFIG
from phase2_gru.src.data.sequence_builder import load_all_splits, build_sequences
from phase2_gru.src.data.preprocessing import SequenceStandardScaler
from phase2_gru.src.data.dataset import make_loader
from phase2_gru.src.training.trainer import train

# E2b artifact paths
E2B_DIR       = pathlib.Path("phase2_gru/artifacts/E2b_scaled")
CHECKPOINT    = E2B_DIR / "model" / "gru_best.pt"
HISTORY_CSV   = E2B_DIR / "training_history.csv"
METADATA_JSON = E2B_DIR / "run_metadata.json"

# Override trainer artifact paths for E2b
import phase2_gru.src.training.trainer as _trainer_module
_trainer_module.CHECKPOINT_PATH = CHECKPOINT
_trainer_module.HISTORY_CSV     = HISTORY_CSV
_trainer_module.METADATA_JSON   = METADATA_JSON

SEP = "=" * 65

def sep(title=""):
    if title:
        log.info("%s", SEP)
        log.info("  %s", title)
        log.info("%s", SEP)
    else:
        log.info("%s", SEP)


def main():
    t_script_start = time.time()

    sep("STEP 8 E2b - SCALED GRU TRAINING")
    log.info("Variant: E2b (train-fitted StandardScaler preprocessing)")
    log.info("E2a reference val PR-AUC = 0.0345 (near-random baseline ~0.0339)")
    log.info("Seed: %d", SEED)

    # Reproducibility + device
    device = configure(SEED)
    log.info("PyTorch version: %s", torch.__version__)
    log.info("Device: %s  (%s)", device,
             torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        log.info("VRAM: %.2f GB", props.total_memory / 1e9)
        log.info("CUDA version: %s", torch.version.cuda)

    # Abort if CUDA unavailable
    if device.type != "cuda":
        log.error("CUDA unavailable. Training not launched. Check GPU/driver.")
        sys.exit(1)

    # Generate sequences (same population as E2a)
    sep("Generating E2 sequences (Option A, strict DT ordering)")
    log.info("Loading all parquets...")
    df = load_all_splits()
    log.info("Rows loaded: %d", len(df))
    log.info("Building sequences (this takes ~17 min)...")
    seqs = build_sequences(df, smoke_test_n=None, verbose=True)
    n_excl = seqs["_meta"]["n_excluded_dup_dt"]
    log.info("Sequences built. Excluded (duplicate DT): %d", n_excl)

    # Extract raw arrays
    X_train_raw, y_train = seqs["train"]["X"],      seqs["train"]["y"]
    X_val_raw,   y_val   = seqs["validation"]["X"], seqs["validation"]["y"]
    # X_test extracted but NOT scaled or used during training
    X_test_raw,  y_test  = seqs["test"]["X"],       seqs["test"]["y"]

    sep("Data summary (before scaling)")
    for split, X, y in [("train", X_train_raw, y_train),
                        ("validation", X_val_raw, y_val),
                        ("test", X_test_raw, y_test)]:
        n_fraud = int(y.sum())
        log.info("[%s] sequences=%d  fraud=%d (%.2f%%)  X.shape=%s",
                 split, len(y), n_fraud, 100*n_fraud/len(y), X.shape)

    # Fit scaler on training sequences ONLY
    sep("Fitting StandardScaler on training sequences only")
    scaler = SequenceStandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    log.info(scaler.summary())

    # Transform validation (using frozen train stats)
    X_val_scaled = scaler.transform(X_val_raw)
    log.info("Validation transformed using train-fitted scaler.")
    log.info("Train scaled: NaN=%s  Inf=%s", False, False)

    # Save scaler
    scaler.save()
    log.info("Scaler saved.")

    # Verify no NaN/Inf after scaling
    import numpy as np
    assert not np.any(np.isnan(X_train_scaled)), "NaN in scaled train X"
    assert not np.any(np.isinf(X_train_scaled)), "Inf in scaled train X"
    assert not np.any(np.isnan(X_val_scaled)),   "NaN in scaled val X"
    assert not np.any(np.isinf(X_val_scaled)),   "Inf in scaled val X"
    log.info("Post-scale NaN/Inf check: PASS")

    # DataLoaders
    sep("Building DataLoaders (scaled)")
    train_loader = make_loader(X_train_scaled, y_train, batch_size=CONFIG.batch_size, shuffle=True)
    val_loader   = make_loader(X_val_scaled,   y_val,   batch_size=CONFIG.batch_size, shuffle=False)
    log.info("Train loader: %d batches", len(train_loader))
    log.info("Val   loader: %d batches", len(val_loader))

    # Train
    sep("Training E2b (max %d epochs, patience=%d)" % (CONFIG.max_epochs, CONFIG.early_stop_patience))
    (E2B_DIR / "model").mkdir(parents=True, exist_ok=True)
    summary, history = train(train_loader, val_loader, device, CONFIG)

    # Save metadata
    split_info = {
        "train":      {"n": len(y_train), "n_fraud": int(y_train.sum())},
        "validation": {"n": len(y_val),   "n_fraud": int(y_val.sum())},
        "test":       {"n": len(y_test),  "n_fraud": int(y_test.sum())},
    }
    metadata = {
        "variant":        "E2b_scaled",
        "step":           "STEP 8 E2b - Scaled GRU Training",
        "seed":           SEED,
        "device":         str(device),
        "gpu_name":       torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "torch_version":  torch.__version__,
        "cuda_version":   torch.version.cuda,
        "preprocessing":  "StandardScaler (fit on train X only)",
        "architecture": {
            "input_size":  CONFIG.input_size,
            "hidden_size": CONFIG.hidden_size,
            "num_layers":  CONFIG.num_layers,
            "dropout":     CONFIG.dropout,
            "seq_len":     CONFIG.seq_len,
        },
        "training": {
            "optimizer":          CONFIG.optimizer,
            "learning_rate":      CONFIG.learning_rate,
            "batch_size":         CONFIG.batch_size,
            "max_epochs":         CONFIG.max_epochs,
            "early_stop_patience":CONFIG.early_stop_patience,
            "monitor":            CONFIG.early_stop_monitor,
            "loss":               CONFIG.loss_fn,
            "pos_weight":         CONFIG.pos_weight,
        },
        "dataset":           split_info,
        "n_excluded_dup_dt": n_excl,
        "e2a_reference": {
            "best_val_pr_auc": 0.0345,
            "epochs_completed": 6,
            "preprocessing": "none (unscaled)",
        },
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
    log.info("E2b metadata saved: %s", METADATA_JSON)

    # Final summary
    sep("E2b TRAINING COMPLETE")
    log.info("")
    log.info("  E2b Epoch history:")
    log.info("  %-6s  %-12s  %-12s  %-12s  %-10s",
             "Epoch", "Train Loss", "Val Loss", "Val PR-AUC", "Time(s)")
    log.info("  " + "-"*58)
    for r in history:
        best_marker = " <-- BEST" if r.epoch == summary["best_epoch"] else ""
        log.info("  %-6d  %-12.4f  %-12.4f  %-12.4f  %-10.1f%s",
                 r.epoch, r.train_loss, r.val_loss, r.val_pr_auc,
                 r.epoch_time, best_marker)
    log.info("")
    log.info("  E2a (unscaled) best val PR-AUC : 0.0345")
    log.info("  E2b (scaled)   best val PR-AUC : %.4f", summary["best_val_pr_auc"])
    log.info("  Best epoch     : %d", summary["best_epoch"])
    log.info("  Epochs done    : %d / %d", summary["epochs_completed"], CONFIG.max_epochs)
    log.info("  Early stopped  : %s", summary["early_stopped"])
    log.info("  Training time  : %.1f s (%.1f min)", summary["total_time_s"], summary["total_time_s"]/60)
    log.info("  Checkpoint     : %s", CHECKPOINT)
    log.info("")
    log.info("  TEST SET: NOT evaluated (reserved for STEP 9)")
    log.info("  E1 artifacts  : NOT modified")
    log.info("  E2a artifacts : NOT modified")
    log.info("")
    log.info("E2b COMPLETE. STOP. Awaiting STEP 9 approval.")
    sep()

    total_elapsed = time.time() - t_script_start
    log.info("Total E2b script time: %.1f min", total_elapsed / 60)


if __name__ == "__main__":
    main()
