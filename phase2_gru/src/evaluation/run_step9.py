"""
Phase 2 — STEP 9: E2b Test Evaluation and E1 vs E2b Paired Comparison
=======================================================================
This is the DEFINITIVE final evaluation stage.

Protocol:
  1. Load frozen E1 test predictions (predictions.parquet)
  2. Load frozen E2b checkpoint (epoch 2) + frozen StandardScaler
  3. Generate E2b test sequences (same population as training)
  4. Apply frozen StandardScaler (train-fitted, NOT refitted)
  5. Run E2b inference → fraud probabilities
  6. Select E2b threshold from VALIDATION ONLY (F1-max)
  7. Compute full test metrics on the 76,520 COMMON TransactionIDs
  8. Restrict E1 predictions to the same 76,520 TransactionIDs
  9. Run 2,000-resample paired bootstrap on the common population
  10. Save all artifacts, write reports

CRITICAL RULES:
  - No refitting of any kind using test data
  - No threshold selection using test labels
  - Test labels used ONLY for final metric computation
  - E1 artifacts: NOT MODIFIED
  - E2a artifacts: NOT MODIFIED
  - E2b checkpoint: NOT MODIFIED
  - StandardScaler: NOT REFITTED

STOP CONDITION:
  Report results and stop. Do NOT proceed to STEP 10 automatically.
"""

import sys
import pathlib
import json
import logging
import time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger(__name__)

from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score,
    matthews_corrcoef, balanced_accuracy_score,
    confusion_matrix, brier_score_loss,
    precision_recall_curve,
)

from phase2_gru.src.utils.reproducibility import configure, SEED
from phase2_gru.src.training.training_config import CONFIG
from phase2_gru.src.data.sequence_builder import load_all_splits, build_sequences
from phase2_gru.src.data.preprocessing import SequenceStandardScaler
from phase2_gru.src.data.dataset import make_loader
from phase2_gru.src.models.gru_model import FraudGRU

# ── Paths ─────────────────────────────────────────────────────────────────────
E1_DIR         = pathlib.Path("experiments/E1_lightgbm")
E2B_DIR        = pathlib.Path("phase2_gru/artifacts/E2b_scaled")
OUT_DIR        = E2B_DIR
REPORTS_DIR    = pathlib.Path("phase2_gru/reports")

E1_PREDICTIONS = E1_DIR / "predictions.parquet"
E1_METRICS     = E1_DIR / "metrics.json"
E2B_CHECKPOINT = E2B_DIR / "model" / "gru_best.pt"
E2B_SCALER     = E2B_DIR / "preprocessing" / "standard_scaler.pkl"

SEP = "=" * 65
def sep(title=""):
    if title:
        log.info("%s", SEP)
        log.info("  %s", title)
        log.info("%s", SEP)
    else:
        log.info("%s", SEP)


# ── Helper: operational metrics ───────────────────────────────────────────────
def recall_at_fpr(y_true, y_prob, target_fpr):
    """Max recall achievable at or below target_fpr."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    eligible = tpr[fpr <= target_fpr]
    return float(eligible.max()) if len(eligible) > 0 else 0.0

def precision_at_topk(y_true, y_prob, k):
    """Precision among top-k highest-probability transactions."""
    idx = np.argsort(y_prob)[::-1][:k]
    return float(y_true[idx].mean())

def f1_max_threshold(y_true, y_prob):
    """Find threshold maximising F1 on given data."""
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    best_idx = np.argmax(f1_scores[:-1])   # len(thresholds) = len(prec)-1
    return float(thresholds[best_idx]), float(f1_scores[best_idx])

def compute_metrics(y_true, y_prob, threshold, split_name=""):
    """Compute full metric suite."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    pr_auc = average_precision_score(y_true, y_prob)
    roc    = roc_auc_score(y_true, y_prob)
    prec   = precision_score(y_true, y_pred, zero_division=0)
    rec    = recall_score(y_true, y_pred, zero_division=0)
    f1     = f1_score(y_true, y_pred, zero_division=0)
    mcc    = matthews_corrcoef(y_true, y_pred)
    bal    = balanced_accuracy_score(y_true, y_pred)
    brier  = brier_score_loss(y_true, y_prob)
    total  = len(y_true)
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr_val = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {
        "split": split_name,
        "n_samples": total,
        "n_fraud": int(y_true.sum()),
        "n_legit": int(total - y_true.sum()),
        "threshold": threshold,
        "pr_auc": round(pr_auc, 6),
        "roc_auc": round(roc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "mcc": round(mcc, 6),
        "balanced_accuracy": round(bal, 6),
        "brier_score": round(brier, 6),
        "fpr": round(fpr_val, 6),
        "fnr": round(fnr_val, 6),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }

def compute_operational(y_true, y_prob):
    return {
        "recall_at_fpr_0.001": round(recall_at_fpr(y_true, y_prob, 0.001), 6),
        "recall_at_fpr_0.005": round(recall_at_fpr(y_true, y_prob, 0.005), 6),
        "recall_at_fpr_0.010": round(recall_at_fpr(y_true, y_prob, 0.010), 6),
        "recall_at_fpr_0.020": round(recall_at_fpr(y_true, y_prob, 0.020), 6),
        "precision_at_top_100":  round(precision_at_topk(y_true, y_prob, 100), 6),
        "precision_at_top_500":  round(precision_at_topk(y_true, y_prob, 500), 6),
        "precision_at_top_1000": round(precision_at_topk(y_true, y_prob, 1000), 6),
        "precision_at_top_5000": round(precision_at_topk(y_true, y_prob, 5000), 6),
    }

# ── Paired bootstrap ──────────────────────────────────────────────────────────
def paired_bootstrap(y_true, prob_e1, prob_e2b,
                     n_boot=2000, seed=42, metric_fn=average_precision_score):
    """
    Paired bootstrap CI for Delta = metric(E2b) - metric(E1).
    Same indices resampled for both models on every iteration.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue   # skip degenerate sample
        m_e1  = metric_fn(yt, prob_e1[idx])
        m_e2b = metric_fn(yt, prob_e2b[idx])
        deltas.append(m_e2b - m_e1)
    deltas = np.array(deltas)
    return {
        "n_valid": len(deltas),
        "observed_delta": float(metric_fn(y_true, prob_e2b) - metric_fn(y_true, prob_e1)),
        "mean_delta": float(deltas.mean()),
        "std_delta":  float(deltas.std()),
        "ci_lower_95": float(np.percentile(deltas, 2.5)),
        "ci_upper_95": float(np.percentile(deltas, 97.5)),
        "ci_excludes_zero": bool(
            np.percentile(deltas, 2.5) > 0 or np.percentile(deltas, 97.5) < 0
        ),
    }


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    sep("STEP 9 - E2b TEST EVALUATION + E1 vs E2b PAIRED COMPARISON")
    log.info("CRITICAL: No refitting. No test-driven decisions. Single test evaluation.")

    device = configure(SEED)
    if device.type != "cuda":
        log.error("CUDA unavailable. Stopping.")
        sys.exit(1)
    log.info("Device: %s (%s)", device, torch.cuda.get_device_name(0))

    # ── 1. Load frozen E1 predictions ────────────────────────────────────────
    sep("1. Loading frozen E1 predictions")
    e1_preds = pd.read_parquet(E1_PREDICTIONS)
    e1_test  = e1_preds[e1_preds["split"] == "test"].copy()
    e1_val   = e1_preds[e1_preds["split"] == "validation"].copy()
    log.info("E1 test rows: %d  (fraud=%d)", len(e1_test), e1_test["isFraud"].sum())
    log.info("E1 val  rows: %d  (fraud=%d)", len(e1_val),  e1_val["isFraud"].sum())

    with open(E1_METRICS) as f:
        e1_metrics_frozen = json.load(f)
    e1_frozen_threshold = e1_metrics_frozen["test"]["threshold"]
    log.info("E1 frozen threshold: %.6f", e1_frozen_threshold)

    # ── 2. Load frozen E2b checkpoint + scaler ────────────────────────────────
    sep("2. Loading frozen E2b checkpoint (epoch 2) and StandardScaler")
    ckpt = torch.load(E2B_CHECKPOINT, map_location=device, weights_only=False)
    log.info("E2b checkpoint epoch: %d  |  val_PR_AUC: %.4f",
             ckpt["epoch"], ckpt["val_pr_auc"])
    assert ckpt["epoch"] == 2, f"Expected epoch 2, got {ckpt['epoch']}"

    model = FraudGRU(
        input_size  = ckpt["config"]["input_size"],
        hidden_size = ckpt["config"]["hidden_size"],
        num_layers  = ckpt["config"]["num_layers"],
        dropout     = ckpt["config"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log.info("E2b model loaded: %d parameters", sum(p.numel() for p in model.parameters()))

    scaler = SequenceStandardScaler.load(E2B_SCALER)
    log.info("Scaler loaded: fitted=%s, features=%d", scaler.is_fitted, scaler.n_features)
    log.info("Authoritative scaler statistics (full train fit):")
    log.info("  scale > 100 : %d features", int((scaler.scale_ > 100).sum()))
    log.info("  scale 1-100 : %d features", int(((scaler.scale_ >= 1) & (scaler.scale_ <= 100)).sum()))
    log.info("  scale < 1   : %d features", int((scaler.scale_ < 1).sum()))
    log.info("  near-zero-var (scale=1.0): %d features",
             int(np.sum((scaler.scale_ == 1.0) & (scaler.mean_ != 0) |
                        (scaler.scale_ == 1.0) & (scaler.mean_ == 0))))

    # ── 3. Generate E2b sequences ─────────────────────────────────────────────
    sep("3. Generating E2b sequences (same population as E2b training)")
    df = load_all_splits()
    log.info("Loaded %d total transactions", len(df))
    seqs = build_sequences(df, smoke_test_n=None, verbose=True)
    n_excl = seqs["_meta"]["n_excluded_dup_dt"]
    log.info("Excluded (duplicate DT): %d", n_excl)

    # Validation sequences (needed for threshold selection)
    X_val_raw = seqs["validation"]["X"]
    y_val     = seqs["validation"]["y"]
    val_tids  = seqs["validation"]["target_tid"]

    # Test sequences
    X_test_raw = seqs["test"]["X"]
    y_test     = seqs["test"]["y"]
    test_tids  = seqs["test"]["target_tid"]

    log.info("E2b val  sequences: %d  (fraud=%d)", len(y_val),  int(y_val.sum()))
    log.info("E2b test sequences: %d  (fraud=%d)", len(y_test), int(y_test.sum()))

    # ── 4. Apply frozen StandardScaler ───────────────────────────────────────
    sep("4. Applying frozen train-fitted StandardScaler")
    X_val_scaled  = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    assert not np.any(np.isnan(X_val_scaled)),  "NaN in scaled val"
    assert not np.any(np.isinf(X_val_scaled)),  "Inf in scaled val"
    assert not np.any(np.isnan(X_test_scaled)), "NaN in scaled test"
    assert not np.any(np.isinf(X_test_scaled)), "Inf in scaled test"
    log.info("Post-scale NaN/Inf: PASS (val and test)")

    # ── 5. E2b inference ──────────────────────────────────────────────────────
    sep("5. E2b inference (eval mode, sigmoid probabilities)")
    def run_inference(X_scaled, y, batch_size=512):
        loader = make_loader(X_scaled, y, batch_size=batch_size, shuffle=False)
        all_proba = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(device)
                logits = model(xb)
                proba  = torch.sigmoid(logits).cpu().numpy()
                all_proba.append(proba)
        proba = np.concatenate(all_proba).ravel()
        assert np.all(proba >= 0) and np.all(proba <= 1), "Probabilities out of [0,1]"
        return proba

    prob_val  = run_inference(X_val_scaled,  y_val)
    prob_test = run_inference(X_test_scaled, y_test)
    log.info("Val  inference done:  proba range [%.4f, %.4f]", prob_val.min(),  prob_val.max())
    log.info("Test inference done:  proba range [%.4f, %.4f]", prob_test.min(), prob_test.max())

    # ── 6. E2b threshold selection (validation only) ──────────────────────────
    sep("6. E2b threshold selection — validation only (F1-max)")
    e2b_threshold, e2b_val_f1 = f1_max_threshold(y_val, prob_val)
    log.info("E2b selected threshold: %.6f  (val F1=%.4f)", e2b_threshold, e2b_val_f1)
    log.info("E2b val PR-AUC: %.4f", average_precision_score(y_val, prob_val))
    log.info("THRESHOLD FROZEN. No further test-based selection.")

    # ── 7. Common test population: intersection ────────────────────────────────
    sep("7. Establishing common test population (76,520 TransactionIDs)")
    e1_test_ids  = set(e1_test["TransactionID"].values)
    e2b_test_ids = set(test_tids)
    common_ids   = sorted(e1_test_ids & e2b_test_ids)
    log.info("E1  test TransactionIDs: %d", len(e1_test_ids))
    log.info("E2b test TransactionIDs: %d", len(e2b_test_ids))
    log.info("Common intersection    : %d", len(common_ids))
    log.info("E1-only                : %d", len(e1_test_ids - e2b_test_ids))
    log.info("E2b coverage           : %.2f%%", 100 * len(common_ids) / len(e1_test_ids))

    if len(common_ids) != 76520:
        log.error("STOP: expected 76,520 common IDs, got %d", len(common_ids))
        sys.exit(1)
    log.info("Common population verified: 76,520 TransactionIDs")

    # Save common ID list
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    common_ids_path = OUT_DIR / "common_test_transaction_ids.txt"
    with open(common_ids_path, "w") as f:
        f.write("\n".join(str(x) for x in common_ids))
    log.info("Common IDs saved: %s", common_ids_path)

    # ── 8. Align predictions to common population ─────────────────────────────
    sep("8. Aligning E1 and E2b to common 76,520 population")
    common_set = set(common_ids)

    # E1 common
    e1_common = e1_test[e1_test["TransactionID"].isin(common_set)].sort_values("TransactionID")
    assert len(e1_common) == 76520, f"E1 common: {len(e1_common)}"

    # E2b common: build mapping tid → prob
    tid_to_prob_e2b = dict(zip(test_tids, prob_test))
    tid_to_y_e2b    = dict(zip(test_tids, y_test))

    e2b_common_prob = np.array([tid_to_prob_e2b[tid] for tid in common_ids])
    e2b_common_y    = np.array([tid_to_y_e2b[tid]    for tid in common_ids])
    e1_common_prob  = e1_common["fraud_probability"].values
    e1_common_y     = e1_common["isFraud"].values

    # Verify labels match
    assert np.array_equal(e1_common_y, e2b_common_y), \
        "CRITICAL: E1 and E2b labels do not match for common population!"
    log.info("Label alignment: VERIFIED (E1 and E2b labels identical for 76,520 IDs)")

    y_common = e1_common_y  # ground truth (same for both)
    log.info("Common population: %d fraud, %d legit", int(y_common.sum()), int((1-y_common).sum()))

    # ── 9. Full test metrics ───────────────────────────────────────────────────
    sep("9. Computing full test metrics on 76,520 common population")

    # E1 on common population (frozen threshold)
    e1_metrics_common = compute_metrics(y_common, e1_common_prob, e1_frozen_threshold, "E1_common")
    e1_ops_common     = compute_operational(y_common, e1_common_prob)

    # E2b on common population (val-selected threshold)
    e2b_metrics_common = compute_metrics(y_common, e2b_common_prob, e2b_threshold, "E2b_common")
    e2b_ops_common     = compute_operational(y_common, e2b_common_prob)

    # E1 on FULL test population (for documentation)
    e1_metrics_full = compute_metrics(
        e1_test["isFraud"].values,
        e1_test["fraud_probability"].values,
        e1_frozen_threshold, "E1_full"
    )

    log.info("")
    log.info("  %-30s  %-12s  %-12s", "Metric", "E1_common", "E2b_common")
    log.info("  " + "-"*56)
    for k in ["pr_auc", "roc_auc", "precision", "recall", "f1", "mcc",
              "balanced_accuracy", "brier_score", "fpr", "fnr"]:
        log.info("  %-30s  %-12.4f  %-12.4f",
                 k, e1_metrics_common.get(k, 0), e2b_metrics_common.get(k, 0))

    # ── 10. Paired bootstrap ──────────────────────────────────────────────────
    sep("10. Paired bootstrap (2,000 resamples, seed=42)")
    log.info("Primary metric: Delta PR-AUC = PR-AUC(E2b) - PR-AUC(E1)")

    boot_pr_auc = paired_bootstrap(y_common, e1_common_prob, e2b_common_prob,
                                   n_boot=2000, seed=42, metric_fn=average_precision_score)
    boot_roc    = paired_bootstrap(y_common, e1_common_prob, e2b_common_prob,
                                   n_boot=2000, seed=42, metric_fn=roc_auc_score)

    log.info("Paired bootstrap PR-AUC:")
    log.info("  Observed E1  PR-AUC : %.4f", e1_metrics_common["pr_auc"])
    log.info("  Observed E2b PR-AUC : %.4f", e2b_metrics_common["pr_auc"])
    log.info("  Observed Delta      : %.4f", boot_pr_auc["observed_delta"])
    log.info("  95%% CI             : [%.4f, %.4f]",
             boot_pr_auc["ci_lower_95"], boot_pr_auc["ci_upper_95"])
    log.info("  CI excludes zero    : %s", boot_pr_auc["ci_excludes_zero"])
    log.info("  n_valid resamples   : %d", boot_pr_auc["n_valid"])

    # ── 11. Calibration ───────────────────────────────────────────────────────
    from sklearn.calibration import calibration_curve
    def ece_score(y_true, y_prob, n_bins=10):
        """Expected Calibration Error."""
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
            if mask.sum() == 0:
                continue
            acc  = y_true[mask].mean()
            conf = y_prob[mask].mean()
            ece += mask.sum() * abs(acc - conf)
        return ece / len(y_true)

    e1_ece  = ece_score(y_common, e1_common_prob)
    e2b_ece = ece_score(y_common, e2b_common_prob)
    log.info("ECE: E1=%.4f  E2b=%.4f", e1_ece, e2b_ece)

    # ── 12. Save all artifacts ────────────────────────────────────────────────
    sep("12. Saving artifacts")

    # Test predictions parquet
    pred_df = pd.DataFrame({
        "TransactionID":      common_ids,
        "isFraud":            y_common,
        "e1_fraud_probability":   e1_common_prob,
        "e2b_fraud_probability":  e2b_common_prob,
        "e1_prediction":  (e1_common_prob  >= e1_frozen_threshold).astype(int),
        "e2b_prediction": (e2b_common_prob >= e2b_threshold).astype(int),
    })
    pred_path = OUT_DIR / "test_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)
    log.info("Predictions saved: %s", pred_path)

    # Test metrics JSON
    test_metrics_out = {
        "step": "STEP 9",
        "common_population": 76520,
        "e1_only": len(e1_test_ids - e2b_test_ids),
        "e1_threshold_frozen": e1_frozen_threshold,
        "e2b_threshold_val_selected": e2b_threshold,
        "e2b_val_f1_at_threshold": e2b_val_f1,
        "e1_common": e1_metrics_common,
        "e2b_common": e2b_metrics_common,
        "e1_full_test": e1_metrics_full,
        "e1_operational_common": e1_ops_common,
        "e2b_operational_common": e2b_ops_common,
        "calibration": {
            "e1_brier": e1_metrics_common["brier_score"],
            "e2b_brier": e2b_metrics_common["brier_score"],
            "e1_ece": round(e1_ece, 6),
            "e2b_ece": round(e2b_ece, 6),
        },
    }
    metrics_path = OUT_DIR / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(test_metrics_out, f, indent=2)
    log.info("Metrics saved: %s", metrics_path)

    # Bootstrap JSON
    bootstrap_out = {
        "n_resamples_target": 2000,
        "seed": 42,
        "common_population": 76520,
        "primary_metric": "PR-AUC",
        "convention": "Delta = PR-AUC(E2b) - PR-AUC(E1)",
        "paired_bootstrap_pr_auc": boot_pr_auc,
        "paired_bootstrap_roc_auc": boot_roc,
        "interpretation_note": (
            "CI excludes zero means the paired bootstrap consistently finds "
            "one model superior on this metric across resamples. "
            "This does not establish causality."
        ),
    }
    boot_path = OUT_DIR / "paired_bootstrap.json"
    with open(boot_path, "w") as f:
        json.dump(bootstrap_out, f, indent=2)
    log.info("Bootstrap saved: %s", boot_path)

    # Integrity audit JSON
    import hashlib
    def file_hash(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read())
        return h.hexdigest()[:16]

    integrity = {
        "step": "STEP 9 integrity audit",
        "e1_predictions_hash":  file_hash(E1_PREDICTIONS),
        "e2b_checkpoint_hash":  file_hash(E2B_CHECKPOINT),
        "e2b_scaler_hash":      file_hash(E2B_SCALER),
        "e1_metrics_hash":      file_hash(E1_METRICS),
        "seed": SEED,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "common_population": len(common_ids),
        "labels_verified": True,
        "scaler_refitted": False,
        "test_labels_used_for": "final metric computation only",
        "threshold_selected_on": "validation only (F1-max)",
        "e1_modified": False,
        "e2a_modified": False,
        "e2b_checkpoint_modified": False,
    }
    integrity_path = REPORTS_DIR / "STEP9_integrity_audit.json"
    with open(integrity_path, "w") as f:
        json.dump(integrity, f, indent=2)
    log.info("Integrity audit saved: %s", integrity_path)

    # ── 13. Final summary log ─────────────────────────────────────────────────
    elapsed = time.time() - t0
    sep("STEP 9 COMPLETE — FINAL RESULTS")
    log.info("")
    log.info("  POPULATION")
    log.info("  E1 full test              : %d", len(e1_test))
    log.info("  E2b eligible test         : %d", len(y_test))
    log.info("  Common (paired)           : %d", len(common_ids))
    log.info("  E1-only                   : %d", len(e1_test_ids - e2b_test_ids))
    log.info("  E2b coverage              : %.2f%%", 100*len(common_ids)/len(e1_test))
    log.info("")
    log.info("  THRESHOLDS")
    log.info("  E1 (frozen)               : %.6f", e1_frozen_threshold)
    log.info("  E2b (val F1-max)          : %.6f", e2b_threshold)
    log.info("")
    log.info("  TEST METRICS — COMMON 76,520 POPULATION")
    log.info("  %-35s  %-12s  %-12s", "Metric", "E1", "E2b")
    log.info("  " + "-"*60)
    for k in ["pr_auc", "roc_auc", "f1", "precision", "recall",
              "mcc", "balanced_accuracy", "brier_score", "fpr", "fnr"]:
        log.info("  %-35s  %-12.4f  %-12.4f",
                 k, e1_metrics_common[k], e2b_metrics_common[k])
    log.info("")
    log.info("  CONFUSION MATRIX — E1 (common 76,520)")
    log.info("  TP=%d  FP=%d  TN=%d  FN=%d",
             e1_metrics_common["tp"], e1_metrics_common["fp"],
             e1_metrics_common["tn"], e1_metrics_common["fn"])
    log.info("  CONFUSION MATRIX — E2b (common 76,520)")
    log.info("  TP=%d  FP=%d  TN=%d  FN=%d",
             e2b_metrics_common["tp"], e2b_metrics_common["fp"],
             e2b_metrics_common["tn"], e2b_metrics_common["fn"])
    log.info("")
    log.info("  OPERATIONAL METRICS — E1 vs E2b (common 76,520)")
    for k in e1_ops_common:
        log.info("  %-35s  %-12.4f  %-12.4f", k,
                 e1_ops_common[k], e2b_ops_common[k])
    log.info("")
    log.info("  PAIRED BOOTSTRAP — PR-AUC (2,000 resamples)")
    log.info("  E1  PR-AUC        : %.4f", e1_metrics_common["pr_auc"])
    log.info("  E2b PR-AUC        : %.4f", e2b_metrics_common["pr_auc"])
    log.info("  Delta             : %.4f", boot_pr_auc["observed_delta"])
    log.info("  95%% CI           : [%.4f, %.4f]",
             boot_pr_auc["ci_lower_95"], boot_pr_auc["ci_upper_95"])
    log.info("  CI excludes zero  : %s", boot_pr_auc["ci_excludes_zero"])
    log.info("")
    log.info("  CALIBRATION")
    log.info("  Brier E1   : %.4f", e1_metrics_common["brier_score"])
    log.info("  Brier E2b  : %.4f", e2b_metrics_common["brier_score"])
    log.info("  ECE   E1   : %.4f", e1_ece)
    log.info("  ECE   E2b  : %.4f", e2b_ece)
    log.info("")
    log.info("  ARTIFACTS")
    log.info("  %s", pred_path)
    log.info("  %s", metrics_path)
    log.info("  %s", boot_path)
    log.info("  %s", integrity_path)
    log.info("  %s", common_ids_path)
    log.info("")
    log.info("STEP 9 COMPLETE. STOP. Awaiting review before STEP 10.")
    sep()
    log.info("Total STEP 9 time: %.1f min", elapsed / 60)


if __name__ == "__main__":
    main()
