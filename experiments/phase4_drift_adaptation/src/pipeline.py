"""
pipeline.py — Phase 4 Master Experiment Pipeline
==================================================
Orchestrates the complete sequential adaptation experiment.

EXECUTION ORDER (protocol-mandated, cannot be reordered):
  Step 1  — Load + verify dataset
  Step 2  — Build temporal splits
  Step 3  — Train v1 (months 0-3)
  Step 4  — Month 4: select and FREEZE threshold using v1 + Month-4
  Step 5  — Build PSI reference (months 0-3, never reset)
  Step 6  — Month 5: PSI monitoring window 1
  Step 7  — Conditional v2 retraining (if month-5 triggered)
  Step 8  — Month 6: evaluate static vs adaptive
  Step 9  — Month 6: PSI monitoring window 2 (AFTER evaluation)
  Step 10 — Conditional v3 retraining (if month-6 triggered)
  Step 11 — Month 7: PROTECTED final evaluation (exactly once)
  Step 12 — Paired bootstrap for both evaluation periods
  Step 13 — Save all artifacts, predictions, metrics
  Step 14 — Final audit log

INVARIANT: Month 7 is never touched before Step 11.
"""
from __future__ import annotations
import hashlib, json, pathlib, sys, io, time
import numpy as np
import pandas as pd

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Add src dir to path
SRC_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

from data_loader  import (load_baf, get_temporal_splits, split_Xy,
                           verify_period_counts, FEATURE_COLS,
                           NUMERICAL_COLS, CATEGORICAL_COLS, TARGET_COL, TIME_COL)
from model        import train_model, predict_proba, LOCKED_PARAMS
from psi          import PSIDetector
from threshold    import select_threshold, load_frozen_threshold, apply_threshold
from evaluation   import compare_periods, compute_metrics
from bootstrap    import paired_bootstrap_pr_auc, format_bootstrap_result

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT    = pathlib.Path(__file__).resolve().parents[3]
PHASE4  = ROOT / "experiments/phase4_drift_adaptation"
ARTS    = PHASE4 / "artifacts"
MODELS  = ARTS / "models"
DRIFT   = ARTS / "drift"
THRESH  = ARTS / "thresholds"
PREDS   = ARTS / "predictions"
METRICS = ARTS / "metrics"
LOGS    = PHASE4 / "logs"

for d in [MODELS, DRIFT, THRESH, PREDS, METRICS, LOGS]:
    d.mkdir(parents=True, exist_ok=True)

BAF_PATH = ROOT / "Datasets/BAF/Base.csv"
RUN_ID   = time.strftime("%Y%m%d_%H%M%S")
LOG_PATH = LOGS / f"run_{RUN_ID}.log"

log_lines: list[str] = []

def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    log_lines.append(line)

def save_json(obj: dict | list, path: pathlib.Path) -> str:
    """Save JSON and return SHA256 hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, default=str)
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()

def save_preds(proba: np.ndarray, y_true: np.ndarray, name: str) -> str:
    """Save predictions and return hash."""
    arr = np.stack([y_true, proba], axis=1)
    p   = PREDS / f"{name}.npy"
    np.save(str(p), arr)
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    return h


# ===========================================================================
#  STEP 1 — LOAD + VERIFY DATASET
# ===========================================================================
log("=" * 60)
log("PHASE 4 EXPERIMENT PIPELINE — Protocol v1.1")
log(f"Run ID: {RUN_ID}")
log("=" * 60)

log("STEP 1 — Loading BAF Base.csv")
df = load_baf(BAF_PATH, verify=True)
log(f"  Loaded: {len(df):,} rows, {len(df.columns)} columns")


# ===========================================================================
#  STEP 2 — TEMPORAL SPLITS
# ===========================================================================
log("STEP 2 — Building locked temporal splits (no random split)")
splits = get_temporal_splits(df)
verify_period_counts(splits)
log("  All period counts verified against protocol.")

train_df = splits["train"]    # Months 0-3
val_df   = splits["val"]      # Month 4 (threshold ONLY)
mon1_df  = splits["monitor1"] # Month 5
eval1_df = splits["eval1"]    # Month 6
eval2_df = splits["eval2"]    # Month 7 (PROTECTED — do not touch until Step 11)

X_train, y_train = split_Xy(train_df)
X_val,   y_val   = split_Xy(val_df)
X_mon1,  y_mon1  = split_Xy(mon1_df)
X_eval1, y_eval1 = split_Xy(eval1_df)
# X_eval2, y_eval2 intentionally deferred until Step 11


# ===========================================================================
#  STEP 3 — TRAIN v1 (months 0-3)
# ===========================================================================
log("STEP 3 — Training Model v1 on months 0-3")
v1_result = train_model(
    X_train, y_train,
    version_label="v1",
    training_months=[0, 1, 2, 3],
    output_dir=MODELS,
)
# Static model gets its own frozen copy (same object — v1 never retrained)
static_model = v1_result
adaptive_model = v1_result  # will be replaced by v2/v3 if triggered
adaptive_version = "v1"
log(f"  v1 trained. Hash: {v1_result['model_hash'][:16]}")


# ===========================================================================
#  STEP 4 — MONTH 4: THRESHOLD SELECTION + FREEZE
# ===========================================================================
log("STEP 4 — Month 4: Threshold selection (v1 predictions, PERMANENT FREEZE)")
val_proba_v1 = predict_proba(v1_result, X_val)
thresh_result = select_threshold(
    y_val=y_val.values,
    proba_val=val_proba_v1,
    validation_period=4,
    model_version="v1",
    output_path=THRESH / "frozen_threshold_v1.json",
)
FROZEN_THRESHOLD = thresh_result["threshold"]
log(f"  Threshold FROZEN: {FROZEN_THRESHOLD:.6f} (val F1={thresh_result['validation_f1']:.4f})")
log("  This threshold will NOT change for v2 or v3.")


# ===========================================================================
#  STEP 5 — BUILD PSI REFERENCE (months 0-3, never reset)
# ===========================================================================
log("STEP 5 — Building PSI reference distribution (months 0-3, fixed forever)")
psi_detector = PSIDetector(
    numerical_cols=NUMERICAL_COLS,
    categorical_cols=CATEGORICAL_COLS,
)
psi_detector.fit(X_train)  # fit() raises if called a second time
psi_detector.save_reference(DRIFT / "psi_reference_months0_3.json")
log("  PSI reference built and saved (months 0-3 ONLY).")


# ===========================================================================
#  STEP 6 — MONTH 5: PSI MONITORING WINDOW 1
# ===========================================================================
log("STEP 6 — Month 5: PSI drift monitoring (Window 1)")
psi_result_m5 = psi_detector.compute(X_mon1, window_label="month5")
save_json(psi_result_m5, DRIFT / "psi_month5.json")
log(f"  Month 5 PSI: {psi_result_m5['triggered_count']}/{psi_result_m5['feature_denominator']} "
    f"features >= {psi_result_m5['psi_threshold']} | Triggered: {psi_result_m5['triggered']}")
if psi_result_m5["triggered"]:
    top5 = sorted(psi_result_m5["psi_values"].items(), key=lambda x: -x[1])[:5]
    log(f"  Top shifted features: {[(f, round(v,4)) for f,v in top5]}")


# ===========================================================================
#  STEP 7 — CONDITIONAL v2 RETRAINING
# ===========================================================================
v2_result = None
if psi_result_m5["triggered"]:
    log("STEP 7 — Month 5 TRIGGERED: Training v2 on months 0-5")
    train_0_5 = df[df[TIME_COL].isin([0, 1, 2, 3, 4, 5])].reset_index(drop=True)
    X_train_v2, y_train_v2 = split_Xy(train_0_5)
    v2_result = train_model(
        X_train_v2, y_train_v2,
        version_label="v2",
        training_months=[0, 1, 2, 3, 4, 5],
        output_dir=MODELS,
    )
    adaptive_model   = v2_result
    adaptive_version = "v2"
    log(f"  v2 trained. Hash: {v2_result['model_hash'][:16]}")
    log("  Threshold remains FROZEN (not re-selected for v2).")
else:
    log("STEP 7 — Month 5 NOT triggered. Adaptive model remains v1.")


# ===========================================================================
#  STEP 8 — MONTH 6: EVALUATION 1 (static vs adaptive)
# ===========================================================================
log("STEP 8 — Month 6: Evaluation (static v1 vs adaptive {})".format(adaptive_version))
static_proba_m6   = predict_proba(static_model,   X_eval1)
adaptive_proba_m6 = predict_proba(adaptive_model, X_eval1)

# Save predictions
h_s6 = save_preds(static_proba_m6,   y_eval1.values, f"static_v1_month6")
h_a6 = save_preds(adaptive_proba_m6, y_eval1.values, f"adaptive_{adaptive_version}_month6")

m6_comparison = compare_periods(
    y_true=y_eval1.values,
    static_proba=static_proba_m6,
    adaptive_proba=adaptive_proba_m6,
    threshold=FROZEN_THRESHOLD,
    period_label="month6",
    adaptive_version=adaptive_version,
)
save_json(m6_comparison, METRICS / "comparison_month6.json")
log(f"  Month 6 Static  PR-AUC: {m6_comparison['static']['PR_AUC']:.6f}")
log(f"  Month 6 Adaptive PR-AUC: {m6_comparison['adaptive']['PR_AUC']:.6f}")
log(f"  Month 6 Delta: {m6_comparison['delta_PR_AUC']:+.6f}")


# ===========================================================================
#  STEP 9 — MONTH 6: PSI MONITORING WINDOW 2 (AFTER evaluation)
# ===========================================================================
log("STEP 9 — Month 6: PSI drift monitoring (Window 2, AFTER evaluation)")
psi_result_m6 = psi_detector.compute(X_eval1, window_label="month6")
save_json(psi_result_m6, DRIFT / "psi_month6.json")
log(f"  Month 6 PSI: {psi_result_m6['triggered_count']}/{psi_result_m6['feature_denominator']} "
    f"features >= {psi_result_m6['psi_threshold']} | Triggered: {psi_result_m6['triggered']}")
if psi_result_m6["triggered"]:
    top5 = sorted(psi_result_m6["psi_values"].items(), key=lambda x: -x[1])[:5]
    log(f"  Top shifted features: {[(f, round(v,4)) for f,v in top5]}")


# ===========================================================================
#  STEP 10 — CONDITIONAL v3 RETRAINING
# ===========================================================================
v3_result = None
if psi_result_m6["triggered"]:
    log("STEP 10 — Month 6 TRIGGERED: Training v3 on months 0-6")
    train_0_6 = df[df[TIME_COL].isin([0, 1, 2, 3, 4, 5, 6])].reset_index(drop=True)
    X_train_v3, y_train_v3 = split_Xy(train_0_6)
    v3_result = train_model(
        X_train_v3, y_train_v3,
        version_label="v3",
        training_months=[0, 1, 2, 3, 4, 5, 6],
        output_dir=MODELS,
    )
    adaptive_model   = v3_result
    adaptive_version = "v3"
    log(f"  v3 trained. Hash: {v3_result['model_hash'][:16]}")
    log("  Threshold remains FROZEN (not re-selected for v3).")
else:
    log(f"STEP 10 — Month 6 NOT triggered. Adaptive model remains {adaptive_version}.")


# ===========================================================================
#  STEP 11 — MONTH 7: PROTECTED FINAL EVALUATION (exactly once)
# ===========================================================================
log("STEP 11 — Month 7: PROTECTED FINAL EVALUATION (evaluated EXACTLY ONCE)")
log(f"  Static: v1 | Adaptive: {adaptive_version} | Threshold: {FROZEN_THRESHOLD:.6f}")
# Only now do we extract Month 7
X_eval2, y_eval2 = split_Xy(eval2_df)

static_proba_m7   = predict_proba(static_model,   X_eval2)
adaptive_proba_m7 = predict_proba(adaptive_model, X_eval2)

# Save predictions (immutable — must not re-run)
h_s7 = save_preds(static_proba_m7,   y_eval2.values, f"static_v1_month7")
h_a7 = save_preds(adaptive_proba_m7, y_eval2.values, f"adaptive_{adaptive_version}_month7")

m7_comparison = compare_periods(
    y_true=y_eval2.values,
    static_proba=static_proba_m7,
    adaptive_proba=adaptive_proba_m7,
    threshold=FROZEN_THRESHOLD,
    period_label="month7",
    adaptive_version=adaptive_version,
)
save_json(m7_comparison, METRICS / "comparison_month7.json")
log(f"  Month 7 Static   PR-AUC: {m7_comparison['static']['PR_AUC']:.6f}")
log(f"  Month 7 Adaptive PR-AUC: {m7_comparison['adaptive']['PR_AUC']:.6f}")
log(f"  Month 7 Delta:           {m7_comparison['delta_PR_AUC']:+.6f}")
log("  Month 7 evaluation COMPLETE. No further tuning or retraining permitted.")


# ===========================================================================
#  STEP 12 — PAIRED BOOTSTRAP (both periods)
# ===========================================================================
log("STEP 12 — Paired bootstrap (2000 resamples, seed=42)")
bs_m6 = paired_bootstrap_pr_auc(
    y_eval1.values, static_proba_m6, adaptive_proba_m6,
    period_label="month6",
)
bs_m7 = paired_bootstrap_pr_auc(
    y_eval2.values, static_proba_m7, adaptive_proba_m7,
    period_label="month7",
)
save_json(bs_m6, METRICS / "bootstrap_month6.json")
save_json(bs_m7, METRICS / "bootstrap_month7.json")
log(format_bootstrap_result(bs_m6))
log(format_bootstrap_result(bs_m7))


# ===========================================================================
#  STEP 13 — SAVE MASTER RESULTS MANIFEST
# ===========================================================================
log("STEP 13 — Saving master results manifest")
results = {
    "run_id":             RUN_ID,
    "protocol_version":   "1.1",
    "dataset_hash":       "7bf10a37ce07e72e",
    "frozen_threshold":   FROZEN_THRESHOLD,
    "threshold_source":   "v1 predictions on Month 4",
    "adaptive_path":      f"v1 -> {'v2 -> ' if v2_result else ''}{'v3' if v3_result else adaptive_version}",
    "month5_psi": {
        "triggered":         psi_result_m5["triggered"],
        "triggered_count":   psi_result_m5["triggered_count"],
        "trigger_fraction":  psi_result_m5["trigger_fraction"],
    },
    "month6_psi": {
        "triggered":         psi_result_m6["triggered"],
        "triggered_count":   psi_result_m6["triggered_count"],
        "trigger_fraction":  psi_result_m6["trigger_fraction"],
    },
    "month6_results": m6_comparison,
    "month7_results": m7_comparison,
    "bootstrap_month6": {k: v for k, v in bs_m6.items() if k != "bootstrap_deltas"},
    "bootstrap_month7": {k: v for k, v in bs_m7.items() if k != "bootstrap_deltas"},
}
save_json(results, ARTS / "manifests" / f"results_{RUN_ID}.json")

# ===========================================================================
#  STEP 14 — FINAL LOG
# ===========================================================================
log("=" * 60)
log("PHASE 4 STATUS: COMPLETE")
log(f"  Dataset:              BAF Base (hash: 7bf10a37ce07e72e)")
log(f"  Protocol:             v1.1")
log(f"  Frozen threshold:     {FROZEN_THRESHOLD:.6f}")
log(f"  Month 5 retrain:      {'YES -> v2' if psi_result_m5['triggered'] else 'NO (v1 retained)'}")
log(f"  Month 6 retrain:      {'YES -> v3' if psi_result_m6['triggered'] else 'NO'}")
log(f"  Adaptive final ver:   {adaptive_version}")
log(f"  Month 6 delta PR-AUC: {m6_comparison['delta_PR_AUC']:+.6f}")
log(f"  Month 7 delta PR-AUC: {m7_comparison['delta_PR_AUC']:+.6f}")
log(f"  Month 6 95% CI:       [{bs_m6['ci_lower']:+.6f}, {bs_m6['ci_upper']:+.6f}]")
log(f"  Month 7 95% CI:       [{bs_m7['ci_lower']:+.6f}, {bs_m7['ci_upper']:+.6f}]")
log("=" * 60)
log("STOPPING. Phase 5 / streaming / deployment NOT started.")

LOG_PATH.write_text("\n".join(log_lines), encoding="utf-8")
print(f"\nLog saved -> {LOG_PATH}")
