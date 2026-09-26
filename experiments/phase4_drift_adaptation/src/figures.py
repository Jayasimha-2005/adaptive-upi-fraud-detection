"""
figures.py — Phase 4 Research Visualisations
=============================================
Generates all 13 required plots from Protocol v1.1 Section 28.

SCOPE RULES (protocol-mandated):
  - Distribution / drift plots: Months 0-7
  - Model performance plots:    Months 6-7 ONLY (no in-sample predictions)
  - Adaptation timeline:        Months 5-6

DO NOT generate model performance curves for months 0-5.
"""
from __future__ import annotations
import json, pathlib, sys, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT   = pathlib.Path(__file__).resolve().parents[3]
PHASE4 = ROOT / "experiments/phase4_drift_adaptation"
ARTS   = PHASE4 / "artifacts"
FIGS   = PHASE4 / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# Load saved artifacts
def load_json(p): return json.loads(pathlib.Path(p).read_text())

psi_m5    = load_json(ARTS / "drift/psi_month5.json")
psi_m6    = load_json(ARTS / "drift/psi_month6.json")
m6_comp   = load_json(ARTS / "metrics/comparison_month6.json")
m7_comp   = load_json(ARTS / "metrics/comparison_month7.json")
bs_m6     = load_json(ARTS / "metrics/bootstrap_month6.json")
bs_m7     = load_json(ARTS / "metrics/bootstrap_month7.json")
thresh_a  = load_json(ARTS / "thresholds/frozen_threshold_v1.json")

# Load predictions
def load_preds(name):
    arr = np.load(str(ARTS / f"predictions/{name}.npy"))
    return arr[:, 0].astype(int), arr[:, 1]

y_m6, s_proba_m6 = load_preds("static_v1_month6")
_,    a_proba_m6 = load_preds("adaptive_v2_month6")
y_m7, s_proba_m7 = load_preds("static_v1_month7")
_,    a_proba_m7 = load_preds("adaptive_v3_month7")

# Load BAF for fraud prevalence
BAF_PATH = ROOT / "Datasets/BAF/Base.csv"

# Colour palette
COL_STATIC   = "#E05C5C"
COL_ADAPTIVE = "#4A90D9"
COL_PSI      = "#E8A838"
COL_TRIGGER  = "#D94A4A"
COL_GRAY     = "#888888"

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi":      120,
})

saved = []

def savefig(name):
    p = FIGS / name
    plt.savefig(str(p), bbox_inches="tight")
    plt.close()
    saved.append(name)
    print(f"  Saved: {name}")


# ---------------------------------------------------------------------------
# FIG 1 — Fraud prevalence by month 0-7
# ---------------------------------------------------------------------------
print("Fig 1: Fraud prevalence by month 0-7")
df_baf = pd.read_csv(BAF_PATH, usecols=["month", "fraud_bool"], low_memory=False)
monthly = df_baf.groupby("month")["fraud_bool"].agg(["sum", "count"])
monthly["rate"] = monthly["sum"] / monthly["count"] * 100

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(monthly.index, monthly["rate"], color=COL_ADAPTIVE, alpha=0.8, width=0.6)
ax.axvspan(4.5, 7.5, alpha=0.08, color="orange", label="Evaluation region")
ax.axvline(4.5, color=COL_GRAY, ls="--", lw=0.8)
ax.set_xlabel("Month"); ax.set_ylabel("Fraud rate (%)")
ax.set_title("Fraud Prevalence by Month (BAF Base, N=1,000,000)")
ax.set_xticks(range(8))
for bar, rate in zip(bars, monthly["rate"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{rate:.3f}%", ha="center", va="bottom", fontsize=9)
ax.legend(frameon=False)
savefig("fig01_fraud_prevalence_by_month.png")


# ---------------------------------------------------------------------------
# FIG 2 — PSI by feature (Month 5)
# ---------------------------------------------------------------------------
print("Fig 2: PSI by feature (Month 5)")
psi_vals_m5 = psi_m5["psi_values"]
features = sorted(psi_vals_m5.keys(), key=lambda k: psi_vals_m5[k], reverse=True)
vals = [psi_vals_m5[f] for f in features]
colours = [COL_TRIGGER if v >= 0.10 else COL_GRAY for v in vals]

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.barh(range(len(features)), vals, color=colours)
ax.axvline(0.10, color="black", ls="--", lw=1.2, label="PSI threshold = 0.10")
ax.set_yticks(range(len(features))); ax.set_yticklabels(features, fontsize=9)
ax.set_xlabel("PSI value"); ax.set_title(f"Month 5 PSI by Feature ({psi_m5['triggered_count']}/29 triggered)")
trigger_patch = mpatches.Patch(color=COL_TRIGGER, label="Above threshold")
normal_patch  = mpatches.Patch(color=COL_GRAY,    label="Below threshold")
ax.legend(handles=[trigger_patch, normal_patch], frameon=False)
plt.tight_layout()
savefig("fig02_psi_by_feature_month5.png")


# ---------------------------------------------------------------------------
# FIG 3 — PSI by feature (Month 6)
# ---------------------------------------------------------------------------
print("Fig 3: PSI by feature (Month 6)")
psi_vals_m6 = psi_m6["psi_values"]
features6 = sorted(psi_vals_m6.keys(), key=lambda k: psi_vals_m6[k], reverse=True)
vals6 = [psi_vals_m6[f] for f in features6]
colours6 = [COL_TRIGGER if v >= 0.10 else COL_GRAY for v in vals6]

fig, ax = plt.subplots(figsize=(12, 5))
ax.barh(range(len(features6)), vals6, color=colours6)
ax.axvline(0.10, color="black", ls="--", lw=1.2, label="PSI threshold = 0.10")
ax.set_yticks(range(len(features6))); ax.set_yticklabels(features6, fontsize=9)
ax.set_xlabel("PSI value"); ax.set_title(f"Month 6 PSI by Feature ({psi_m6['triggered_count']}/29 triggered)")
ax.legend(handles=[trigger_patch, normal_patch], frameon=False)
plt.tight_layout()
savefig("fig03_psi_by_feature_month6.png")


# ---------------------------------------------------------------------------
# FIG 4 — Triggered feature count by monitoring window
# ---------------------------------------------------------------------------
print("Fig 4: PSI trigger count by monitoring window")
fig, ax = plt.subplots(figsize=(6, 4))
windows = ["Month 5\n(Window 1)", "Month 6\n(Window 2)"]
counts  = [psi_m5["triggered_count"], psi_m6["triggered_count"]]
colours_bars = [COL_TRIGGER, COL_TRIGGER]
ax.bar(windows, counts, color=colours_bars, alpha=0.85, width=0.4)
ax.axhline(6, color="black", ls="--", lw=1.2, label="Trigger threshold (6/29)")
for i, (w, c) in enumerate(zip(windows, counts)):
    ax.text(i, c + 0.3, f"{c}/29", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Features with PSI >= 0.10")
ax.set_title("PSI-Triggered Features per Monitoring Window")
ax.legend(frameon=False)
savefig("fig04_psi_trigger_count.png")


# ---------------------------------------------------------------------------
# FIG 5 — Adaptation timeline (v1 → v2 → v3)
# ---------------------------------------------------------------------------
print("Fig 5: Adaptation timeline")
fig, ax = plt.subplots(figsize=(12, 3))
ax.set_xlim(-0.5, 7.5); ax.set_ylim(-0.8, 1.4)
ax.axis("off")
# Month boxes
month_colors = {
    0: "#C8E6C9", 1: "#C8E6C9", 2: "#C8E6C9", 3: "#C8E6C9",  # train
    4: "#FFF9C4",  # val
    5: "#FFCC80",  # monitor 1
    6: "#FFAB91",  # eval 1 + monitor 2
    7: "#EF9A9A",  # protected
}
labels = {0:"Train",1:"Train",2:"Train",3:"Train",4:"Val (thresh)",5:"Monitor 1",6:"Eval 1 +\nMonitor 2",7:"Protected\nFinal Eval"}
for m in range(8):
    ax.add_patch(mpatches.FancyBboxPatch((m-0.4, 0.2), 0.8, 0.6,
        boxstyle="round,pad=0.05", fc=month_colors[m], ec="gray", lw=0.8))
    ax.text(m, 0.5, f"M{m}", ha="center", va="center", fontweight="bold", fontsize=10)
    ax.text(m, 0.15, labels[m], ha="center", va="top", fontsize=7, color="#333")

# Model version annotations
ax.annotate("v1\n(0-3)", xy=(1.5, 0.8), fontsize=10, ha="center",
    bbox=dict(boxstyle="round", fc=COL_ADAPTIVE, alpha=0.3))
ax.annotate("PSI: 17/29 ✓\n→ retrain v2", xy=(5, 0.85), fontsize=9, ha="center", color=COL_TRIGGER)
ax.annotate("v2\n(0-5)", xy=(5.8, 1.15), fontsize=10, ha="center",
    bbox=dict(boxstyle="round", fc=COL_ADAPTIVE, alpha=0.4))
ax.annotate("PSI: 18/29 ✓\n→ retrain v3", xy=(6, -0.15), fontsize=9, ha="center", color=COL_TRIGGER)
ax.annotate("v3\n(0-6)", xy=(6.8, -0.55), fontsize=10, ha="center",
    bbox=dict(boxstyle="round", fc=COL_ADAPTIVE, alpha=0.5))

ax.set_title("Phase 4 — Adaptation Timeline (Path D: v1 → v2 → v3)", fontsize=12)
savefig("fig05_adaptation_timeline.png")


# ---------------------------------------------------------------------------
# FIG 6 — Model performance comparison (Month 6 + 7)
# ---------------------------------------------------------------------------
print("Fig 6: PR-AUC comparison by period")
fig, ax = plt.subplots(figsize=(7, 4))
periods = ["Month 6", "Month 7"]
static_auc   = [m6_comp["static"]["PR_AUC"],   m7_comp["static"]["PR_AUC"]]
adaptive_auc = [m6_comp["adaptive"]["PR_AUC"],  m7_comp["adaptive"]["PR_AUC"]]
x = np.arange(2)
w = 0.32
b1 = ax.bar(x - w/2, static_auc,   w, label="Static v1",      color=COL_STATIC,   alpha=0.85)
b2 = ax.bar(x + w/2, adaptive_auc, w, label="Adaptive (v2/v3)",color=COL_ADAPTIVE, alpha=0.85)
for bar, val in zip(list(b1)+list(b2), static_auc+adaptive_auc):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(periods)
ax.set_ylabel("PR-AUC")
ax.set_title("PR-AUC: Static vs Adaptive (Evaluation Periods Only)")
ax.legend(frameon=False)
savefig("fig06_prauc_comparison.png")


# ---------------------------------------------------------------------------
# FIG 7 + 8 — PR Curves Month 6 and Month 7
# ---------------------------------------------------------------------------
for label, y_true, s_prob, a_prob, fname in [
    ("Month 6", y_m6, s_proba_m6, a_proba_m6, "fig07_pr_curve_month6.png"),
    ("Month 7", y_m7, s_proba_m7, a_proba_m7, "fig08_pr_curve_month7.png"),
]:
    print(f"Fig PR curve: {label}")
    fig, ax = plt.subplots(figsize=(6, 5))
    for proba, color, lbl in [
        (s_prob, COL_STATIC,   f"Static v1 (PR-AUC={average_precision_score(y_true, s_prob):.4f})"),
        (a_prob, COL_ADAPTIVE, f"Adaptive (PR-AUC={average_precision_score(y_true, a_prob):.4f})"),
    ]:
        p, r, _ = precision_recall_curve(y_true, proba)
        ax.plot(r, p, color=color, lw=2, label=lbl)
    # Baseline
    ax.axhline(y_true.mean(), color="gray", ls="--", lw=1, label=f"Baseline (fraud rate={y_true.mean():.4f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {label}")
    ax.legend(frameon=False, fontsize=9)
    savefig(fname)


# ---------------------------------------------------------------------------
# FIG 9 + 10 — ROC Curves Month 6 and Month 7
# ---------------------------------------------------------------------------
from sklearn.metrics import roc_auc_score
for label, y_true, s_prob, a_prob, fname in [
    ("Month 6", y_m6, s_proba_m6, a_proba_m6, "fig09_roc_curve_month6.png"),
    ("Month 7", y_m7, s_proba_m7, a_proba_m7, "fig10_roc_curve_month7.png"),
]:
    print(f"Fig ROC curve: {label}")
    fig, ax = plt.subplots(figsize=(5, 5))
    for proba, color, lbl in [
        (s_prob, COL_STATIC,   f"Static v1 (AUC={roc_auc_score(y_true, s_prob):.4f})"),
        (a_prob, COL_ADAPTIVE, f"Adaptive (AUC={roc_auc_score(y_true, a_prob):.4f})"),
    ]:
        fpr, tpr, _ = roc_curve(y_true, proba)
        ax.plot(fpr, tpr, color=color, lw=2, label=lbl)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"ROC Curve — {label}")
    ax.legend(frameon=False, fontsize=9)
    savefig(fname)


# ---------------------------------------------------------------------------
# FIG 11 — Top-K Precision comparison
# ---------------------------------------------------------------------------
print("Fig 11: Precision @ Top-K")
fig, ax = plt.subplots(figsize=(7, 4))
ks = [100, 500, 1000]
s_pk_m7 = [m7_comp["static"][f"P_at_{k}"] for k in ks]
a_pk_m7 = [m7_comp["adaptive"][f"P_at_{k}"] for k in ks]
x = np.arange(len(ks))
ax.bar(x - 0.18, s_pk_m7, 0.32, label="Static v1",      color=COL_STATIC,   alpha=0.85)
ax.bar(x + 0.18, a_pk_m7, 0.32, label="Adaptive v3",    color=COL_ADAPTIVE, alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels([f"Top-{k}" for k in ks])
ax.set_ylabel("Precision @ K"); ax.set_title("Precision @ Top-K — Month 7")
ax.legend(frameon=False)
savefig("fig11_precision_at_topk_month7.png")


# ---------------------------------------------------------------------------
# FIG 12 — Recall @ FPR comparison (Month 7)
# ---------------------------------------------------------------------------
print("Fig 12: Recall @ FPR")
fig, ax = plt.subplots(figsize=(7, 4))
fpr_levels = ["0.1%", "0.5%", "1%", "2%"]
fpr_keys   = ["Recall_at_FPR_0001","Recall_at_FPR_0005","Recall_at_FPR_001","Recall_at_FPR_002"]
s_rec = [m7_comp["static"][k]   for k in fpr_keys]
a_rec = [m7_comp["adaptive"][k] for k in fpr_keys]
x = np.arange(len(fpr_levels))
ax.bar(x - 0.18, s_rec, 0.32, label="Static v1",   color=COL_STATIC,   alpha=0.85)
ax.bar(x + 0.18, a_rec, 0.32, label="Adaptive v3", color=COL_ADAPTIVE, alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels([f"FPR ≤ {f}" for f in fpr_levels])
ax.set_ylabel("Recall"); ax.set_title("Recall @ FPR Constraint — Month 7")
ax.legend(frameon=False)
savefig("fig12_recall_at_fpr_month7.png")


# ---------------------------------------------------------------------------
# FIG 13 — Bootstrap delta distribution (Month 7)
# ---------------------------------------------------------------------------
print("Fig 13: Bootstrap delta distribution (Month 7)")
deltas = np.array(bs_m7["bootstrap_deltas"])
obs    = bs_m7["observed_delta"]
ci_lo  = bs_m7["ci_lower"]
ci_hi  = bs_m7["ci_upper"]

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(deltas, bins=60, color=COL_ADAPTIVE, alpha=0.7, edgecolor="white")
ax.axvline(obs,   color="black",      lw=2,   label=f"Observed delta = {obs:+.4f}")
ax.axvline(ci_lo, color=COL_TRIGGER,  lw=1.5, ls="--", label=f"95% CI lower = {ci_lo:+.4f}")
ax.axvline(ci_hi, color=COL_TRIGGER,  lw=1.5, ls="--", label=f"95% CI upper = {ci_hi:+.4f}")
ax.axvline(0,     color="gray",       lw=1,   ls=":", label="Zero (no difference)")
ax.set_xlabel("Delta PR-AUC (Adaptive - Static)")
ax.set_ylabel("Bootstrap resample count")
ax.set_title("Paired Bootstrap Distribution — Month 7 (2000 resamples)")
ax.legend(frameon=False, fontsize=9)
savefig("fig13_bootstrap_delta_month7.png")


print(f"\nAll {len(saved)} figures saved to: {FIGS}")
