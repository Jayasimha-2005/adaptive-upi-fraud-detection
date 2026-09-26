"""
Phase 4 — BAF Dataset Audit Script (READ-ONLY)
================================================
Inspects ALL BAF variants locally present.
Produces: baf_data_audit.json  +  console report
DO NOT modify any experiment artifacts.
"""
import hashlib, json, pathlib, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT     = pathlib.Path(__file__).resolve().parents[3]
BAF_DIR  = ROOT / "Datasets" / "BAF"
OUT_DIR  = ROOT / "experiments" / "phase4_drift_adaptation" / "data_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()[:16]

def audit_file(csv_path: pathlib.Path) -> dict:
    print(f"\n{'='*65}")
    print(f"Auditing: {csv_path.name}  ({csv_path.stat().st_size/1e6:.1f} MB)")
    print('='*65)

    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Shape: {df.shape}")

    target_col = None
    for cand in ["fraud_bool", "fraud", "label", "isFraud", "is_fraud", "target"]:
        if cand in df.columns:
            target_col = cand; break

    time_col = None
    for cand in ["month", "time", "timestamp", "period", "week", "day"]:
        if cand in df.columns:
            time_col = cand; break

    id_cols = [c for c in df.columns if c.lower() in
               ("id", "application_id", "customer_id", "uid", "user_id", "account_id")]

    numerical = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Target stats
    fraud_info = {}
    if target_col:
        vc = df[target_col].value_counts()
        fraud_info = {
            "target_col": target_col,
            "n_fraud": int(vc.get(1, 0)),
            "n_legit": int(vc.get(0, 0)),
            "fraud_rate": round(float(vc.get(1, 0)) / len(df), 6),
        }
        print(f"  Target: {target_col} | fraud={fraud_info['n_fraud']} ({fraud_info['fraud_rate']*100:.2f}%)")

    # Temporal stats
    temporal_info = {}
    if time_col:
        periods = sorted(df[time_col].unique())
        temporal_info["time_col"] = time_col
        temporal_info["n_periods"] = len(periods)
        temporal_info["periods"] = [int(p) for p in periods]
        print(f"  Time col: {time_col} | periods: {periods}")

        by_period = []
        for p in periods:
            mask = df[time_col] == p
            sub  = df[mask]
            n    = len(sub)
            nf   = int(sub[target_col].sum()) if target_col else None
            fr   = round(nf/n, 6) if nf is not None and n>0 else None
            miss = round(sub.isnull().mean().mean(), 6)
            by_period.append({
                "period": int(p), "rows": n,
                "fraud": nf, "fraud_rate": fr,
                "mean_missingness": miss,
            })
            print(f"    Period {p}: rows={n:>7,}  fraud={nf:>5}  rate={fr:.3%}  miss={miss:.4f}")
        temporal_info["by_period"] = by_period

    # Quality checks
    n_dup = int(df.duplicated().sum())
    n_const = [c for c in df.columns if df[c].nunique() <= 1]
    near_const = [c for c in numerical if df[c].nunique() <= 5 and c != target_col]
    missing_by_col = {c: round(float(df[c].isnull().mean()),4)
                      for c in df.columns if df[c].isnull().mean() > 0}

    print(f"  Duplicate rows: {n_dup}")
    print(f"  Constant cols:  {n_const}")
    print(f"  Missing cols:   {len(missing_by_col)}")

    # PSI adjacent months (numerical median)
    psi_info = {}
    if time_col and len(temporal_info.get("periods", [])) >= 2:
        periods = temporal_info["periods"]
        ref_df  = df[df[time_col] == periods[0]]
        psi_rows = []
        for p in periods[1:]:
            cur_df = df[df[time_col] == p]
            ks_vals = []
            for col in numerical[:10]:  # top 10 numerics for speed
                if col in (target_col, time_col): continue
                try:
                    stat, _ = ks_2samp(ref_df[col].dropna(), cur_df[col].dropna())
                    ks_vals.append(stat)
                except:
                    pass
            mean_ks = round(float(np.mean(ks_vals)), 4) if ks_vals else None
            psi_rows.append({"period": p, "mean_ks_vs_period0": mean_ks})
        psi_info["ks_vs_initial"] = psi_rows

    result = {
        "file": csv_path.name,
        "size_bytes": csv_path.stat().st_size,
        "sha256_16": sha16(csv_path),
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": df.columns.tolist(),
        "target": fraud_info,
        "temporal": temporal_info,
        "id_cols": id_cols,
        "numerical_cols": numerical,
        "categorical_cols": categorical,
        "duplicate_rows": n_dup,
        "constant_cols": n_const,
        "near_constant_cols": near_const,
        "missing_by_col": missing_by_col,
        "distribution_shift": psi_info,
    }
    return result

def main():
    all_results = {}
    for csv_path in sorted(BAF_DIR.glob("*.csv")):
        try:
            all_results[csv_path.name] = audit_file(csv_path)
        except Exception as e:
            all_results[csv_path.name] = {"error": str(e)}

    out_path = OUT_DIR / "baf_data_audit.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n\n{'='*65}")
    print(f"AUDIT SAVED -> {out_path}")
    print('='*65)

if __name__ == "__main__":
    main()
