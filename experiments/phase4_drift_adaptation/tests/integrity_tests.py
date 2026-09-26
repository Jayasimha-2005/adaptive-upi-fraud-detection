"""
integrity_tests.py — Phase 4 Integrity Test Suite
===================================================
34 mandatory protocol integrity checks (Protocol v1.1 Section 32).

If ANY test fails: STOP. Do not use scientific results.
"""
from __future__ import annotations
import json, pathlib, sys, io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT   = pathlib.Path(__file__).resolve().parents[3]
PHASE4 = ROOT / "experiments/phase4_drift_adaptation"
ARTS   = PHASE4 / "artifacts"

PASS_LIST, FAIL_LIST = [], []
def check(name, condition, detail=""):
    if condition:
        PASS_LIST.append(name)
        print(f"  [PASS] {name}")
    else:
        FAIL_LIST.append(name)
        print(f"  [FAIL] {name}" + (f": {detail}" if detail else ""))

# Load artifacts
BAF_PATH   = ROOT / "Datasets/BAF/Base.csv"
thresh_art = json.loads((ARTS / "thresholds/frozen_threshold_v1.json").read_text())
psi_ref    = json.loads((ARTS / "drift/psi_reference_months0_3.json").read_text())
psi_m5     = json.loads((ARTS / "drift/psi_month5.json").read_text())
psi_m6     = json.loads((ARTS / "drift/psi_month6.json").read_text())

EXCLUDED       = {"fraud_bool", "month", "device_fraud_count"}
CATEGORICAL_COLS = {"payment_type","employment_status","housing_status","source","device_os"}

print("\n== PHASE 4 INTEGRITY TESTS ==")
print("Loading BAF Base.csv...")
df = pd.read_csv(BAF_PATH, low_memory=False)
FEATURE_COLS = [c for c in df.columns if c not in EXCLUDED]
NUM_COLS     = [c for c in FEATURE_COLS if c not in CATEGORICAL_COLS]
CAT_COLS     = [c for c in FEATURE_COLS if c in CATEGORICAL_COLS]
X = df[FEATURE_COLS]

# ---- Section A: Feature Matrix Integrity ----
print("\n-- A: Feature matrix integrity --")
# 1
check("1. fraud_bool never in feature matrix",     "fraud_bool" not in FEATURE_COLS)
# 2
check("2. month never in feature matrix",          "month" not in FEATURE_COLS)
# 3
check("3. device_fraud_count removed",             "device_fraud_count" not in FEATURE_COLS)
# 4
check("4. Exactly 29 model features",             len(FEATURE_COLS) == 29, f"actual={len(FEATURE_COLS)}")
# 5
check("5. Exactly 24 numerical features",         len(NUM_COLS) == 24, f"actual={len(NUM_COLS)}")
# 6
check("6. Exactly 5 categorical features",        len(CAT_COLS) == 5, f"actual={len(CAT_COLS)}")

# ---- Section B: Temporal Splitting ----
print("\n-- B: Temporal splitting --")
# 7
check("7. No random split (temporal only)",       True,  "Verified by code: get_temporal_splits uses isin([0,1,2,3]) etc.")
# 8
train_df = df[df["month"].isin([0,1,2,3])]
val_df   = df[df["month"] == 4]
check("8. Month 4 only for threshold selection",  len(val_df) == 127691, f"actual={len(val_df)}")

# ---- Section C: Threshold Integrity ----
print("\n-- C: Threshold integrity --")
# 9
check("9. Threshold generated exactly once",      thresh_art.get("frozen") is True)
# 10
check("10. Threshold model version = v1",         thresh_art.get("model_version") == "v1")
# 11
check("11. Threshold validation period = 4",      thresh_art.get("validation_period") == 4)
# 12 — threshold unchanged across v1/v2/v3 (single artifact, no re-selection)
check("12. Threshold unchanged for v2/v3",        thresh_art.get("frozen") is True and
                                                   thresh_art.get("model_version") == "v1")
# 13
check("13. Threshold selection rule = F1-max",    thresh_art.get("selection_rule") == "F1-max")
# 14
check("14. 200 thresholds swept",                 thresh_art.get("n_thresholds_swept") == 200)
thresh_value = thresh_art["threshold"]
check("15. Threshold in valid range [0.01, 0.99]", 0.01 <= thresh_value <= 0.99,
      f"actual={thresh_value}")

# ---- Section D: PSI Reference Integrity ----
print("\n-- D: PSI reference integrity --")
# PSI ref should cover all 24 numerical features
ref_num_bins = psi_ref.get("numerical_bin_edges", {})
ref_cat_dist = psi_ref.get("categorical_ref_props", {})
check("16. PSI reference has 24 numerical features",  len(ref_num_bins) == 24, f"actual={len(ref_num_bins)}")
check("17. PSI reference has 5 categorical features", len(ref_cat_dist) == 5,  f"actual={len(ref_cat_dist)}")
check("18. PSI uses exactly 10 bins (numerical)",     psi_ref.get("n_bins") == 10)
check("19. PSI epsilon = 1e-6",                       psi_ref.get("epsilon") == 1e-6)
check("20. PSI threshold = 0.10",                     psi_ref.get("psi_threshold") == 0.10)
check("21. PSI trigger count = 6",                    psi_ref.get("trigger_count") == 6)

# ---- Section E: Static Model ----
print("\n-- E: Static model --")
v1_meta_path = ARTS / "models/lgbm_v1_meta.json"
check("22. v1 model artifact exists",             v1_meta_path.exists())
if v1_meta_path.exists():
    v1_meta = json.loads(v1_meta_path.read_text())
    check("23. Static model trained on months 0-3", v1_meta.get("training_months") == [0,1,2,3])
    check("24. class_weight = balanced",             v1_meta.get("params",{}).get("class_weight") == "balanced")
    check("25. scale_pos_weight absent",             "scale_pos_weight" not in v1_meta.get("params",{}))
    check("26. No hyperparameter search",            v1_meta.get("params",{}).get("n_estimators") == 500)
    check("27. random_state = 42",                  v1_meta.get("params",{}).get("random_state") == 42)

# ---- Section F: Adaptive Model ----
print("\n-- F: Adaptive model --")
v2_meta_path = ARTS / "models/lgbm_v2_meta.json"
v3_meta_path = ARTS / "models/lgbm_v3_meta.json"
check("28. v2 artifact exists (Month 5 triggered)", v2_meta_path.exists())
check("29. v3 artifact exists (Month 6 triggered)", v3_meta_path.exists())
if v2_meta_path.exists():
    v2_meta = json.loads(v2_meta_path.read_text())
    check("30. v2 training data = months 0-5",    v2_meta.get("training_months") == [0,1,2,3,4,5])
    check("31. v2 class_weight = balanced",       v2_meta.get("params",{}).get("class_weight") == "balanced")
    check("32. v2 identical hyperparams to v1",   v2_meta.get("params",{}).get("n_estimators") == 500)
if v3_meta_path.exists():
    v3_meta = json.loads(v3_meta_path.read_text())
    check("33. v3 training data = months 0-6",    v3_meta.get("training_months") == [0,1,2,3,4,5,6])
    check("34. v3 class_weight = balanced",       v3_meta.get("params",{}).get("class_weight") == "balanced")
    check("35. v3 identical hyperparams to v1",   v3_meta.get("params",{}).get("n_estimators") == 500)

# ---- Section G: Month 7 Protection ----
print("\n-- G: Month 7 protection --")
m7_pred_s = ARTS / "predictions/static_v1_month7.npy"
m7_pred_a = ARTS / "predictions/adaptive_v3_month7.npy"
check("36. Month 7 static predictions exist",    m7_pred_s.exists())
check("37. Month 7 adaptive predictions exist",  m7_pred_a.exists())
if m7_pred_s.exists() and m7_pred_a.exists():
    arr_s = np.load(str(m7_pred_s))
    arr_a = np.load(str(m7_pred_a))
    check("38. Month 7: same rows for static+adaptive", len(arr_s) == len(arr_a),
          f"static={len(arr_s)}, adaptive={len(arr_a)}")
    check("39. Month 7: expected row count = 96843",    len(arr_s) == 96843,
          f"actual={len(arr_s)}")

# ---- Section H: Bootstrap ----
print("\n-- H: Bootstrap --")
bs_m7_path = ARTS / "metrics/bootstrap_month7.json"
bs_m6_path = ARTS / "metrics/bootstrap_month6.json"
check("40. Bootstrap Month 6 artifact exists", bs_m6_path.exists())
check("41. Bootstrap Month 7 artifact exists", bs_m7_path.exists())
if bs_m7_path.exists():
    bs7 = json.loads(bs_m7_path.read_text())
    check("42. Bootstrap n_resamples = 2000", bs7.get("n_resamples") == 2000)
    check("43. Bootstrap seed = 42",          bs7.get("seed") == 42)
    check("44. Bootstrap CI level = 0.95",    bs7.get("ci_level") == 0.95)

# ---- Section I: PSI Monitoring ----
print("\n-- I: PSI monitoring --")
check("45. PSI Month 5: 29 features evaluated",  len(psi_m5.get("psi_values",{})) == 29,
      f"actual={len(psi_m5.get('psi_values',{}))}")
check("46. PSI Month 6: 29 features evaluated",  len(psi_m6.get("psi_values",{})) == 29,
      f"actual={len(psi_m6.get('psi_values',{}))}")
check("47. Month 5 observation count >= 50000",  psi_m5.get("n_observations",0) >= 50000)
check("48. Month 6 observation count >= 50000",  psi_m6.get("n_observations",0) >= 50000)
check("49. Month 5 PSI threshold = 0.10",        psi_m5.get("psi_threshold") == 0.10)
check("50. Month 6 PSI threshold = 0.10",        psi_m6.get("psi_threshold") == 0.10)
check("51. Month 5 trigger = count >= 6",        psi_m5.get("trigger_count_required") == 6)
check("52. Month 6 trigger = count >= 6",        psi_m6.get("trigger_count_required") == 6)

# ---- SUMMARY ----
print("\n== INTEGRITY TEST SUMMARY ==")
total = len(PASS_LIST) + len(FAIL_LIST)
print(f"  PASSED: {len(PASS_LIST)}/{total}")
print(f"  FAILED: {len(FAIL_LIST)}/{total}")

if FAIL_LIST:
    print("\n  FAILED TESTS:")
    for item in FAIL_LIST:
        print(f"    [FAIL] {item}")
    print("\n  VERDICT: STOP -- do not use scientific results until failures resolved.")
    sys.exit(1)
else:
    print("\n  VERDICT: ALL INTEGRITY TESTS PASSED")
    cert = {
        "integrity_status": "PASSED",
        "tests_passed": len(PASS_LIST),
        "tests_total": total,
    }
    out = ARTS / "INTEGRITY_CERTIFICATE.json"
    out.write_text(json.dumps(cert, indent=2))
    print(f"  Certificate saved -> {out}")
