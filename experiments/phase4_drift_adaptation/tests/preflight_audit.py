"""
PHASE 4 PREFLIGHT AUDIT
========================
Mandatory safety check before any implementation code is written.
Verifies: frozen artifacts, dataset hash, row counts, feature structure,
governing document presence, and protocol consistency.
DO NOT modify any existing experiment artifacts.
"""
import hashlib, json, pathlib, sys, io
import pandas as pd
import numpy as np

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = pathlib.Path(__file__).resolve().parents[3]
PASS_LIST, FAIL_LIST = [], []

def check(name, condition, detail=""):
    if condition:
        PASS_LIST.append(name)
        print(f"  [PASS] {name}")
    else:
        FAIL_LIST.append(name)
        print(f"  [FAIL] {name}" + (f": {detail}" if detail else ""))

def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()[:16]

# -- 1. FROZEN EXPERIMENT SAFETY --
print("\n== 1. FROZEN EXPERIMENT SAFETY ==")
frozen_dirs = {
    "E1_lightgbm":          ROOT / "experiments/E1_lightgbm",
    "E3_hybrid":            ROOT / "experiments/E3_hybrid",
    "phase2_gru (E2a/E2b)": ROOT / "phase2_gru",
}
for label, path in frozen_dirs.items():
    check(f"{label} directory exists", path.exists(), str(path))
    check(f"{label} not empty",        path.exists() and any(path.rglob("*")))

e1_model = ROOT / "experiments/E1_lightgbm"
e1_files = list(e1_model.rglob("*")) if e1_model.exists() else []
check("E1 artifacts count >= 15", len([f for f in e1_files if f.is_file()]) >= 15,
      f"found {len([f for f in e1_files if f.is_file()])} files")

e3 = ROOT / "experiments/E3_hybrid"
e3_files = list(e3.rglob("*")) if e3.exists() else []
check("E3 artifacts count >= 40", len([f for f in e3_files if f.is_file()]) >= 40,
      f"found {len([f for f in e3_files if f.is_file()])} files")

p4 = ROOT / "experiments/phase4_drift_adaptation"
p4_files = [f for f in p4.rglob("*") if f.is_file()] if p4.exists() else []
p4_names = [str(f) for f in p4_files]
check("Phase4 dir exists",                p4.exists())
check("Phase4 isolated from E1",          not any("E1_lightgbm" in n for n in p4_names))
check("Phase4 isolated from E3",          not any("E3_hybrid" in n for n in p4_names))
check("Phase4 isolated from phase2_gru",  not any("phase2_gru" in n for n in p4_names))
check("requirements.txt present",         (ROOT / "requirements.txt").exists())
check(".gitignore present",               (ROOT / ".gitignore").exists())

# -- 2. GOVERNING DOCUMENTS --
print("\n== 2. GOVERNING DOCUMENTS ==")
docs = {
    "PHASE4_RESEARCH_PROTOCOL.md":   p4 / "docs/PHASE4_RESEARCH_PROTOCOL.md",
    "PHASE4_DESIGN_DECISIONS.md":    p4 / "docs/PHASE4_DESIGN_DECISIONS.md",
    "PHASE4_DATASET_AUDIT.md":       p4 / "docs/PHASE4_DATASET_AUDIT.md",
    "PHASE4_PROTOCOL_MANIFEST.json": p4 / "artifacts/PHASE4_PROTOCOL_MANIFEST.json",
    "baf_data_audit.json":           p4 / "data_audit/baf_data_audit.json",
}
for name, path in docs.items():
    check(f"{name} present", path.exists(), str(path))

manifest_path = p4 / "artifacts/PHASE4_PROTOCOL_MANIFEST.json"
if manifest_path.exists():
    with open(manifest_path) as f:
        manifest = json.load(f)
    check("Manifest version = 1.1",           manifest.get("protocol_version") == "1.1")
    check("Manifest go_no_go = APPROVED",     "APPROVED" in manifest.get("go_no_go", ""))
    check("Manifest status = FINAL",          "FINAL" in manifest.get("status", ""))
    check("Dataset = BAF Base",               manifest.get("dataset", {}).get("name") == "BAF Base")
    check("Usable features = 29",             manifest.get("dataset", {}).get("usable_features") == 29)
    check("Numerical features = 24",          manifest.get("dataset", {}).get("numerical_features") == 24)
    check("Categorical features = 5",         manifest.get("dataset", {}).get("categorical_features") == 5)
    check("Training months = [0,1,2,3]",      manifest.get("temporal_split", {}).get("initial_training") == [0,1,2,3])
    check("Validation month = [4]",           manifest.get("temporal_split", {}).get("validation_threshold_only") == [4])
    check("Protected period = [7]",           manifest.get("temporal_split", {}).get("evaluation_period_2_protected") == [7])
    check("PSI threshold = 0.10",             manifest.get("drift_detection", {}).get("per_feature_threshold") == 0.10)
    check("Aggregate trigger count = 6",      manifest.get("drift_detection", {}).get("aggregate_count_required") == 6)
    check("Feature denominator = 29",         manifest.get("drift_detection", {}).get("features_denominator") == 29)
    check("Threshold frozen permanently",     manifest.get("threshold_policy", {}).get("threshold_frozen_permanently") is True)
    check("Re-selection prohibited",          manifest.get("threshold_policy", {}).get("re_selection_prohibited") is True)
    check("class_weight = balanced",          manifest.get("static_model", {}).get("hyperparameters", {}).get("class_weight") == "balanced")
    check("Categorical = LightGBM native",   "LightGBM native" in str(manifest.get("static_model", {}).get("categorical_encoding", {}).get("method", "")))

if docs["PHASE4_RESEARCH_PROTOCOL.md"].exists():
    content = docs["PHASE4_RESEARCH_PROTOCOL.md"].read_text(encoding="utf-8")
    check("Protocol doc version = 1.1",       "1.1" in content)
    check("Protocol doc status = APPROVED",   "APPROVED FOR IMPLEMENTATION" in content)
    check("class_weight balanced in protocol", 'class_weight     = "balanced"' in content)
    check("LightGBM native in protocol",      "LightGBM native categorical" in content)
    check("Threshold frozen in protocol",     "THRESHOLD FROZEN" in content)
    check("scale_pos_weight excluded",        "scale_pos_weight is NOT used" in content or "NOT used" in content)

# -- 3. DATASET HASH + INTEGRITY --
print("\n== 3. DATASET HASH + INTEGRITY ==")
baf_path = ROOT / "Datasets/BAF/Base.csv"
check("Base.csv exists", baf_path.exists(), str(baf_path))
actual_hash = None

if baf_path.exists():
    print("  Computing SHA256...")
    actual_hash = sha16(baf_path)
    expected_hash = "7bf10a37ce07e72e"
    check(f"SHA256[:16] matches {expected_hash}", actual_hash == expected_hash,
          f"actual={actual_hash}")

    print("  Loading Base.csv for integrity verification...")
    df = pd.read_csv(baf_path, low_memory=False)

    check("Row count = 1,000,000",  len(df) == 1_000_000,     f"actual={len(df)}")
    check("Column count = 32",      len(df.columns) == 32,     f"actual={len(df.columns)}")
    check("fraud_bool present",     "fraud_bool" in df.columns)
    check("month present",          "month" in df.columns)
    check("device_fraud_count present", "device_fraud_count" in df.columns)
    check("n_periods = 8",          df["month"].nunique() == 8)
    check("Periods 0..7 only",      set(df["month"].unique()) == set(range(8)))
    check("No missing values",      df.isnull().sum().sum() == 0)
    check("No duplicate rows",      df.duplicated().sum() == 0)
    check("Target is binary {0,1}", set(df["fraud_bool"].unique()).issubset({0, 1}))

    expected_periods = {
        0: (132440, 1500), 1: (127620, 1198), 2: (136979, 1198),
        3: (150936, 1392), 4: (127691, 1452), 5: (119323, 1411),
        6: (108168, 1450), 7: (96843,  1428),
    }
    for m, (exp_rows, exp_fraud) in expected_periods.items():
        sub = df[df["month"] == m]
        check(f"Month {m}: rows = {exp_rows}",   len(sub) == exp_rows,
              f"actual={len(sub)}")
        check(f"Month {m}: fraud = {exp_fraud}", int(sub["fraud_bool"].sum()) == exp_fraud,
              f"actual={int(sub['fraud_bool'].sum())}")

    train = df[df["month"].isin([0,1,2,3])]
    check("Training rows = 547,975",  len(train) == 547975,               f"actual={len(train)}")
    check("Training fraud = 5,288",   int(train["fraud_bool"].sum()) == 5288, f"actual={int(train['fraud_bool'].sum())}")

    # -- 4. FEATURE STRUCTURE --
    print("\n== 4. FEATURE STRUCTURE ==")
    EXCLUDED         = {"fraud_bool", "month", "device_fraud_count"}
    CATEGORICAL_COLS = {"payment_type", "employment_status", "housing_status", "source", "device_os"}
    feature_cols     = [c for c in df.columns if c not in EXCLUDED]
    numerical_cols   = [c for c in feature_cols if c not in CATEGORICAL_COLS]
    cat_cols         = [c for c in feature_cols if c in CATEGORICAL_COLS]

    check("Usable features = 29",     len(feature_cols) == 29,   f"actual={len(feature_cols)}")
    check("Numerical features = 24",  len(numerical_cols) == 24, f"actual={len(numerical_cols)}")
    check("Categorical features = 5", len(cat_cols) == 5,        f"actual={len(cat_cols)}")
    check("fraud_bool excluded",      "fraud_bool" not in feature_cols)
    check("month excluded",           "month" not in feature_cols)
    check("device_fraud_count excluded", "device_fraud_count" not in feature_cols)
    for col in ["payment_type", "employment_status", "housing_status", "source", "device_os"]:
        check(f"Categorical '{col}' present", col in df.columns)
    check("device_fraud_count is constant (all zero)", df["device_fraud_count"].nunique() == 1)

# -- SUMMARY --
print("\n== PREFLIGHT AUDIT SUMMARY ==")
total = len(PASS_LIST) + len(FAIL_LIST)
print(f"  PASSED: {len(PASS_LIST)}/{total}")
print(f"  FAILED: {len(FAIL_LIST)}/{total}")

if FAIL_LIST:
    print("\n  FAILED CHECKS:")
    for item in FAIL_LIST:
        print(f"    [FAIL] {item}")
    print("\n  VERDICT: STOP -- resolve all failures before implementation.")
    sys.exit(1)
else:
    print("\n  VERDICT: ALL CHECKS PASSED -- CLEAR TO IMPLEMENT")
    cert = {
        "preflight_status": "PASSED",
        "checks_passed": len(PASS_LIST),
        "checks_total": total,
        "dataset_hash": actual_hash,
        "protocol_version": "1.1",
        "timestamp": str(pd.Timestamp.now()),
    }
    cert_path = p4 / "artifacts/PREFLIGHT_CERTIFICATE.json"
    with open(cert_path, "w") as fout:
        json.dump(cert, fout, indent=2)
    print(f"  Certificate saved -> {cert_path}")
