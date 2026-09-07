"""
src/data/validate.py
Data validation pipeline for IEEE-CIS train_transaction + train_identity.

DESIGN PRINCIPLES
-----------------
- Never modifies raw files.
- All statistics are computed from the actual data (never invented).
- Reports any schema violations, duplicates, target leakage risks.
- Produces both a human-readable Markdown report and a machine-readable JSON.
"""
from __future__ import annotations

import json
import time
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

def validate_ieee_cis(
    tx_path: str | Path,
    id_path: str | Path,
    reports_dir: str | Path,
) -> dict[str, Any]:
    """
    Validate IEEE-CIS train_transaction.csv and train_identity.csv.

    Parameters
    ----------
    tx_path : path to train_transaction.csv
    id_path : path to train_identity.csv
    reports_dir : directory where validation reports are written

    Returns
    -------
    dict
        Validation report (also written to reports_dir/data_validation.json).

    Raises
    ------
    ValueError
        If critical validation checks fail (e.g. duplicate TransactionIDs).
    """
    tx_path  = Path(tx_path)
    id_path  = Path(id_path)
    rep_dir  = Path(reports_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {}
    errors:  list[str] = []
    warnings: list[str] = []

    # ── 1. File existence ──────────────────────────────────────────────────────
    logger.info("Checking file existence...")
    for p in [tx_path, id_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")
        size_mb = p.stat().st_size / 1e6
        logger.info("  %s — %.1f MB", p.name, size_mb)
    report["files"] = {
        "train_transaction": {"path": str(tx_path), "size_mb": round(tx_path.stat().st_size / 1e6, 2)},
        "train_identity":    {"path": str(id_path), "size_mb": round(id_path.stat().st_size / 1e6, 2)},
    }

    # ── 2. Load transaction (efficient dtypes) ─────────────────────────────────
    logger.info("Loading train_transaction.csv (chunked to reduce peak memory)...")
    t0 = time.time()
    tx_chunks = []
    for chunk in pd.read_csv(tx_path, chunksize=100_000, low_memory=False):
        tx_chunks.append(chunk)
    df_tx = pd.concat(tx_chunks, ignore_index=True)
    del tx_chunks
    logger.info("  Loaded %d rows x %d cols in %.1fs", len(df_tx), len(df_tx.columns), time.time() - t0)

    # ── 3. Transaction basic stats ─────────────────────────────────────────────
    n_tx      = len(df_tx)
    n_tx_cols = len(df_tx.columns)
    tx_dtypes = df_tx.dtypes.astype(str).to_dict()

    # Duplicate TransactionID
    n_tx_id_dup = int(df_tx["TransactionID"].duplicated().sum())
    if n_tx_id_dup > 0:
        errors.append(f"CRITICAL: {n_tx_id_dup} duplicate TransactionIDs in train_transaction")
    else:
        logger.info("  TransactionID: all unique (%d)", n_tx)

    tx_id_min = int(df_tx["TransactionID"].min())
    tx_id_max = int(df_tx["TransactionID"].max())

    # Target
    fraud_count = int(df_tx["isFraud"].sum())
    legit_count = n_tx - fraud_count
    fraud_pct   = round(fraud_count / n_tx * 100, 4)
    class_ratio = round(legit_count / fraud_count, 1) if fraud_count > 0 else float("inf")

    # TransactionDT
    dt_min = float(df_tx["TransactionDT"].min())
    dt_max = float(df_tx["TransactionDT"].max())
    dt_span_days = (dt_max - dt_min) / 86400.0

    # TransactionAmt
    amt_stats = df_tx["TransactionAmt"].describe().round(4).to_dict()

    # Missing values
    tx_missing = df_tx.isnull().sum()
    tx_missing_pct = (tx_missing / n_tx * 100).round(4)
    n_zero_missing    = int((tx_missing == 0).sum())
    n_low_missing     = int(((tx_missing_pct > 0) & (tx_missing_pct <= 20)).sum())
    n_mid_missing     = int(((tx_missing_pct > 20) & (tx_missing_pct <= 80)).sum())
    n_high_missing    = int((tx_missing_pct > 80).sum())

    report["train_transaction"] = {
        "rows": n_tx,
        "cols": n_tx_cols,
        "tx_id_min": tx_id_min,
        "tx_id_max": tx_id_max,
        "duplicate_tx_ids": n_tx_id_dup,
        "fraud_count": fraud_count,
        "legit_count": legit_count,
        "fraud_pct": fraud_pct,
        "class_ratio": class_ratio,
        "dt_min_sec": dt_min,
        "dt_max_sec": dt_max,
        "dt_span_days": round(dt_span_days, 2),
        "dt_unit": "seconds_relative",
        "amount_stats": amt_stats,
        "missing_zero_cols": n_zero_missing,
        "missing_low_cols_0_20": n_low_missing,
        "missing_mid_cols_20_80": n_mid_missing,
        "missing_high_cols_over_80": n_high_missing,
    }
    logger.info("  Rows=%d | Fraud=%d (%.2f%%) | DT span=%.1f days",
                n_tx, fraud_count, fraud_pct, dt_span_days)

    # ── 4. Load identity ───────────────────────────────────────────────────────
    logger.info("Loading train_identity.csv...")
    df_id = pd.read_csv(id_path, low_memory=False)
    n_id      = len(df_id)
    n_id_cols = len(df_id.columns)
    n_id_dup  = int(df_id["TransactionID"].duplicated().sum())
    if n_id_dup > 0:
        errors.append(f"CRITICAL: {n_id_dup} duplicate TransactionIDs in train_identity")

    id_missing     = df_id.isnull().sum()
    id_missing_pct = (id_missing / n_id * 100).round(4)

    report["train_identity"] = {
        "rows": n_id,
        "cols": n_id_cols,
        "duplicate_tx_ids": n_id_dup,
        "missing_any_col_pct": round(float(id_missing_pct.max()), 2),
    }
    logger.info("  Rows=%d | Cols=%d | Dup IDs=%d", n_id, n_id_cols, n_id_dup)

    # ── 5. TransactionID linkage ───────────────────────────────────────────────
    logger.info("Verifying TransactionID linkage...")
    tx_ids    = set(df_tx["TransactionID"].tolist())
    id_ids    = set(df_id["TransactionID"].tolist())
    overlap   = len(tx_ids & id_ids)
    only_tx   = len(tx_ids - id_ids)
    only_id   = len(id_ids - tx_ids)
    pct_with_id = round(overlap / n_tx * 100, 2)

    if only_id > 0:
        warnings.append(f"{only_id} identity rows have no matching transaction — will be dropped in join")

    report["linkage"] = {
        "tx_unique_ids": len(tx_ids),
        "id_unique_ids": len(id_ids),
        "overlap": overlap,
        "pct_transactions_with_identity": pct_with_id,
        "tx_only": only_tx,
        "id_only": only_id,
        "join_type": "LEFT JOIN (transaction is primary)",
    }
    logger.info("  Overlap=%d (%.1f%% of transactions have identity)", overlap, pct_with_id)

    # ── 6. Target check ───────────────────────────────────────────────────────
    target_uniq = sorted(df_tx["isFraud"].dropna().unique().tolist())
    if set(target_uniq) != {0, 1}:
        errors.append(f"isFraud has unexpected values: {target_uniq}")
    if df_tx["isFraud"].isnull().sum() > 0:
        errors.append("isFraud contains null values")
    report["target"] = {
        "column": "isFraud",
        "unique_values": target_uniq,
        "null_count": int(df_tx["isFraud"].isnull().sum()),
        "fraud_count": fraud_count,
        "legit_count": legit_count,
        "fraud_pct": fraud_pct,
    }

    # ── 7. Compile errors/warnings ─────────────────────────────────────────────
    report["validation_status"] = "PASS" if not errors else "FAIL"
    report["errors"]   = errors
    report["warnings"] = warnings

    # ── 8. Write reports ──────────────────────────────────────────────────────
    json_path = rep_dir / "data_validation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote validation JSON: %s", json_path)

    _write_validation_md(report, rep_dir / "data_validation_report.md")

    if errors:
        raise ValueError(f"Data validation FAILED:\n" + "\n".join(errors))

    logger.info("Data validation PASSED.")
    return report


# ── Private ───────────────────────────────────────────────────────────────────

def _write_validation_md(report: dict, path: Path) -> None:
    """Write a human-readable Markdown validation report."""
    lines = [
        "# IEEE-CIS Data Validation Report",
        "",
        f"**Status:** {report['validation_status']}",
        "",
        "## File Summary",
        "",
        "| File | Path | Size (MB) |",
        "|------|------|-----------|",
    ]
    for k, v in report.get("files", {}).items():
        lines.append(f"| {k} | `{Path(v['path']).name}` | {v['size_mb']} |")

    tx = report.get("train_transaction", {})
    lines += [
        "",
        "## train_transaction.csv",
        "",
        f"- **Rows:** {tx.get('rows', 'N/A'):,}",
        f"- **Columns:** {tx.get('cols', 'N/A')}",
        f"- **Duplicate TransactionIDs:** {tx.get('duplicate_tx_ids', 'N/A')}",
        f"- **Fraud cases:** {tx.get('fraud_count', 'N/A'):,} ({tx.get('fraud_pct', 'N/A')}%)",
        f"- **Legitimate cases:** {tx.get('legit_count', 'N/A'):,}",
        f"- **Class ratio (legit:fraud):** {tx.get('class_ratio', 'N/A')}:1",
        f"- **TransactionDT range:** {tx.get('dt_min_sec', 'N/A')} → {tx.get('dt_max_sec', 'N/A')} sec",
        f"- **Temporal span:** {tx.get('dt_span_days', 'N/A')} days",
        f"- **Columns with 0% missing:** {tx.get('missing_zero_cols', 'N/A')}",
        f"- **Columns with 0–20% missing:** {tx.get('missing_low_cols_0_20', 'N/A')}",
        f"- **Columns with 20–80% missing:** {tx.get('missing_mid_cols_20_80', 'N/A')}",
        f"- **Columns with >80% missing:** {tx.get('missing_high_cols_over_80', 'N/A')}",
    ]

    idc = report.get("train_identity", {})
    lines += [
        "",
        "## train_identity.csv",
        "",
        f"- **Rows:** {idc.get('rows', 'N/A'):,}",
        f"- **Columns:** {idc.get('cols', 'N/A')}",
        f"- **Duplicate TransactionIDs:** {idc.get('duplicate_tx_ids', 'N/A')}",
    ]

    lnk = report.get("linkage", {})
    lines += [
        "",
        "## TransactionID Linkage",
        "",
        f"- **Transactions with identity record:** {lnk.get('overlap', 'N/A'):,} ({lnk.get('pct_transactions_with_identity', 'N/A')}%)",
        f"- **Transactions without identity:** {lnk.get('tx_only', 'N/A'):,}",
        f"- **Join type:** {lnk.get('join_type', 'N/A')}",
    ]

    if report.get("errors"):
        lines += ["", "## Errors", ""]
        for e in report["errors"]:
            lines.append(f"- **ERROR:** {e}")

    if report.get("warnings"):
        lines += ["", "## Warnings", ""]
        for w in report["warnings"]:
            lines.append(f"- WARNING: {w}")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote validation MD: %s", path)
