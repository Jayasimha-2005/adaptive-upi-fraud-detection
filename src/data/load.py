"""
src/data/load.py
Loads and joins IEEE-CIS train_transaction + train_identity.

DESIGN PRINCIPLES
-----------------
- Raw files are NEVER modified.
- Transaction table is always the primary (left) side of the join.
- A 1:1 join guarantee check is performed post-join.
- Returns the combined dataframe plus metadata.
"""
from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def load_and_join(
    tx_path: str | Path,
    id_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load train_transaction.csv and train_identity.csv, join them, and return
    the combined dataframe.

    Join strategy
    -------------
    LEFT JOIN on TransactionID.
    Transactions without identity records keep their rows (identity columns = NaN).
    Identity rows with no matching transaction are discarded (zero orphans confirmed
    in forensic analysis).

    The joined frame has exactly as many rows as train_transaction.csv.

    Parameters
    ----------
    tx_path : path to train_transaction.csv
    id_path : path to train_identity.csv

    Returns
    -------
    df_joined : pd.DataFrame
        Combined frame. Row count == len(train_transaction).
    meta : dict
        Join metadata for reporting.
    """
    tx_path = Path(tx_path)
    id_path = Path(id_path)

    # ── Load transaction (chunked) ─────────────────────────────────────────────
    logger.info("Loading train_transaction.csv ...")
    t0 = time.time()
    chunks = []
    for chunk in pd.read_csv(tx_path, chunksize=100_000, low_memory=False):
        chunks.append(chunk)
    df_tx = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    n_tx = len(df_tx)
    logger.info("  Loaded %d rows x %d cols in %.1fs", n_tx, len(df_tx.columns), time.time() - t0)

    # ── Load identity (small — full load) ─────────────────────────────────────
    logger.info("Loading train_identity.csv ...")
    df_id = pd.read_csv(id_path, low_memory=False)
    n_id = len(df_id)
    logger.info("  Loaded %d rows x %d cols", n_id, len(df_id.columns))

    # ── LEFT JOIN ─────────────────────────────────────────────────────────────
    logger.info("Performing LEFT JOIN on TransactionID ...")
    t1 = time.time()
    df_joined = df_tx.merge(df_id, on="TransactionID", how="left")
    logger.info("  Join completed in %.1fs", time.time() - t1)

    # ── Post-join integrity check ─────────────────────────────────────────────
    n_joined = len(df_joined)
    if n_joined != n_tx:
        raise ValueError(
            f"Join produced {n_joined} rows but expected {n_tx}. "
            "This indicates a many-to-many or one-to-many join. "
            "Inspect for duplicate TransactionIDs in identity file."
        )

    dup_after = int(df_joined["TransactionID"].duplicated().sum())
    if dup_after > 0:
        raise ValueError(f"Post-join duplicate TransactionIDs detected: {dup_after}")

    n_with_id = int(df_joined["DeviceType"].notna().sum())   # DeviceType as identity proxy
    pct_with_id = round(n_with_id / n_joined * 100, 2)

    meta = {
        "tx_rows_before_join":   n_tx,
        "id_rows_before_join":   n_id,
        "joined_rows":           n_joined,
        "rows_with_identity":    n_with_id,
        "pct_with_identity":     pct_with_id,
        "rows_without_identity": n_joined - n_with_id,
        "join_type":             "LEFT JOIN on TransactionID",
        "post_join_duplicates":  dup_after,
        "total_columns":         len(df_joined.columns),
    }
    logger.info(
        "  Joined: %d rows | %.1f%% have identity info | %d total cols",
        n_joined, pct_with_id, len(df_joined.columns),
    )
    return df_joined, meta
