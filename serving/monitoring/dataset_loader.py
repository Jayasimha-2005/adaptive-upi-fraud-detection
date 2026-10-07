"""
monitoring/dataset_loader.py
Dataset discovery and unseen test transaction streaming loader for the
FastAPI fraud serving dashboard and automated test clients.

Loads strictly from the official IEEE-CIS test_transaction.csv dataset.
Ensures zero target leakage: isFraud is never loaded, read, or forwarded.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

DEFAULT_DATASET_LOCATIONS = [
    Path("/app/data/test_transaction.csv"),
    Path("/app/serving/data/test_transaction.csv"),
    Path("./data/test_transaction.csv"),
    Path("serving/data/test_transaction.csv"),
    Path("d:/fraud-model-serving/Datasets/IEEE CIS-20260829T103704Z-1-001/IEEE CIS/test_transaction.csv"),
    Path("d:/adaptive-upi-fraud-detection/IEEE CIS-20260829T103704Z-1-001/IEEE CIS/test_transaction.csv"),
    Path("./IEEE CIS-20260829T103704Z-1-001/IEEE CIS/test_transaction.csv"),
    Path("../IEEE CIS-20260829T103704Z-1-001/IEEE CIS/test_transaction.csv"),
    Path("./test_transaction.csv"),
]


def find_dataset_file(explicit_path: Optional[str] = None) -> Path:
    """
    Resolve the file location of the IEEE-CIS test_transaction.csv dataset.
    Searches known container and local filesystem paths.
    """
    if explicit_path:
        p = Path(explicit_path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Specified dataset file not found: {explicit_path}")

    for candidate in DEFAULT_DATASET_LOCATIONS:
        if candidate.is_file():
            return candidate

    # Search current and parent directory for test_transaction.csv
    for root in [Path("."), Path(".."), Path("/app"), Path("d:/")]:
        try:
            matches = list(root.glob("**/test_transaction.csv"))
            if matches:
                return matches[0]
        except Exception:
            continue

    raise FileNotFoundError(
        "Could not automatically locate test_transaction.csv. "
        "Please place test_transaction.csv in serving/data/ or provide an explicit path."
    )


def row_to_payload(row: Dict[str, str]) -> Dict[str, Any]:
    """
    Convert a CSV row from test_transaction.csv into a clean dictionary payload
    compatible with the FastAPI /predict endpoint schema (TransactionRequest).
    Strictly excludes isFraud or ground-truth columns.
    """
    payload: Dict[str, Any] = {}
    for col, val in row.items():
        if col in ("isFraud", "is_fraud", "fraud_bool"):
            continue  # Strict target isolation
        if val == "" or val is None:
            continue
        try:
            if "." in val or "e" in val.lower():
                payload[col] = float(val)
            else:
                payload[col] = int(val)
        except ValueError:
            payload[col] = val

    return payload


def load_unseen_test_transactions(
    count: int,
    mode: str = "fixed",
    seed: Optional[int] = None,
    dataset_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load exactly `count` unseen test transactions from IEEE-CIS test_transaction.csv.

    Parameters
    ----------
    count : int
        Number of test transactions to select (e.g. 5, 10, 50, 100, 1000, 500).
    mode : str
        'fixed' to read top N sequential unseen transactions, or
        'random' to sample N random unseen transactions across the test set.
    seed : int, optional
        Deterministic random seed if mode == 'random'.
    dataset_path : str, optional
        Explicit path to test_transaction.csv.

    Returns
    -------
    List[Dict[str, Any]]
        List of cleaned transaction dictionaries ready for scoring.
    """
    csv_file = find_dataset_file(dataset_path)
    count = max(1, count)
    results: List[Dict[str, Any]] = []

    if mode == "fixed":
        with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row_to_payload(row))
                if len(results) >= count:
                    break
    elif mode == "random":
        rng = random.Random(seed) if seed is not None else random.Random()
        # Sample random indices from the first 150k rows or full range
        sample_pool = 150000 if count <= 1000 else 506691
        target_indices: Set[int] = set(rng.sample(range(sample_pool), count))
        max_idx = max(target_indices)

        with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i in target_indices:
                    results.append(row_to_payload(row))
                if i >= max_idx or len(results) >= count:
                    break
    else:
        raise ValueError(f"Unsupported mode: {mode}. Must be 'fixed' or 'random'.")

    return results
