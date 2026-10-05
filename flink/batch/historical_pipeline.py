from pathlib import Path
import argparse
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flink.batch.data_loader import load_ieee_cis
from flink.batch.data_cleaner import clean_transactions
from flink.batch.dataset_joiner import join_transaction_identity
from flink.batch.feature_engineering import build_historical_features
from flink.batch.output_writer import write_features


TX_COLUMNS = [
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "addr1"
]

ID_COLUMNS = [
    "TransactionID",
    "DeviceType",
    "DeviceInfo"
]

CHUNK_SIZE = 50_000


def resolve_dataset_dir() -> Path:
    """Resolve IEEE-CIS dataset directory relative to repository root. Fails loudly if missing."""
    candidate_paths = [
        ROOT / "Datasets" / "IEEE CIS-20260829T103704Z-1-001" / "IEEE CIS",
        ROOT / "Datasets" / "raw",
        ROOT / "Datasets",
    ]
    for p in candidate_paths:
        if (p / "train_transaction.csv").is_file():
            return p
    raise FileNotFoundError(
        "CRITICAL ERROR: IEEE-CIS dataset (train_transaction.csv) not found in candidate paths:\n"
        + "\n".join(f"  - {p}" for p in candidate_paths)
        + "\nPlease download or place train_transaction.csv under Datasets/ or specify --dataset-dir."
    )


def main():
    parser = argparse.ArgumentParser(description="Flink historical batch feature engineering pipeline")
    parser.add_argument("--dataset-dir", default=None, help="Directory containing train_transaction.csv and train_identity.csv")
    args = parser.parse_args()

    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
    else:
        dataset_dir = resolve_dataset_dir()

    tx = dataset_dir / "train_transaction.csv"
    identity = dataset_dir / "train_identity.csv"

    if not tx.is_file():
        raise FileNotFoundError(f"CRITICAL ERROR: Transaction dataset not found at: {tx}")
    if not identity.is_file():
        raise FileNotFoundError(f"CRITICAL ERROR: Identity dataset not found at: {identity}")

    print("[Flink Batch] Loading IEEE-CIS...")
    print(f"[Flink Batch] Chunk size: {CHUNK_SIZE:,}")

    transaction_chunks, identities = load_ieee_cis(
        tx,
        identity,
        TX_COLUMNS,
        ID_COLUMNS,
        CHUNK_SIZE
    )

    output_dir = ROOT / "flink" / "output" / "batch"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_features = []
    total_rows = 0
    chunk_number = 0

    for transactions in transaction_chunks:
        chunk_number += 1
        total_rows += len(transactions)

        print(
            f"[Flink Batch] Processing chunk "
            f"{chunk_number:,} | rows: {len(transactions):,} "
            f"| total: {total_rows:,}"
        )

        transactions = clean_transactions(transactions)

        joined = join_transaction_identity(
            transactions,
            identities
        )

        features = build_historical_features(joined)

        all_features.append(features)

        del transactions
        del joined

    if not all_features:
        raise RuntimeError("No transaction data was processed.")

    print("[Flink Batch] Combining processed chunks...")

    features = pd.concat(
        all_features,
        ignore_index=True
    )

    output = write_features(
        features,
        output_dir
    )

    print(f"[Flink Batch] Processed rows: {total_rows:,}")
    print(f"[Flink Batch] Feature rows: {len(features):,}")
    print(f"[Flink Batch] Output: {output}")


if __name__ == "__main__":
    main()
