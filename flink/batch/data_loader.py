from pathlib import Path
import pandas as pd


def load_identity(identity_path, identity_columns=None):
    """Load the smaller IEEE-CIS identity dataset."""
    if identity_path and Path(identity_path).exists():
        return pd.read_csv(
            Path(identity_path),
            usecols=identity_columns,
            low_memory=False
        )
    return None


def load_transaction_chunks(
    transaction_path,
    transaction_columns=None,
    chunksize=50_000
):
    """Read the large transaction CSV in memory-safe chunks."""
    return pd.read_csv(
        Path(transaction_path),
        usecols=transaction_columns,
        chunksize=chunksize,
        low_memory=False
    )


def load_ieee_cis(
    transaction_path,
    identity_path=None,
    transaction_columns=None,
    identity_columns=None,
    chunksize=50_000
):
    """
    Backward-compatible loader.

    Returns:
        transaction_chunks: pandas TextFileReader
        identity: pandas DataFrame
    """
    transactions = load_transaction_chunks(
        transaction_path,
        transaction_columns,
        chunksize
    )

    identity = load_identity(
        identity_path,
        identity_columns
    )

    return transactions, identity
