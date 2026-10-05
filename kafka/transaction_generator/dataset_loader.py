"""
Dataset Loader & Real-Time Financial Transaction Streamer
Supports IEEE-CIS Fraud Detection, Credit Card Fraud 10k, PaySim, and ULB Credit Card datasets.
Provides normalized transaction dictionaries for Kafka producer ingestion and benchmarking.
"""

import csv
import json
import os
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


def get_dataset_paths() -> Dict[str, Path]:
    """Finds all available datasets relative to workspace root."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    datasets_dir = base_dir / "Datasets"

    paths = {
        "ieee_cis": datasets_dir / "IEEE CIS-20260829T103704Z-1-001" / "IEEE CIS" / "train_transaction.csv",
        "ieee_cis_raw": datasets_dir / "raw" / "train_transaction.csv",
        "credit_card_10k": datasets_dir / "Credit card Fraud detection" / "credit_card_fraud_10k.csv",
        "ulb_creditcard": datasets_dir / "ulb_creditcard" / "creditcard.csv",
        "paysim": datasets_dir / "Paysim" / "paysim dataset.csv",
        "baf_base": datasets_dir / "BAF" / "Base.csv",
    }
    return paths


def resolve_dataset_file(name_or_path: Optional[str] = None) -> Optional[Path]:
    """Resolves dataset name or custom filepath to an existing Path object."""
    if not name_or_path or name_or_path.lower() in ("default", "ieee_cis", "ieee-cis", "ieee"):
        known = get_dataset_paths()
        for key in ["ieee_cis", "ieee_cis_raw", "credit_card_10k", "ulb_creditcard", "paysim"]:
            p = known.get(key)
            if p and p.exists():
                return p
        return None

    if name_or_path.lower() in ("creditcard", "credit_card", "credit_card_10k", "10k"):
        p = get_dataset_paths()["credit_card_10k"]
        if p.exists():
            return p

    if name_or_path.lower() in ("paysim", "pay_sim"):
        p = get_dataset_paths()["paysim"]
        if p.exists():
            return p

    if name_or_path.lower() in ("ulb", "ulb_creditcard"):
        p = get_dataset_paths()["ulb_creditcard"]
        if p.exists():
            return p

    if name_or_path.lower() in ("baf", "baf_base"):
        p = get_dataset_paths()["baf_base"]
        if p.exists():
            return p

    # Custom path check
    custom_p = Path(name_or_path)
    if custom_p.exists():
        return custom_p

    return None


def normalize_record(row: Dict[str, str], source_type: str = "auto") -> Dict[str, Any]:
    """Standardizes dataset rows into unified real-time transaction event schema."""
    now_iso = datetime.now(timezone.utc).isoformat()
    now_epoch = time.time()

    # Detect IEEE-CIS
    if "TransactionID" in row or source_type == "ieee_cis":
        tx_id = str(row.get("TransactionID", f"TX-{uuid.uuid4().hex[:8]}"))
        card_raw = row.get("card1", "0")
        card_id = f"CARD-{card_raw}" if card_raw else "CARD-UNKNOWN"
        try:
            amt = float(row.get("TransactionAmt", 0.0))
        except (ValueError, TypeError):
            amt = 0.0
        
        is_fraud = int(row.get("isFraud", 0)) == 1
        merchant = str(row.get("ProductCD", "W"))
        device = "web" if row.get("DeviceType") == "desktop" else "mobile"
        country = str(row.get("addr2", "US") or "US")

        return {
            "transaction_id": f"TX-{tx_id}",
            "card_id": card_id,
            "amount": round(amt, 2),
            "merchant_id": f"M-{merchant}",
            "timestamp": now_iso,
            "epoch_timestamp": now_epoch,
            "device_type": device,
            "country": country,
            "is_fraud": is_fraud,
            "dataset_source": "ieee_cis",
            "card2": row.get("card2", ""),
            "card4": row.get("card4", ""),
            "card6": row.get("card6", ""),
            "P_emaildomain": row.get("P_emaildomain", ""),
        }

    # Detect Credit Card 10k
    if "merchant_category" in row or "device_trust_score" in row:
        tx_id = str(row.get("transaction_id", f"{uuid.uuid4().hex[:8]}"))
        card_age = row.get("cardholder_age", "0")
        card_id = f"CARD-HOLDER-{card_age}-{int(tx_id) % 500:03d}" if tx_id.isdigit() else f"CARD-{card_age}"
        try:
            amt = float(row.get("amount", 0.0))
        except (ValueError, TypeError):
            amt = 0.0
        
        is_fraud = int(row.get("is_fraud", 0)) == 1
        merchant = str(row.get("merchant_category", "Retail"))
        loc_mismatch = int(row.get("location_mismatch", 0)) == 1
        foreign = int(row.get("foreign_transaction", 0)) == 1

        return {
            "transaction_id": f"TX-{tx_id}",
            "card_id": card_id,
            "amount": round(amt, 2),
            "merchant_id": f"CAT-{merchant}",
            "timestamp": now_iso,
            "epoch_timestamp": now_epoch,
            "device_type": "mobile" if int(row.get("device_trust_score", 50)) < 50 else "web",
            "country": "FOREIGN" if foreign else "LOCAL",
            "is_fraud": is_fraud,
            "dataset_source": "credit_card_10k",
            "velocity_24h": row.get("velocity_last_24h", "0"),
            "location_mismatch": loc_mismatch,
        }

    # Detect Paysim
    if "nameOrig" in row or "oldbalanceOrg" in row:
        sender = str(row.get("nameOrig", "UNKNOWN"))
        receiver = str(row.get("nameDest", "UNKNOWN"))
        try:
            amt = float(row.get("amount", 0.0))
        except (ValueError, TypeError):
            amt = 0.0

        is_fraud = int(row.get("isFraud", 0)) == 1
        tx_type = str(row.get("type", "PAYMENT"))

        return {
            "transaction_id": f"TX-PS-{sender[-6:]}-{random.randint(1000, 9999)}",
            "card_id": sender,
            "amount": round(amt, 2),
            "merchant_id": receiver,
            "timestamp": now_iso,
            "epoch_timestamp": now_epoch,
            "device_type": "mobile",
            "country": "GLOBAL",
            "is_fraud": is_fraud,
            "dataset_source": "paysim",
            "type": tx_type,
            "oldbalanceOrg": row.get("oldbalanceOrg", "0"),
            "newbalanceOrig": row.get("newbalanceOrig", "0"),
        }

    # Generic fallback
    tx_id = row.get("transaction_id") or row.get("id") or str(uuid.uuid4().hex[:8])
    card_id = row.get("card_id") or row.get("account_id") or f"CARD-{random.randint(1, 100):03d}"
    try:
        amt = float(row.get("amount", row.get("Amount", 100.0)))
    except (ValueError, TypeError):
        amt = 100.0

    is_fraud = bool(int(row.get("is_fraud", row.get("Class", 0))))

    return {
        "transaction_id": f"TX-{tx_id}",
        "card_id": str(card_id),
        "amount": round(amt, 2),
        "merchant_id": str(row.get("merchant_id", "M100")),
        "timestamp": now_iso,
        "epoch_timestamp": now_epoch,
        "device_type": "web",
        "country": "US",
        "is_fraud": is_fraud,
        "dataset_source": "custom_dataset",
        "raw": {k: v for k, v in row.items() if k not in ["transaction_id", "amount"]},
    }


class DatasetTransactionStreamer:
    """Streams financial transactions continuously or in fixed batches from CSV datasets."""

    def __init__(self, dataset_path: Optional[str] = None, loop: bool = True):
        self.dataset_file = resolve_dataset_file(dataset_path)
        self.loop = loop
        self.file_handle = None
        self.csv_reader = None
        self._total_streamed = 0
        self._fraud_streamed = 0

        if self.dataset_file and self.dataset_file.exists():
            self._open_file()
        else:
            self.dataset_file = None

    def _open_file(self):
        if self.file_handle and not self.file_handle.closed:
            self.file_handle.close()
        self.file_handle = open(self.dataset_file, mode="r", newline="", encoding="utf-8")
        self.csv_reader = csv.DictReader(self.file_handle)

    def next_transaction(self) -> Dict[str, Any]:
        """Fetches next transaction from dataset or loops back to beginning."""
        if not self.dataset_file or not self.csv_reader:
            return self._synthetic_fallback()

        try:
            row = next(self.csv_reader)
        except StopIteration:
            if self.loop:
                self._open_file()
                row = next(self.csv_reader)
            else:
                return None

        record = normalize_record(row)
        self._total_streamed += 1
        if record.get("is_fraud"):
            self._fraud_streamed += 1
        return record

    def _synthetic_fallback(self) -> Dict[str, Any]:
        """Generates synthetic record if dataset file cannot be loaded."""
        card_id = f"CARD{random.randint(1, 50):03d}"
        amt = round(random.uniform(10.0, 5000.0), 2)
        is_fraud = random.random() < 0.05
        return {
            "transaction_id": f"TX-SYN-{uuid.uuid4().hex[:8]}",
            "card_id": card_id,
            "amount": amt,
            "merchant_id": f"M{random.randint(101, 110)}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "epoch_timestamp": time.time(),
            "device_type": random.choice(["mobile", "web", "pos"]),
            "country": random.choice(["IN", "US", "UK", "SG"]),
            "is_fraud": is_fraud,
            "dataset_source": "synthetic_fallback",
        }

    def close(self):
        if self.file_handle and not self.file_handle.closed:
            self.file_handle.close()

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "dataset_path": str(self.dataset_file) if self.dataset_file else "Synthetic",
            "total_streamed": self._total_streamed,
            "fraud_streamed": self._fraud_streamed,
            "fraud_ratio": (self._fraud_streamed / max(1, self._total_streamed)),
        }
