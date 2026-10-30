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
        "base_csv": datasets_dir / "Base.csv",
        "baf_base": datasets_dir / "Base.csv",
        "baf_sub": datasets_dir / "BAF" / "Base.csv",
        "ieee_cis": datasets_dir / "IEEE CIS-20260829T103704Z-1-001" / "IEEE CIS" / "train_transaction.csv",
        "ieee_cis_raw": datasets_dir / "raw" / "train_transaction.csv",
        "credit_card_10k": datasets_dir / "Credit card Fraud detection" / "credit_card_fraud_10k.csv",
        "ulb_creditcard": datasets_dir / "ulb_creditcard" / "creditcard.csv",
        "paysim": datasets_dir / "Paysim" / "paysim dataset.csv",
    }
    return paths


def resolve_dataset_file(name_or_path: Optional[str] = None) -> Optional[Path]:
    """Resolves dataset name or custom filepath to an existing Path object."""
    known = get_dataset_paths()

    # If default or base.csv requested, check base_csv first
    if not name_or_path or name_or_path.lower() in ("default", "base", "base.csv", "baf", "baf_base", "datasets/base.csv"):
        for key in ["base_csv", "baf_base", "baf_sub", "ieee_cis", "ieee_cis_raw", "credit_card_10k", "ulb_creditcard", "paysim"]:
            p = known.get(key)
            if p and p.exists():
                return p
        return None

    if name_or_path.lower() in ("creditcard", "credit_card", "credit_card_10k", "10k"):
        p = known["credit_card_10k"]
        if p.exists():
            return p

    if name_or_path.lower() in ("paysim", "pay_sim"):
        p = known["paysim"]
        if p.exists():
            return p

    if name_or_path.lower() in ("ulb", "ulb_creditcard"):
        p = known["ulb_creditcard"]
        if p.exists():
            return p

    if name_or_path.lower() in ("ieee_cis", "ieee-cis", "ieee"):
        for key in ["ieee_cis", "ieee_cis_raw"]:
            p = known.get(key)
            if p and p.exists():
                return p

    # Custom path check (relative to cwd or workspace)
    custom_p = Path(name_or_path)
    if custom_p.exists():
        return custom_p

    base_dir = Path(__file__).resolve().parent.parent.parent
    rel_p = base_dir / name_or_path
    if rel_p.exists():
        return rel_p

    return None


def normalize_record(
    row: Dict[str, str],
    source_type: str = "auto",
    row_index: Optional[int] = None,
    max_cards: int = 100,
) -> Dict[str, Any]:
    """Standardizes dataset rows into unified real-time transaction event schema."""
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    now_epoch = time.time()

    # Detect Bank Account Fraud (BAF) Base.csv
    if "fraud_bool" in row or "intended_balcon_amount" in row or "proposed_credit_limit" in row or source_type in ("base", "baf", "base_csv"):
        raw_idx = (100001 + row_index) if row_index is not None else random.randint(100001, 999999)
        tx_id = f"TX{raw_idx}"

        # Assign deterministic, repeating card IDs from configured card pool for partition ordering
        card_seed = f"{row.get('customer_age', '30')}_{row.get('zip_count_4w', '1000')}_{row.get('payment_type', 'AA')}_{row.get('month', '0')}_{row.get('employment_status', 'CA')}"
        card_num = (abs(hash(card_seed)) % max(1, max_cards)) + 1
        card_id = f"CARD{card_num:03d}"

        try:
            balcon = float(row.get("intended_balcon_amount", 0.0) or 0.0)
            proposed_limit = float(row.get("proposed_credit_limit", 1500.0) or 1500.0)
            income = float(row.get("income", 0.5) or 0.5)
        except (ValueError, TypeError):
            balcon = 0.0
            proposed_limit = 1500.0
            income = 0.5

        if balcon > 0:
            amt = balcon
        else:
            amt = max(25.0, round((proposed_limit * 0.35) + abs(balcon) + (income * 500.0), 2))

        is_fraud = str(row.get("fraud_bool", "0")).strip() in ("1", "True", "true")

        ptype = str(row.get("payment_type", "AA")).strip()
        merchant_offset = (ord(ptype[0]) if len(ptype) > 0 else 65) % 10 * 5 + (ord(ptype[-1]) if len(ptype) > 1 else 65) % 5
        merchant_id = f"M{101 + merchant_offset}"

        dev_os = str(row.get("device_os", "other")).lower()
        src = str(row.get("source", "INTERNET")).upper()
        if dev_os in ["android", "ios"] or src == "TELEAPP":
            device = "mobile"
        elif dev_os in ["windows", "linux", "macintosh", "x11"]:
            device = "web"
        else:
            device = "mobile"

        foreign = str(row.get("foreign_request", "0")).strip() in ("1", "True", "true")
        country = "US" if foreign else "IN"

        return {
            "transaction_id": tx_id,
            "card_id": card_id,
            "amount": round(amt, 2),
            "merchant_id": merchant_id,
            "timestamp": now_iso,
            "epoch_timestamp": now_epoch,
            "device_type": device,
            "country": country,
            "is_fraud": is_fraud,
            "dataset_source": "base.csv",
            # Temporal period for Phase 4 concept drift monitoring
            "month": int(float(row.get("month", 0) or 0)),
            # Additional features for downstream ML model scoring, CEP & Phase 4 Drift
            "fraud_bool": 1 if is_fraud else 0,
            "income": income,
            "name_email_similarity": float(row.get("name_email_similarity", 0.5) or 0.5),
            "prev_address_months_count": int(float(row.get("prev_address_months_count", 0) or 0)),
            "current_address_months_count": int(float(row.get("current_address_months_count", 0) or 0)),
            "customer_age": int(float(row.get("customer_age", 30) or 30)),
            "days_since_request": float(row.get("days_since_request", 0.0) or 0.0),
            "intended_balcon_amount": float(row.get("intended_balcon_amount", 0.0) or 0.0),
            "zip_count_4w": int(float(row.get("zip_count_4w", 0) or 0)),
            "velocity_6h": float(row.get("velocity_6h", 0.0) or 0.0),
            "velocity_24h": float(row.get("velocity_24h", 0.0) or 0.0),
            "velocity_4w": float(row.get("velocity_4w", 0.0) or 0.0),
            "bank_branch_count_8w": int(float(row.get("bank_branch_count_8w", 0) or 0)),
            "date_of_birth_distinct_emails_4w": int(float(row.get("date_of_birth_distinct_emails_4w", 0) or 0)),
            "employment_status": str(row.get("employment_status", "CA")),
            "credit_risk_score": int(float(row.get("credit_risk_score", 100) or 100)),
            "email_is_free": int(float(row.get("email_is_free", 0) or 0)),
            "housing_status": str(row.get("housing_status", "OA")),
            "phone_home_valid": int(float(row.get("phone_home_valid", 0) or 0)),
            "phone_mobile_valid": int(float(row.get("phone_mobile_valid", 0) or 0)),
            "bank_months_count": int(float(row.get("bank_months_count", 0) or 0)),
            "has_other_cards": int(float(row.get("has_other_cards", 0) or 0)),
            "proposed_credit_limit": proposed_limit,
            "foreign_request": 1 if foreign else 0,
            "source": src,
            "session_length_in_minutes": float(row.get("session_length_in_minutes", 10.0) or 10.0),
            "device_os": dev_os,
            "keep_alive_session": int(float(row.get("keep_alive_session", 0) or 0)),
            "device_distinct_emails_8w": int(float(row.get("device_distinct_emails_8w", 0) or 0)),
            "payment_type": ptype,
            "raw_features": {k: float(v) if k in ["income", "intended_balcon_amount", "proposed_credit_limit", "velocity_6h", "velocity_24h", "velocity_4w", "name_email_similarity", "days_since_request", "session_length_in_minutes"] and v != "" else v for k, v in row.items()},
        }

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

    def __init__(self, dataset_path: Optional[str] = None, loop: bool = True, max_cards: int = 100):
        self.dataset_file = resolve_dataset_file(dataset_path)
        self.loop = loop
        self.max_cards = max_cards
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

        record = normalize_record(row, row_index=self._total_streamed, max_cards=self.max_cards)
        self._total_streamed += 1
        if record.get("is_fraud"):
            self._fraud_streamed += 1
        return record

    def _synthetic_fallback(self) -> Dict[str, Any]:
        """Generates synthetic record if dataset file cannot be loaded."""
        idx = 100001 + self._total_streamed
        card_id = f"CARD{random.randint(1, self.max_cards):03d}"
        amt = round(random.uniform(25.0, 5000.0), 2)
        is_fraud = random.random() < 0.05
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        self._total_streamed += 1
        if is_fraud:
            self._fraud_streamed += 1
        return {
            "transaction_id": f"TX{idx}",
            "card_id": card_id,
            "amount": amt,
            "merchant_id": f"M{random.randint(101, 110)}",
            "timestamp": now_iso,
            "epoch_timestamp": time.time(),
            "device_type": random.choice(["mobile", "web"]),
            "country": random.choice(["IN", "US"]),
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
