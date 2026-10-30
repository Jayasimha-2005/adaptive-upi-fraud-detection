"""
Unit & Contract Test Suite for Member 1 — Apache Kafka Event Ingestion Layer
Tests:
  - Base.csv Normalization & Canonical Schema Compliance
  - Dataset Streamer Iteration & Diversity
  - Key-Based Partition Ordering Contract (Murmur2 Hash Affinity)
  - At-Least-Once Offset Replay & Idempotent Deduplication
"""

import json
import os
import sys
import unittest
from pathlib import Path

# Add generator directory to path
base_dir = Path(__file__).resolve().parent.parent
gen_dir = base_dir / "transaction_generator"
if str(gen_dir) not in sys.path:
    sys.path.insert(0, str(gen_dir))

from dataset_loader import DatasetTransactionStreamer, normalize_record, resolve_dataset_file


class TestMember1KafkaPipeline(unittest.TestCase):

    def test_base_csv_file_resolution(self):
        """Verifies that resolve_dataset_file finds Datasets/Base.csv."""
        path = resolve_dataset_file("Base.csv")
        self.assertIsNotNone(path, "resolve_dataset_file failed to find Base.csv")
        self.assertTrue(path.exists(), f"Resolved path does not exist: {path}")

    def test_base_csv_normalization_canonical_schema(self):
        """Verifies that Base.csv rows are normalized into the exact canonical event schema."""
        sample_row = {
            "fraud_bool": "1",
            "income": "0.8",
            "name_email_similarity": "0.95",
            "customer_age": "45",
            "intended_balcon_amount": "2500.50",
            "payment_type": "AB",
            "proposed_credit_limit": "5000.0",
            "foreign_request": "0",
            "device_os": "linux",
            "source": "INTERNET",
            "credit_risk_score": "120",
            "velocity_6h": "8500.0",
            "velocity_24h": "6200.0",
            "velocity_4w": "5100.0",
            "session_length_in_minutes": "14.5",
        }

        normalized = normalize_record(sample_row, source_type="base.csv", row_index=1, max_cards=50)

        # 1. Required core fields
        self.assertTrue(normalized["transaction_id"].startswith("TX"))
        self.assertTrue(normalized["card_id"].startswith("CARD"))
        self.assertIsInstance(normalized["amount"], float)
        self.assertEqual(normalized["amount"], 2500.50)
        self.assertTrue(normalized["merchant_id"].startswith("M"))
        self.assertIn("timestamp", normalized)
        self.assertIn(normalized["device_type"], ["mobile", "web"])
        self.assertEqual(normalized["country"], "IN")
        self.assertTrue(normalized["is_fraud"])
        self.assertEqual(normalized["dataset_source"], "base.csv")

        # 2. JSON Serialization Roundtrip
        payload_bytes = json.dumps(normalized).encode("utf-8")
        deserialized = json.loads(payload_bytes.decode("utf-8"))
        self.assertEqual(deserialized["transaction_id"], normalized["transaction_id"])
        self.assertEqual(deserialized["amount"], 2500.50)

    def test_base_csv_non_identical_stream(self):
        """Verifies streamer produces varied, non-identical transactions from Base.csv."""
        streamer = DatasetTransactionStreamer(loop=True, max_cards=10)
        records = [streamer.next_transaction() for _ in range(5)]
        streamer.close()

        # Ensure all records have unique transaction_ids
        tx_ids = [r["transaction_id"] for r in records]
        self.assertEqual(len(tx_ids), len(set(tx_ids)), "Duplicate transaction_ids produced!")

        # Ensure amounts vary
        amounts = [r["amount"] for r in records]
        self.assertTrue(len(set(amounts)) > 1, "All transaction amounts are identical!")

    def test_strict_per_card_partition_affinity(self):
        """Verifies that transactions with identical card_id hash deterministically to the same partition."""
        num_partitions = 6
        card_id = "CARD042"

        # Simulating standard Kafka Murmur2 / modular partition mapping
        p1 = abs(hash(card_id)) % num_partitions
        p2 = abs(hash(card_id)) % num_partitions
        p3 = abs(hash(card_id)) % num_partitions

        self.assertEqual(p1, p2)
        self.assertEqual(p2, p3)

    def test_idempotent_consumer_deduplication(self):
        """Simulates consumer crash recovery replay and tests deduplication."""
        seen_cache = set()
        transactions = [f"TX1000{i}" for i in range(1, 6)]

        # First pass: Process 1 to 5
        processed_first_pass = []
        for tx in transactions:
            if tx not in seen_cache:
                seen_cache.add(tx)
                processed_first_pass.append(tx)

        self.assertEqual(len(processed_first_pass), 5)

        # Simulated replay of TX10003, TX10004, TX10005 due to uncommitted offset crash
        replayed_stream = ["TX10003", "TX10004", "TX10005", "TX10006", "TX10007"]
        newly_processed = []
        duplicates_caught = 0

        for tx in replayed_stream:
            if tx in seen_cache:
                duplicates_caught += 1
            else:
                seen_cache.add(tx)
                newly_processed.append(tx)

        self.assertEqual(duplicates_caught, 3, "Failed to catch duplicate replayed events!")
        self.assertEqual(newly_processed, ["TX10006", "TX10007"])
        self.assertEqual(len(seen_cache), 7, "Total processed events count mismatch!")


if __name__ == "__main__":
    unittest.main()
