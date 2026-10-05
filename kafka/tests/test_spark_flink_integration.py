"""
Unit and Contract Verification Test for Kafka -> Spark & Flink Integration
Verifies data schema, key-partitioning affinity, and JSON serialization compatibility.
"""

import json
import sys
import unittest
from pathlib import Path

# Add project root and subdirectories to sys.path
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "kafka" / "transaction_generator"))
sys.path.insert(0, str(base_dir / "kafka" / "consumer"))

# pyrefly: ignore [missing-import]
# type: ignore
from dataset_loader import normalize_record
# pyrefly: ignore [missing-import]
# type: ignore
from flink_cep_consumer import StatefulVelocityWindow


class TestKafkaSparkFlinkContract(unittest.TestCase):

    def test_ieee_cis_normalization_schema(self):
        """Verifies IEEE-CIS row normalizes with all essential Spark & Flink fields."""
        raw_row = {
            "TransactionID": "2987000",
            "isFraud": "1",
            "TransactionDT": "86400",
            "TransactionAmt": "125.50",
            "ProductCD": "W",
            "card1": "13926",
            "card2": "150.0",
            "card4": "discover",
            "card6": "credit",
            "addr1": "315.0",
            "addr2": "87.0",
            "P_emaildomain": "gmail.com",
            "DeviceType": "desktop",
        }

        normalized = normalize_record(raw_row, source_type="ieee_cis")

        # Verify key fields
        self.assertEqual(normalized["transaction_id"], "TX-2987000")
        self.assertEqual(normalized["card_id"], "CARD-13926")
        self.assertEqual(normalized["amount"], 125.50)
        self.assertTrue(normalized["is_fraud"])
        self.assertEqual(normalized["device_type"], "web")

        # Verify JSON serializability
        json_bytes = json.dumps(normalized).encode("utf-8")
        self.assertTrue(len(json_bytes) > 0)

        # Deserialize test
        reloaded = json.loads(json_bytes.decode("utf-8"))
        self.assertEqual(reloaded["transaction_id"], "TX-2987000")

    def test_credit_card_10k_schema(self):
        """Verifies Credit Card 10k normalization schema."""
        raw_row = {
            "transaction_id": "101",
            "amount": "4500.00",
            "cardholder_age": "34",
            "merchant_category": "Electronics",
            "device_trust_score": "30",
            "location_mismatch": "1",
            "foreign_transaction": "1",
            "velocity_last_24h": "7",
            "is_fraud": "1",
        }

        normalized = normalize_record(raw_row)
        self.assertEqual(normalized["transaction_id"], "TX-101")
        self.assertEqual(normalized["amount"], 4500.00)
        self.assertTrue(normalized["is_fraud"])
        self.assertEqual(normalized["country"], "FOREIGN")
        self.assertEqual(normalized["location_mismatch"], True)

    def test_flink_stateful_velocity_window(self):
        """Tests that velocity window alerts when >3 transactions occur for same card."""
        cep = StatefulVelocityWindow(window_seconds=60, max_velocity=3, max_spend=5000.0)
        card = "CARD-TEST-99"

        # First 3 transactions -> Normal
        alerts1, count1, spend1 = cep.process_event(card, 100.0, 50.0, "TX-1", "US")
        self.assertEqual(len(alerts1), 0)
        self.assertEqual(count1, 1)

        alerts2, count2, spend2 = cep.process_event(card, 110.0, 60.0, "TX-2", "US")
        self.assertEqual(len(alerts2), 0)
        self.assertEqual(count2, 2)

        alerts3, count3, spend3 = cep.process_event(card, 120.0, 70.0, "TX-3", "US")
        self.assertEqual(len(alerts3), 0)
        self.assertEqual(count3, 3)

        # 4th transaction within window -> Trigger velocity alert!
        alerts4, count4, spend4 = cep.process_event(card, 130.0, 80.0, "TX-4", "US")
        self.assertTrue(len(alerts4) >= 1)
        self.assertIn("CEP-VELOCITY-ALERT", alerts4[0])
        self.assertEqual(count4, 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
