import argparse
import json
import os
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Import dataset loader
try:
    from dataset_loader import DatasetTransactionStreamer, resolve_dataset_file
except ImportError:
    from .dataset_loader import DatasetTransactionStreamer, resolve_dataset_file

DEFAULT_DEVICES = ["mobile", "web", "tablet"]
DEFAULT_COUNTRIES = ["IN", "US", "UK", "SG", "AE"]
DEFAULT_MERCHANTS = ["M101", "M102", "M103", "M104", "M105"]

# Shared global streamer instance for lightweight procedural access
_GLOBAL_STREAMER = None


def get_global_streamer(dataset_path=None):
    global _GLOBAL_STREAMER
    if _GLOBAL_STREAMER is None:
        _GLOBAL_STREAMER = DatasetTransactionStreamer(dataset_path=dataset_path, loop=True)
    return _GLOBAL_STREAMER


def build_cards(number_of_cards):
    return [f"CARD{i:03d}" for i in range(1, number_of_cards + 1)]


def generate_amount(distribution):
    if distribution == "uniform":
        return round(random.uniform(10, 10000), 2)
    if distribution == "normal":
        amount = random.gauss(3000, 1500)
        amount = max(10, min(10000, amount))
        return round(amount, 2)
    if distribution == "small":
        return round(random.uniform(10, 1000), 2)
    if distribution == "large":
        return round(random.uniform(1000, 10000), 2)
    raise ValueError(f"Unknown amount distribution: {distribution}")


def generate_synthetic_transaction(cards, fraud_ratio, amount_distribution):
    return {
        "transaction_id": "TX-" + str(uuid.uuid4())[:8],
        "card_id": random.choice(cards),
        "amount": generate_amount(amount_distribution),
        "merchant_id": random.choice(DEFAULT_MERCHANTS),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "epoch_timestamp": time.time(),
        "device_type": random.choice(DEFAULT_DEVICES),
        "country": random.choice(DEFAULT_COUNTRIES),
        "is_fraud": random.random() < fraud_ratio,
        "dataset_source": "synthetic",
    }


def generate_transaction(
    cards=None,
    fraud_ratio=0.05,
    amount_distribution="uniform",
    dataset=None,
    use_dataset=True,
):
    """
    Generate or replay a transaction.
    By default, pulls from real dataset (IEEE-CIS, Credit Card 10k, etc.).
    Falls back to synthetic generation if use_dataset is False or if dataset unavailable.
    """
    if use_dataset:
        streamer = get_global_streamer(dataset)
        record = streamer.next_transaction()
        if record:
            return record

    if not cards:
        cards = build_cards(10)
    return generate_synthetic_transaction(cards, fraud_ratio, amount_distribution)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Dataset-powered & Configurable Financial Transaction Generator"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="default",
        help="Dataset name ('ieee_cis', 'credit_card_10k', 'paysim', 'synthetic') or CSV path",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of transactions to generate/stream",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=1,
        help="Target transaction generation rate per second",
    )
    parser.add_argument(
        "--fraud-ratio",
        type=float,
        default=0.05,
        help="Fraud ratio between 0.0 and 1.0 (for synthetic mode)",
    )
    parser.add_argument(
        "--cards",
        type=int,
        default=10,
        help="Number of unique cards (for synthetic mode)",
    )
    parser.add_argument(
        "--amount-distribution",
        choices=["uniform", "normal", "small", "large"],
        default="uniform",
        help="Amount distribution (for synthetic mode)",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.count <= 0:
        raise ValueError("Count must be greater than 0")
    if args.rate <= 0:
        raise ValueError("Rate must be greater than 0")

    use_dataset = args.dataset.lower() != "synthetic"
    streamer = DatasetTransactionStreamer(dataset_path=args.dataset, loop=True) if use_dataset else None

    cards = build_cards(args.cards)
    interval = 1.0 / args.rate

    print("=" * 75)
    print("FINANCIAL TRANSACTION STREAM GENERATOR (DATASET-ENABLED)")
    print("=" * 75)
    print(f"Data Source         : {streamer.dataset_file if streamer and streamer.dataset_file else 'Synthetic'}")
    print(f"Transactions        : {args.count}")
    print(f"Target rate         : {args.rate} tx/sec")
    if not streamer or not streamer.dataset_file:
        print(f"Fraud ratio         : {args.fraud_ratio:.2%}")
        print(f"Number of cards     : {args.cards}")
        print(f"Amount distribution : {args.amount_distribution}")
    print("=" * 75)
    print()

    next_send_time = time.time()

    for idx in range(1, args.count + 1):
        now = time.time()
        if now < next_send_time:
            time.sleep(next_send_time - now)

        if streamer and streamer.dataset_file:
            transaction = streamer.next_transaction()
        else:
            transaction = generate_synthetic_transaction(cards, args.fraud_ratio, args.amount_distribution)

        print(json.dumps(transaction))
        next_send_time += interval

    if streamer:
        streamer.close()


if __name__ == "__main__":
    main()
