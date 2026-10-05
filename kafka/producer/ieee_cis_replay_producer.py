import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# Safe Windows stdout encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from kafka import KafkaProducer
from kafka.errors import KafkaError

DEFAULT_BROKER = "localhost:9092"
DEFAULT_TOPIC = "ieee_cis_transactions"


def find_default_dataset():
    """Locate train_transaction.csv relative to project root across Windows and Linux. Fails loudly if missing."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    possible_paths = [
        base_dir / "Datasets" / "IEEE CIS-20260829T103704Z-1-001" / "IEEE CIS" / "train_transaction.csv",
        base_dir / "Datasets" / "raw" / "train_transaction.csv",
        base_dir / "Datasets" / "train_transaction.csv",
    ]
    for p in possible_paths:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        "CRITICAL ERROR: IEEE-CIS dataset (train_transaction.csv) not found in candidate paths:\n"
        + "\n".join(f"  - {p}" for p in possible_paths)
        + "\nPlease download or place train_transaction.csv under Datasets/ or specify --dataset."
    )


def percentile(values, percentage):
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int((percentage / 100) * len(sorted_values))
    if index >= len(sorted_values):
        index = len(sorted_values) - 1
    return sorted_values[index]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="IEEE-CIS Entire Dataset Streaming Replay Producer for Apache Kafka"
    )
    parser.add_argument(
        "--broker",
        default=DEFAULT_BROKER,
        help=f"Kafka bootstrap broker (default: {DEFAULT_BROKER})",
    )
    parser.add_argument(
        "--topic",
        default=DEFAULT_TOPIC,
        help=f"Target Kafka topic (default: {DEFAULT_TOPIC})",
    )
    parser.add_argument(
        "--dataset",
        default=find_default_dataset(),
        help="Path to IEEE-CIS train_transaction.csv",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=1000,
        help="Target streaming replay rate (tx/sec, 0 for maximum unthrottled throughput, default: 1000)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Number of transactions to stream (0 for ENTIRE 590,540 dataset, default: 0)",
    )
    parser.add_argument(
        "--acks",
        default="all",
        choices=["0", "1", "all"],
        help="Kafka ACK durability mode (default: all)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        default=True,
        help="Send core features for ultra-fast streaming (TransactionID, isFraud, TransactionAmt, card1..card6, ProductCD, etc.)",
    )
    parser.add_argument(
        "--full-payload",
        action="store_true",
        help="Send full raw payload containing all 434 columns",
    )
    return parser.parse_args()


def clean_record(row, compact=True):
    """Format and cast CSV row to JSON-serializable dictionary."""
    if compact:
        return {
            "TransactionID": int(row.get("TransactionID", 0)),
            "isFraud": int(row.get("isFraud", 0)),
            "TransactionDT": int(row.get("TransactionDT", 0)),
            "TransactionAmt": float(row.get("TransactionAmt", 0.0)),
            "ProductCD": row.get("ProductCD", ""),
            "card1": str(row.get("card1", "UNKNOWN")),
            "card2": row.get("card2", ""),
            "card3": row.get("card3", ""),
            "card4": row.get("card4", ""),
            "card5": row.get("card5", ""),
            "card6": row.get("card6", ""),
            "addr1": row.get("addr1", ""),
            "addr2": row.get("addr2", ""),
            "P_emaildomain": row.get("P_emaildomain", ""),
            "R_emaildomain": row.get("R_emaildomain", ""),
            "timestamp": time.time(),
        }

    clean = {}
    for k, v in row.items():
        if v == "" or v is None:
            clean[k] = None
        else:
            try:
                if "." in v:
                    clean[k] = float(v)
                else:
                    clean[k] = int(v)
            except ValueError:
                clean[k] = v
    clean["_replay_timestamp"] = time.time()
    return clean


def stream_dataset(broker=DEFAULT_BROKER, topic=DEFAULT_TOPIC, dataset_path=None, rate=1000, max_count=0, acks="all", compact=True):
    """Streams IEEE-CIS transactions into Kafka with configurable throughput and durability."""
    if not dataset_path:
        dataset_path = find_default_dataset()

    if not os.path.exists(dataset_path):
        print(f"[-] ERROR: Dataset file not found at: {dataset_path}")
        return {"successful": 0, "failed": 0, "fraud_count": 0}

    ack_setting = 0 if acks == "0" else (1 if acks == "1" else "all")
    enable_idempotence = (acks == "all")

    producer = KafkaProducer(
        bootstrap_servers=[broker],
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: str(k).encode("utf-8"),
        acks=ack_setting,
        retries=5 if enable_idempotence else 0,
        enable_idempotence=enable_idempotence,
        batch_size=65536,
        linger_ms=5,
    )

    interval = 1.0 / rate if rate > 0 else 0
    successful = 0
    failed = 0
    fraud_count = 0
    total_volume = 0.0

    start_bench_time = time.time()
    next_send_time = start_bench_time

    print("[+] Starting high-speed stream into Kafka...")

    try:
        with open(dataset_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            for idx, row in enumerate(reader, start=1):
                if max_count > 0 and idx > max_count:
                    break

                now = time.time()
                if interval > 0 and now < next_send_time:
                    time.sleep(next_send_time - now)

                record = clean_record(row, compact=compact)
                card_key = str(row.get("card1", "0"))
                is_fraud = int(row.get("isFraud", 0))
                amt = float(row.get("TransactionAmt", 0.0))
                
                total_volume += amt
                if is_fraud == 1:
                    fraud_count += 1

                try:
                    producer.send(topic, key=card_key, value=record)
                    successful += 1
                except Exception as e:
                    failed += 1
                    if failed <= 5:
                        print(f"[!] Send failed at record {idx}: {e}")

                if idx % 1000 == 0:
                    elapsed = time.time() - start_bench_time
                    curr_rate = successful / elapsed if elapsed > 0 else 0
                    print(
                        f"[{idx:06d}] Replayed: TX-{record.get('TransactionID')} | "
                        f"Card1: {card_key:<8} | Amt: ${amt:>7.2f} | "
                        f"Fraud: {fraud_count:4d} | Rate: {curr_rate:.1f} tx/s"
                    )

                next_send_time += interval

        print("\n[+] Flushing remaining buffered messages to Kafka broker...")
        producer.flush()

    finally:
        producer.close()

    total_time = time.time() - start_bench_time
    actual_rate = successful / total_time if total_time > 0 else 0

    print()
    print("=" * 80)
    print("  IEEE-CIS FULL DATASET STREAMING COMPLETE")
    print("=" * 80)
    print(f"Total Transactions Replayed : {successful:,}")
    print(f"Failed / Lost Transactions  : {failed:,}")
    print(f"Fraudulent Transactions     : {fraud_count:,} ({(fraud_count / max(1, successful)) * 100:.2f}%)")
    print(f"Total Monitored Volume      : ${total_volume:,.2f}")
    print(f"Total Time Taken            : {total_time:.2f} s")
    print(f"Effective Ingestion Rate    : {actual_rate:,.2f} tx/sec")
    print("=" * 80)

    return {
        "successful": successful,
        "failed": failed,
        "fraud_count": fraud_count,
        "total_volume": total_volume,
        "total_time": total_time,
        "rate": actual_rate,
    }


def main():
    args = parse_arguments()
    is_compact = not args.full_payload

    print("=" * 80)
    print("  IEEE-CIS ENTIRE DATASET REPLAY STREAMING PRODUCER")
    print("=" * 80)
    print(f"Broker           : {args.broker}")
    print(f"Topic            : {args.topic}")
    print(f"Dataset Path     : {args.dataset}")
    print(f"Target Rate      : {'MAX (Unthrottled)' if args.rate <= 0 else f'{args.rate} tx/sec'}")
    print(f"Target Count     : {'ALL (Entire Dataset ~590,540 records)' if args.count == 0 else args.count}")
    print(f"ACK Mode         : {args.acks}")
    print(f"Payload Mode     : {'Compact (Core Features - Fast)' if is_compact else 'Full (400+ Features)'}")
    print("=" * 80)
    print()

    stream_dataset(
        broker=args.broker,
        topic=args.topic,
        dataset_path=args.dataset,
        rate=args.rate,
        max_count=args.count,
        acks=args.acks,
        compact=is_compact,
    )


if __name__ == "__main__":
    main()

