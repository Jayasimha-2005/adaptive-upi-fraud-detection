import argparse
import json
import sys
import time
from collections import defaultdict

# Safe Windows stdout encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from kafka import KafkaConsumer

DEFAULT_BROKER = "localhost:9092"
DEFAULT_TOPIC = "ieee_cis_transactions"
DEFAULT_GROUP = "ieee_cis_fraud_group"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="IEEE-CIS Stream Consumer & Fraud Detector for Apache Kafka"
    )
    parser.add_argument(
        "--broker",
        default=DEFAULT_BROKER,
        help=f"Kafka bootstrap broker (default: {DEFAULT_BROKER})",
    )
    parser.add_argument(
        "--topic",
        default=DEFAULT_TOPIC,
        help=f"Kafka topic to consume (default: {DEFAULT_TOPIC})",
    )
    parser.add_argument(
        "--group",
        default=DEFAULT_GROUP,
        help=f"Consumer group ID (default: {DEFAULT_GROUP})",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=0,
        help="Maximum messages to consume before exiting (0 for full dataset/continuous)",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("=" * 80)
    print("  IEEE-CIS REAL-TIME FULL-DATASET STREAM CONSUMER & MONITOR")
    print("=" * 80)
    print(f"Broker         : {args.broker}")
    print(f"Topic          : {args.topic}")
    print(f"Consumer Group : {args.group}")
    print(f"Max Messages   : {'Full Dataset / Continuous' if args.max_messages == 0 else args.max_messages}")
    print("=" * 80)
    print("\n[+] Waiting for IEEE-CIS stream events from Kafka (Press Ctrl+C to stop & see summary)...\n")

    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=[args.broker],
        group_id=args.group,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )

    count = 0
    fraud_count = 0
    total_amount = 0.0
    partition_counts = defaultdict(int)
    start_time = None

    try:
        for message in consumer:
            if start_time is None:
                start_time = time.time()

            count += 1
            record = message.value
            partition = message.partition
            partition_counts[partition] += 1

            tx_id = record.get("TransactionID") or record.get("transaction_id", f"TX-{count}")
            amt = float(record.get("TransactionAmt", record.get("amount", 0.0)))
            is_fraud = int(record.get("isFraud", record.get("is_fraud", 0)))
            card1 = record.get("card1") or record.get("card_id", "N/A")
            total_amount += amt

            if is_fraud == 1:
                fraud_count += 1

            # Print every 500 messages or on fraud detection
            if count % 500 == 0 or is_fraud == 1:
                flag = "[FRAUD ALERT]" if is_fraud == 1 else "[NORMAL]"
                print(
                    f"[{count:06d}] {flag:<14} TX: {tx_id} | Card: {card1} | "
                    f"Amt: ${amt:.2f} | Part: {partition} | Off: {message.offset}"
                )

            # Commit periodically every 100 messages for high throughput
            if count % 100 == 0:
                consumer.commit()

            if args.max_messages > 0 and count >= args.max_messages:
                print(f"\n[+] Reached target count of {args.max_messages} messages.")
                break

    except KeyboardInterrupt:
        print("\n[!] Consumer stopped by user.")
    finally:
        consumer.commit()
        consumer.close()

    elapsed = time.time() - (start_time or time.time())
    rate = count / elapsed if elapsed > 0 else 0

    print()
    print("=" * 80)
    print("  IEEE-CIS CONSUMPTION BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Total Transactions Processed : {count:,}")
    print(f"Total Fraud Detected         : {fraud_count:,} ({(fraud_count / max(1, count)) * 100:.2f}%)")
    print(f"Total Transaction Volume     : ${total_amount:,.2f}")
    print(f"Total Ingestion Time         : {elapsed:.3f} s")
    print(f"Consumption Throughput       : {rate:.2f} tx/sec")
    print("Partition Load Distribution  :")
    for p, c in sorted(partition_counts.items()):
        print(f"  - Partition {p}: {c:,} messages ({(c / max(1, count)) * 100:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
