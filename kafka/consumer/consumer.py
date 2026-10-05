import argparse
import json
import sys
import time

# Ensure stdout handles Windows console encoding safely
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from kafka import KafkaConsumer

DEFAULT_BROKER = "localhost:9092"
DEFAULT_TOPIC = "transactions"
DEFAULT_GROUP = "fraud-detector-group"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Dataset-Compatible Real-Time Kafka Consumer")
    parser.add_argument("--broker", default=DEFAULT_BROKER, help=f"Kafka broker (default: {DEFAULT_BROKER})")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help=f"Kafka topic (default: {DEFAULT_TOPIC})")
    parser.add_argument("--group", default=DEFAULT_GROUP, help=f"Consumer group (default: {DEFAULT_GROUP})")
    parser.add_argument("--max-messages", type=int, default=0, help="Stop after N messages (0 for continuous)")
    return parser.parse_args()


def extract_field(tx, keys, default=None):
    for k in keys:
        if k in tx and tx[k] is not None:
            return tx[k]
    return default


def main():
    args = parse_arguments()

    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=[args.broker],
        group_id=args.group,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        key_deserializer=lambda k: k.decode("utf-8") if k else None,
    )

    processed_count = 0
    unique_transactions = set()
    duplicate_count = 0
    fraud_detected = 0

    print("=" * 80)
    print("KAFKA DATASET CONSUMER")
    print("=" * 80)
    print(f"Broker       : {args.broker}")
    print(f"Topic        : {args.topic}")
    print(f"Consumer ID  : {args.group}")
    print(f"Max Messages : {'Unlimited' if args.max_messages == 0 else args.max_messages}")
    print("=" * 80)
    print("\nWaiting for real-time transactions...\n")

    try:
        for message in consumer:
            tx = message.value

            tx_id = str(extract_field(tx, ["transaction_id", "TransactionID", "id"], f"OFFSET-{message.offset}"))
            card_id = str(extract_field(tx, ["card_id", "card1", "account_id"], message.key or "UNKNOWN"))
            amount = extract_field(tx, ["amount", "TransactionAmt", "amt"], 0.0)
            is_fraud = extract_field(tx, ["is_fraud", "isFraud"], False)
            source = extract_field(tx, ["dataset_source", "source"], "dataset")

            processed_count += 1

            if tx_id in unique_transactions:
                duplicate_count += 1
                print(f"[!] DUPLICATE DETECTED: {tx_id} | Partition: {message.partition} | Offset: {message.offset}")
            else:
                unique_transactions.add(tx_id)
                if is_fraud:
                    fraud_detected += 1
                
                status_tag = "[FRAUD]" if is_fraud else "[OK]"
                print(
                    f"[{processed_count}] {status_tag:<7} | "
                    f"ID: {tx_id} | "
                    f"Card: {card_id} | "
                    f"Amt: ${float(amount):.2f} | "
                    f"Src: {source} | "
                    f"Part: {message.partition} | "
                    f"Off: {message.offset}"
                )

            # Commit offset after successful delivery
            consumer.commit()

            if args.max_messages > 0 and processed_count >= args.max_messages:
                break

    except KeyboardInterrupt:
        print("\nConsumer stopped by user.")
    finally:
        consumer.close()
        print("\n" + "=" * 80)
        print("CONSUMER SUMMARY")
        print("=" * 80)
        print(f"Total Processed     : {processed_count}")
        print(f"Unique Transactions : {len(unique_transactions)}")
        print(f"Duplicate Events    : {duplicate_count}")
        print(f"Fraudulent Events   : {fraud_detected}")
        print("=" * 80)


if __name__ == "__main__":
    main()
