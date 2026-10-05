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
DEFAULT_GROUP = "fraud-detection-engine-group"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Real-Time Dataset-Driven Fraud Detection Consumer Engine")
    parser.add_argument("--broker", default=DEFAULT_BROKER, help=f"Kafka broker (default: {DEFAULT_BROKER})")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help=f"Kafka topic (default: {DEFAULT_TOPIC})")
    parser.add_argument("--group", default=DEFAULT_GROUP, help=f"Consumer group (default: {DEFAULT_GROUP})")
    parser.add_argument("--threshold", type=float, default=5000.0, help="High-risk amount threshold (default: 5000.0)")
    parser.add_argument("--max-messages", type=int, default=0, help="Stop and print summary after N messages (0 for continuous)")
    return parser.parse_args()


def detect_fraud_rules(tx, threshold=5000.0):
    reasons = []
    
    amount = float(tx.get("amount", tx.get("TransactionAmt", 0.0)))
    country = str(tx.get("country", tx.get("addr2", "US")))
    dataset_is_fraud = bool(tx.get("is_fraud", tx.get("isFraud", False)))
    
    if dataset_is_fraud:
        reasons.append("Flagged fraudulent in benchmark ground truth dataset")

    if amount >= threshold:
        reasons.append(f"High transaction amount exceeds safety threshold (${amount:.2f} >= ${threshold:.2f})")

    if tx.get("location_mismatch") or country == "FOREIGN":
        reasons.append("Geographic/location mismatch detected")

    if int(tx.get("velocity_24h", 0)) > 5:
        reasons.append("High 24h card velocity / rapid burst activity")

    return reasons, (len(reasons) > 0 or dataset_is_fraud)


def main():
    args = parse_arguments()

    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=[args.broker],
        group_id=args.group,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
        key_deserializer=lambda k: k.decode("utf-8") if k else None,
    )

    print("=" * 80)
    print("  REAL-TIME FINANCIAL FRAUD DETECTION CONSUMER (DATASET-DRIVEN)")
    print("=" * 80)
    print(f"Broker            : {args.broker}")
    print(f"Topic             : {args.topic}")
    print(f"Consumer Group    : {args.group}")
    print(f"Amount Threshold  : ${args.threshold:.2f}")
    print(f"Max Messages      : {'Continuous (Press Ctrl+C to stop & see summary)' if args.max_messages == 0 else args.max_messages}")
    print("=" * 80)
    print("\nListening for streaming financial transactions...\n")

    processed = 0
    fraud_count = 0
    total_amount = 0.0

    try:
        for message in consumer:
            tx = message.value
            processed += 1

            tx_id = tx.get("transaction_id", tx.get("TransactionID", f"TX-{processed}"))
            card_id = tx.get("card_id", tx.get("card1", message.key or "UNKNOWN"))
            amount = float(tx.get("amount", tx.get("TransactionAmt", 0.0)))
            merchant = tx.get("merchant_id", tx.get("ProductCD", "M-UNKNOWN"))
            device = tx.get("device_type", "mobile")
            country = tx.get("country", "US")
            source = tx.get("dataset_source", "dataset")

            total_amount += amount
            reasons, is_suspicious = detect_fraud_rules(tx, threshold=args.threshold)

            print("-" * 80)
            print(f"Transaction ID : {tx_id} | Source: {source} | Part: {message.partition} | Offset: {message.offset}")
            print(f"Card / User    : {card_id} | Amount: ${amount:.2f} | Merchant: {merchant}")
            print(f"Device / Geo   : {device} | Country: {country}")

            if is_suspicious:
                fraud_count += 1
                print("\n[!] [ALERT] SUSPICIOUS / FRAUDULENT TRANSACTION DETECTED!")
                for r in reasons:
                    print(f"   * {r}")
                print("Status         : BLOCKED / SENT TO FRAUD AUDIT")
            else:
                print("\n[+] Transaction Verified (Normal Behavior)")
                print("Status         : APPROVED")

            consumer.commit()
            print(f"Kafka Offset {message.offset} securely committed.")

            # Print rolling summary every 10 transactions
            if processed % 10 == 0:
                print(f"\n>>> [LIVE STATS] Processed: {processed} | Fraud Detected: {fraud_count} ({(fraud_count/processed)*100:.1f}%) | Total Vol: ${total_amount:,.2f} <<<\n")

            if args.max_messages > 0 and processed >= args.max_messages:
                break

    except KeyboardInterrupt:
        print("\nFraud detection engine stopped by user.")
    finally:
        consumer.close()
        print("\n" + "=" * 80)
        print("FINAL SESSION SUMMARY")
        print("=" * 80)
        print(f"Total Transactions Processed : {processed}")
        print(f"Fraudulent / Flagged Events  : {fraud_count} ({(fraud_count/max(1, processed))*100:.2f}%)")
        print(f"Total Volume Monitored       : ${total_amount:,.2f}")
        print("=" * 80)


if __name__ == "__main__":
    main()
