import json
from kafka import KafkaConsumer

consumer = KafkaConsumer(
    'transactions',
    bootstrap_servers='localhost:9092',
    group_id='failure-test-group',
    auto_offset_reset='earliest',
    enable_auto_commit=False,
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print("Recovery Consumer started...")
print("Topic       : transactions")
print("Consumer ID : failure-test-group")
print("\nWaiting for a transaction...\n")

for message in consumer:

    transaction = message.value

    print(
        f"Received: {transaction['transaction_id']} | "
        f"Partition: {message.partition} | "
        f"Offset: {message.offset}"
    )

    print(f"Amount      : ₹{transaction['amount']}")
    print(f"Merchant    : {transaction['merchant_id']}")
    print(f"Device      : {transaction['device_type']}")
    print(f"Country     : {transaction['country']}")

    print("Processing transaction...")
    print("Transaction processed successfully.")

    consumer.commit()

    print(f"Offset {message.offset} COMMITTED.")
    print("Recovery successful.")

    break

consumer.close()
