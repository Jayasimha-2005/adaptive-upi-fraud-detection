import json
import time
from kafka import KafkaConsumer


BROKER = "localhost:9092"
TOPIC = "acks_test"
GROUP_ID = "acks-test-group"


consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=[BROKER],
    group_id=GROUP_ID,
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    value_deserializer=lambda x: json.loads(x.decode("utf-8"))
)

print("=" * 60)
print("ACKS EXPERIMENT CONSUMER")
print("=" * 60)
print(f"Broker : {BROKER}")
print(f"Topic  : {TOPIC}")
print(f"Group  : {GROUP_ID}")
print("=" * 60)

count = 0
start_time = time.time()

try:
    for message in consumer:
        count += 1

        if count <= 5:
            print(
                f"Received #{count}: "
                f"ID={message.value['transaction_id']} "
                f"partition={message.partition} "
                f"offset={message.offset}"
            )

        if count >= 2000:
            break

except KeyboardInterrupt:
    print("\nConsumer stopped.")

finally:
    elapsed = time.time() - start_time

    if elapsed > 0:
        consumer_rate = count / elapsed
    else:
        consumer_rate = 0

    print("=" * 60)
    print("CONSUMER RESULT")
    print("=" * 60)
    print(f"Messages received : {count}")
    print(f"Total time        : {elapsed:.3f} sec")
    print(f"Consumer rate     : {consumer_rate:.2f} tx/sec")
    print("=" * 60)

    consumer.close()
