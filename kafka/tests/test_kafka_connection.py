from kafka import KafkaAdminClient


BROKER = "localhost:9092"


def main():
    print("=" * 70)
    print("KAFKA CONNECTION TEST")
    print("=" * 70)
    print(f"Broker: {BROKER}")
    print()

    admin = None

    try:
        admin = KafkaAdminClient(
            bootstrap_servers=[BROKER],
            client_id="member1_connection_test",
        )

        topics = sorted(admin.list_topics())

        print("Kafka broker connection: SUCCESS")
        print(f"Available topics: {len(topics)}")

        if topics:
            print()
            print("Topics:")
            for topic in topics:
                print(f"  - {topic}")

        print()
        print("RESULT: PASS")

    except Exception as error:
        print()
        print("Kafka broker connection: FAILED")
        print(f"Error: {type(error).__name__}: {error}")
        print()
        print("RESULT: FAIL")

    finally:
        if admin is not None:
            admin.close()

    print("=" * 70)


if __name__ == "__main__":
    main()
