from pyflink.datastream import StreamExecutionEnvironment
from pyflink.common import Types


def main():
    print("[TEST] Creating Flink environment...")

    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)

    numbers = env.from_collection(
        [1, 2, 3, 4, 5],
        type_info=Types.INT()
    )

    numbers.print()

    print("[TEST] Executing Flink job...")

    env.execute("Java Flink Streaming Test")

    print("[TEST] Flink job completed successfully.")


if __name__ == "__main__":
    main()