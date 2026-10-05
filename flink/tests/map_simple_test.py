import pytest

pytest.importorskip("pyflink")

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.common import Types


def double(x):
    return x * 2


def main():
    print("[TEST] Starting simple map test...")

    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)

    numbers = env.from_collection(
        [1, 2, 3, 4, 5],
        type_info=Types.INT()
    )

    doubled = numbers.map(
        double,
        output_type=Types.INT()
    )

    doubled.print()

    print("[TEST] Executing...")

    env.execute("Flink Simple Map Test")

    print("[TEST] Completed.")


if __name__ == "__main__":
    main()