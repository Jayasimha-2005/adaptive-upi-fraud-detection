from pyflink.datastream import StreamExecutionEnvironment


def main():
    env = StreamExecutionEnvironment.get_execution_environment()

    # Explicitly tell Flink which Python interpreter
    # must be used by Python UDF workers.
    env.set_python_executable(
        r"C:\Users\Admin\Desktop\adaptive-upi-fraud-detection\flink_venv\Scripts\python.exe"
    )

    env.set_parallelism(1)

    ds = env.from_collection([1, 2, 3, 4, 5])

    # This is a Python DataStream sink, so it tests the
    # actual Python worker configuration.
    ds.print()

    env.execute("Flink Python Worker Test")


if __name__ == "__main__":
    main()