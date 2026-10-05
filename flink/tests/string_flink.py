from pyflink.datastream import StreamExecutionEnvironment
from pyflink.common import Types

env = StreamExecutionEnvironment.get_execution_environment()

ds = env.from_collection(
    ["transaction_1", "transaction_2", "transaction_3"],
    type_info=Types.STRING()
)

ds.print()

env.execute("Flink String Test")
