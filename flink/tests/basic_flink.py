from pyflink.datastream import StreamExecutionEnvironment
from pyflink.common import Types

env = StreamExecutionEnvironment.get_execution_environment()

ds = env.from_collection(
    [1, 2, 3],
    type_info=Types.INT()
)

ds.print()

env.execute("Basic Flink Test")
