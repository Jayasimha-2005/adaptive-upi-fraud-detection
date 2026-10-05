"""
streaming package
End-to-End Streaming to Serving Bridge for Adaptive Financial Fraud Detection.
"""
from __future__ import annotations

from streaming.kafka_broker import (
    ConsumerRecord,
    LocalKafkaBroker,
    LocalKafkaConsumer,
    LocalKafkaProducer,
    RecordMetadata,
    TopicPartition,
)
from streaming.flink_processor import (
    FlinkStatefulVelocityEngine,
    FlinkStreamProcessor,
)
from streaming.stream_serving_bridge import (
    CorrelationBuffer,
    StreamServingBridge,
)
from streaming.stream_pipeline import (
    StreamingPipelineOrchestrator,
    TransactionTrace,
)

__all__ = [
    "LocalKafkaBroker",
    "LocalKafkaProducer",
    "LocalKafkaConsumer",
    "TopicPartition",
    "ConsumerRecord",
    "RecordMetadata",
    "FlinkStatefulVelocityEngine",
    "FlinkStreamProcessor",
    "StreamServingBridge",
    "CorrelationBuffer",
    "StreamingPipelineOrchestrator",
    "TransactionTrace",
]
