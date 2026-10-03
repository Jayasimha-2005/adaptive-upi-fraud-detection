# Flink Architecture

IEEE-CIS -> batch preparation -> cleaning -> transaction/identity join
-> feature engineering -> Parquet/CSV

Kafka -> Flink Source -> JSON parsing -> event time -> watermark
-> keyBy(user_id) -> 10-minute event-time window / 5-minute slide
-> real-time features -> fraud model/output

Core concepts: event time, watermarks, keyed state, windows,
checkpointing, Kafka integration, batch/stream processing.
