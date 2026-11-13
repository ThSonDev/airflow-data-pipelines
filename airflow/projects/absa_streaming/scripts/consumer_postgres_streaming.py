import torch
from absa_core import ASPECTS, get_logger, get_settings
from inference import cnn_inference_udf
from pyspark.sql import SparkSession
from pyspark.sql import types as T
from pyspark.sql.functions import col, from_json
from sink import write_to_postgres

log = get_logger("consumer")
cfg = get_settings()
SCALA_VERSION = "2.12"
SPARK_VERSION = "3.5.1"


def main():
    spark = (
        SparkSession.builder
        .appName("CNN_Kafka_ABSA_Postgres")
        .config("spark.jars.packages",
                f"org.apache.spark:spark-sql-kafka-0-10_{SCALA_VERSION}:{SPARK_VERSION},"
                "org.postgresql:postgresql:42.6.0")
        .config("spark.sql.streaming.checkpointLocation", "/opt/airflow/checkpoints/absa_streaming_checkpoint")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    df_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", cfg.kafka_bootstrap)
        .option("subscribe", cfg.kafka_topic)
        .option("startingOffsets", "latest")
        .option("maxOffsetsPerTrigger", 10)
        .load()
    )

    review_schema = T.StructType([
        T.StructField("id", T.StringType()),
        T.StructField("review", T.StringType())
    ])
    df_parsed = (df_stream.selectExpr("CAST(value AS STRING) as json_str")
                 .withColumn("data", from_json(col("json_str"), review_schema)))
    df_input = df_parsed.withColumn("ReviewText", col("data.review")).filter(col("ReviewText").isNotNull())

    df_final = df_input.withColumn("predictions_array", cnn_inference_udf(col("ReviewText")))
    for i, aspect in enumerate(ASPECTS):
        df_final = df_final.withColumn(aspect, col("predictions_array")[i])

    query = (
        df_final.writeStream
        .foreachBatch(write_to_postgres)
        .outputMode("append")
        .trigger(processingTime="5 seconds")
        .start()
    )
    log.info("Streaming job started, listening to Kafka")
    query.awaitTermination()


if __name__ == "__main__":
    log.info("GPU available: %s", torch.cuda.is_available())
    main()
