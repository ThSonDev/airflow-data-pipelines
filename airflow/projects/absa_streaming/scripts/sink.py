"""foreachBatch writer: logs a preview and appends rows to PostgreSQL."""
import json
import sys

from absa_core import ASPECTS, get_logger, get_settings

log = get_logger("sink")
cfg = get_settings()
JDBC_URL = f"jdbc:postgresql://{cfg.postgres_host}:{cfg.postgres_port}/{cfg.postgres_db}"


def write_to_postgres(batch_df, batch_id):
    sys.stdout.reconfigure(encoding='utf-8')
    total_rows = batch_df.count()
    log.info("Batch %s: received %d rows", batch_id, total_rows)

    if total_rows > 0:
        preview = batch_df.select("ReviewText", *ASPECTS).limit(5).toPandas().to_dict(orient="records")
        log.info("Batch %s preview: %s", batch_id, json.dumps(preview, ensure_ascii=False))

    if batch_id == 7:
        log.error("Batch %s: simulated crash", batch_id)
        raise Exception(f"Simulated crash at batch {batch_id}")

    try:
        (batch_df
            .select("ReviewText", *ASPECTS)
            .write
            .format("jdbc")
            .option("url", JDBC_URL)
            .option("dbtable", "absa_results")
            .option("user", cfg.postgres_user)
            .option("password", cfg.postgres_password)
            .option("driver", "org.postgresql.Driver")
            .option("charset", "utf8")
            .mode("append")
            .save()
        )
        log.info("Batch %s: wrote %d rows to PostgreSQL", batch_id, total_rows)
    except Exception as e:
        log.error("Batch %s: error writing to PostgreSQL: %s", batch_id, e)
