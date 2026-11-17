"""foreachBatch writer: logs a preview and appends rows to PostgreSQL."""
import json
import sys
import time
from pathlib import Path

from absa_core import ASPECTS, get_logger, get_settings

log = get_logger("sink")
cfg = get_settings()
JDBC_URL = f"jdbc:postgresql://{cfg.postgres_host}:{cfg.postgres_port}/{cfg.postgres_db}"


def write_to_postgres(batch_df, batch_id):
    sys.stdout.reconfigure(encoding='utf-8')
    start = time.monotonic()
    # Without persist the Kafka read and the UDF would run once per action (count, preview, write).
    batch_df = batch_df.persist()
    try:
        total_rows = batch_df.count()
        log.info("Batch %s: received %d rows", batch_id, total_rows)

        if total_rows > 0:
            preview = batch_df.select("ReviewText", *ASPECTS).limit(5).toPandas().to_dict(orient="records")
            log.info("Batch %s preview: %s", batch_id, json.dumps(preview, ensure_ascii=False))

        # Crash once only: after a resume Spark replays this batch, and it must not crash forever.
        marker = Path(cfg.checkpoint_dir) / "crash_done"
        if batch_id == cfg.crash_batch and not marker.exists():
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
            log.error("Batch %s: simulated crash", batch_id)
            raise RuntimeError(f"Simulated crash at batch {batch_id}")

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
        except Exception:
            # Re-raise so Spark does not mark the batch done; the next start replays it.
            log.exception("Batch %s: error writing to PostgreSQL", batch_id)
            raise
        log.info("Batch %s: wrote %d rows to PostgreSQL in %.2f s", batch_id, total_rows, time.monotonic() - start)
    finally:
        batch_df.unpersist()
