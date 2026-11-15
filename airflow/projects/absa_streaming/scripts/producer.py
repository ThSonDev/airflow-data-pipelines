import json
import sys
import time

import pandas as pd
from absa_core import get_logger, get_settings
from confluent_kafka import Producer

log = get_logger("producer")
cfg = get_settings()
CSV_PATH = f"{cfg.data_dir}/test_data.csv"
DELAY = 1.0
FLUSH_SECONDS = 30
failures = 0


def delivery_report(err, msg):
    global failures
    if err is not None:
        failures += 1
        log.error("Message delivery failed: %s", err)


def main():
    producer = Producer({"bootstrap.servers": cfg.kafka_bootstrap})
    df = pd.read_csv(CSV_PATH)
    log.info("Loaded %d rows from %s, sending to topic '%s'", len(df), CSV_PATH, cfg.kafka_topic)

    deadline = time.monotonic() + cfg.run_seconds
    sent = 0
    for i, row in df.iterrows():
        if time.monotonic() >= deadline:
            log.info("Run time of %ds reached, stopping after %d of %d rows", cfg.run_seconds, sent, len(df))
            break
        message = {"review": str(row.iloc[0])}  # text is the first column
        producer.produce(cfg.kafka_topic, value=json.dumps(message).encode("utf-8"), callback=delivery_report)
        producer.poll(0)
        sent += 1
        log.info("[%d/%d] Sent: %s", i + 1, len(df), message)
        time.sleep(DELAY)

    remaining = producer.flush(FLUSH_SECONDS)
    if remaining > 0 or failures > 0:
        log.error("Not all messages were delivered: %d unflushed, %d failed", remaining, failures)
        sys.exit(1)
    log.info("All %d messages sent", sent)


if __name__ == "__main__":
    main()
