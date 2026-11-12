import json
import time

import pandas as pd
from absa_core import get_logger, get_settings
from confluent_kafka import Producer

log = get_logger("producer")
cfg = get_settings()
CSV_PATH = f"{cfg.data_dir}/test_data.csv"
DELAY = 1.0


def delivery_report(err, msg):
    if err is not None:
        log.error("Message delivery failed: %s", err)


producer = Producer({"bootstrap.servers": cfg.kafka_bootstrap})
df = pd.read_csv(CSV_PATH)
log.info("Loaded %d rows from %s, sending to topic '%s'", len(df), CSV_PATH, cfg.kafka_topic)

for i, row in df.iterrows():
    message = {"review": str(row.iloc[0])}  # text is the first column
    producer.produce(cfg.kafka_topic, value=json.dumps(message).encode("utf-8"), callback=delivery_report)
    producer.poll(0)
    log.info("[%d/%d] Sent: %s", i + 1, len(df), message)
    time.sleep(DELAY)

producer.flush()
log.info("All messages sent")
