from confluent_kafka import Producer
import pandas as pd
import json
import time

# Config
conf = {"bootstrap.servers": "kafka:9092"}
TOPIC = "absa-reviews"
CSV_PATH = "/opt/airflow/projects/absa_streaming/data/test_data.csv" 
DELAY = 1.0 

producer = Producer(conf)

def delivery_report(err, msg):
    if err is not None:
        print(f"Message delivery failed: {err}")

# Load CSV (Assumes text is in the first column)
print(f"Loading data from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows.")

print(f"Starting production to topic '{TOPIC}'...")

for i, row in df.iterrows():
    # Take text from the first column only
    text = str(row.iloc[0])

    # Create simple message without ID
    message = {"review": text}

    producer.produce(
        TOPIC,
        value=json.dumps(message).encode("utf-8"),
        callback=delivery_report
    )
    
    producer.poll(0)
    print(f"[{i+1}/{len(df)}] Sent: {message}")
    time.sleep(DELAY)

producer.flush()
print("All messages sent successfully.")