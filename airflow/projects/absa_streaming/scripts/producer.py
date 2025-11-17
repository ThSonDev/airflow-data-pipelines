# SE363 – Phát triển ứng dụng trên nền tảng dữ liệu lớn
# Khoa Công nghệ Phần mềm – Trường Đại học Công nghệ Thông tin, ĐHQG-HCM
# HopDT – Faculty of Software Engineering, University of Information Technology (FSE-UIT)

# producer.py
# ======================================
# Producer đọc dữ liệu từ file "test_data.csv"
# và gửi từng dòng dữ liệu lên Kafka topic "absa-reviews".

from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError
import json, time, pandas as pd

# --- Cấu hình ---
KAFKA_SERVER = "kafka:9092"  # dùng hostname trong docker network
TOPIC = "absa-reviews"
CSV_PATH = "/opt/airflow/projects/absa_streaming/data/test_data.csv"
DELAY = 1.0  # giây giữa mỗi message

# --- Đảm bảo topic tồn tại ---
try:
    admin = KafkaAdminClient(bootstrap_servers=KAFKA_SERVER, client_id="absa_admin")
    topic = NewTopic(name=TOPIC, num_partitions=1, replication_factor=1)
    admin.create_topics([topic])
    print(f"✅ Created topic: {TOPIC}")
except TopicAlreadyExistsError:
    print(f"ℹ️ Topic {TOPIC} already exists")
except Exception as e:
    print(f"⚠️ Skipped topic creation: {e}")
finally:
    try:
        admin.close()
    except:
        pass

# --- Khởi tạo Producer ---
producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

# --- Đọc dữ liệu ---
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows from {CSV_PATH}")

# --- Gửi từng dòng ---
for i, row in df.iterrows():
    text = row["text"] if "text" in row else row.iloc[0]
    msg = {"id": int(i), "review": text}
    producer.send(TOPIC, msg)
    print(f"[{i+1}/{len(df)}] Sent → {msg}")
    time.sleep(DELAY)

producer.flush()
print("✅ All messages sent successfully.")
