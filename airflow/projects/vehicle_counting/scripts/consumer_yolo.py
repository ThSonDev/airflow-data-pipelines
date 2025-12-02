import sys
import os
import json
import base64
import numpy as np
import cv2
import pandas as pd
import torch
import psycopg2
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf, from_json, struct
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# Constants
KAFKA_BROKER = "kafka:9092"
POSTGRES_URL = "jdbc:postgresql://postgres:5432/airflow"
POSTGRES_PROPS = {
    "user": "airflow",
    "password": "airflow",
    "driver": "org.postgresql.Driver"
}
OUTPUT_IMAGE_DIR = "/opt/airflow/static/processed_frames" 
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

# Database Initialization Logic
def initialize_database():
    """
    Creates the necessary tables with the correct schema (Primary Keys, Timezones)
    before Spark starts writing data.
    """
    print("Checking/Creating PostgreSQL tables...")
    try:
        conn = psycopg2.connect(
            host="postgres",
            port=5432,
            dbname="airflow",
            user="airflow",
            password="airflow"
        )
        cur = conn.cursor()

        # Define the 3 tables
        tables = ["yolo_results_video1", "yolo_results_video2", "yolo_results_video3"]

        for table in tables:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    id SERIAL PRIMARY KEY,
                    frame_id INT,
                    video_second INT,
                    total_count INT,
                    class_counts TEXT,
                    image_path TEXT,
                    annotation_json TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            print(f"Table '{table}' ensured.")

        conn.commit()
        cur.close()
        conn.close()
        print("Database initialization complete.")
    except Exception as e:
        print(f"Warning: Database initialization failed. Spark might fail if tables don't exist. Error: {e}")

# YOLO Mapping
# COCO Class IDs: 2=Car, 3=Motorcycle, 5=Bus, 7=Truck
CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Result Schema
result_schema = StructType([
    StructField("total_count", IntegerType()),
    StructField("class_counts", StringType()), 
    StructField("boxes_json", StringType()),
    StructField("annotated_b64", StringType())
])

@pandas_udf(result_schema)
def predict_batch_udf(image_b64_series: pd.Series) -> pd.DataFrame:
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt') 
    
    results_list = []
    
    for b64_str in image_b64_series:
        try:
            # Decode Image
            img_data = base64.b64decode(b64_str)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Run Inference (Only vehicle classes)
            results = model(img, classes=[2, 3, 5, 7], verbose=False)
            result = results[0]
            
            # --- Counting Logic ---
            counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
            boxes_data = []
            
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = CLASS_NAMES.get(cls_id, "unknown")
                if label in counts:
                    counts[label] += 1
                
                boxes_data.append({
                    "label": label,
                    "conf": float(box.conf[0]),
                    "xyxy": box.xyxy[0].tolist()
                })
            
            total_count = sum(counts.values())
            
            # Annotate Image
            annotated_frame = result.plot()
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            annotated_b64 = base64.b64encode(buffer).decode('utf-8')

            results_list.append({
                "total_count": total_count,
                "class_counts": json.dumps(counts),
                "boxes_json": json.dumps(boxes_data),
                "annotated_b64": annotated_b64
            })
        except Exception as e:
            results_list.append({
                "total_count": 0,
                "class_counts": "{}",
                "boxes_json": "[]",
                "annotated_b64": ""
            })
    
    return pd.DataFrame(results_list)

def write_to_postgres(batch_df, batch_id):
    batch_df.persist()
    rows = batch_df.collect()
    
    for row in rows:
        try:
            video_source = row['source_video']
            frame_id = row['frame_id']
            
            # Save Image
            file_path = ""
            if row['inference']['annotated_b64']:
                img_data = base64.b64decode(row['inference']['annotated_b64'])
                filename = f"{video_source}_frame_{frame_id}.jpg"
                file_path = os.path.join(OUTPUT_IMAGE_DIR, filename)
                with open(file_path, "wb") as f:
                    f.write(img_data)
            
            # Prepare Data
            save_data = [(
                row['frame_id'],
                row['video_second'],
                row['inference']['total_count'],
                row['inference']['class_counts'],
                file_path,
                row['inference']['boxes_json']
            )]
            
            save_df = SparkSession.getActiveSession().createDataFrame(
                save_data, 
                ["frame_id", "video_second", "total_count", "class_counts", "image_path", "annotation_json"]
            )
            
            save_df.write \
                .format("jdbc") \
                .option("url", POSTGRES_URL) \
                .option("dbtable", f"yolo_results_{video_source}") \
                .option("user", POSTGRES_PROPS["user"]) \
                .option("password", POSTGRES_PROPS["password"]) \
                .option("driver", "org.postgresql.Driver") \
                .mode("append") \
                .save()
                
        except Exception as e:
            print(f"Error writing batch {batch_id}: {e}")
            
    batch_df.unpersist()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--topics', type=str, required=True)
    args = parser.parse_args()

    initialize_database()

    spark = SparkSession.builder \
        .appName("YOLO_Vehicle_Counter") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.postgresql:postgresql:42.6.0") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # Input Schema
    json_schema = StructType([
        StructField("frame_id", IntegerType()),
        StructField("video_second", IntegerType()),
        StructField("image_b64", StringType()),
        StructField("source_video", StringType())
    ])

    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BROKER) \
        .option("subscribe", args.topics) \
        .option("startingOffsets", "latest") \
        .load()

    parsed_df = df.select(from_json(col("value").cast("string"), json_schema).alias("data")).select("data.*")
    
    processed_df = parsed_df.withColumn("inference", predict_batch_udf(col("image_b64")))

    processed_df.writeStream \
        .foreachBatch(write_to_postgres) \
        .trigger(processingTime='5 seconds') \
        .start() \
        .awaitTermination()

if __name__ == "__main__":
    main()