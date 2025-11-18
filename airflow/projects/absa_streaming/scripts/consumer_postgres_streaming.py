# cnn_consumer_postgres.py
# ======================================
# Consumer reading from Kafka "absa-reviews"
# → Reconstructs TextCNN from notebook
# → Preprocesses and Inferences using PySpark UDF
# → Decodes labels (NONE, NEG, POS, NEU)
# → Writes to PostgreSQL with specific logging
# ======================================

import torch
import sys
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.sql.functions import pandas_udf, from_json, col
import pandas as pd
import torch.nn as nn
import numpy as np
import json
import re
import unicodedata
import os
import psycopg2

# === 1. Configuration & Constants ===
SCALA_VERSION = "2.12"
SPARK_VERSION = "3.5.1"

# Architecture Constants (from Notebook)
EMBED_DIM = 128
NUM_FILTERS = 192
WORD_WINDOW = 5
NUM_CLASSES = 4
ASPECT_COLUMNS = ['Price', 'Shipping', 'Outlook', 'Quality', 'Size', 'Shop_Service', 'General', 'Others']

# Paths - Updated to cnn_best.pth
# Assuming /opt/airflow/models/ based on your previous file structure
MODEL_PATH = "/opt/airflow/models/cnn_best.pth"
NEWER_MODELS_DIR = "/opt/airflow/models/newer_models"
VOCAB_PATH = "/opt/airflow/models/vocab.json"

# Label Mapping
PREDICTION_MAP = {
    0: "NONE",
    1: "NEG",
    2: "POS",
    3: "NEU"
}

# Globals for UDF optimization
_model = None
_vocab = None
_device = None

# === 2. Preprocessing Logic (Strictly from Notebook) ===
REMOVE_TONE = False
REMOVE_NOISE = True

def remove_vietnamese_tone(text):
    text = unicodedata.normalize('NFD', text)
    text = re.sub(r'[\u0300-\u036f]', '', text)
    text = unicodedata.normalize('NFC', text)
    return text

def remove_non_vietnamese_chars(text):
    return re.sub(
        r"[^a-zàáạảãăắằặẳẵâầấậẩẫèéẹẻẽêềếệểễ"
        r"ìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữ"
        r"ỳýỵỷỹđ0-9\s]",
        " ",
        text
    )

def clean(text):
    text = str(text).lower().strip()
    if REMOVE_TONE:
        text = remove_vietnamese_tone(text)
    if REMOVE_NOISE:
        text = remove_non_vietnamese_chars(text)
    text = re.sub(r"\s+", " ", text)
    return text

# === 3. Model Architecture (Strictly from Notebook) ===
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_labels):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, NUM_FILTERS, kernel_size=WORD_WINDOW, padding=int(WORD_WINDOW // 2))
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.shared_dense = nn.Linear(NUM_FILTERS, 128)
        self.dropout = nn.Dropout(0.5)
        
        self.output_heads = nn.ModuleList([ 
            nn.Linear(128, NUM_CLASSES) for _ in range(num_labels)
        ])

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = torch.relu(self.conv(x))
        x = self.pool(x).squeeze(2)
        x = torch.relu(self.shared_dense(x)) 
        x = self.dropout(x)

        outputs = []
        for head in self.output_heads:
            outputs.append(head(x))
            
        return torch.stack(outputs, dim=1)

# === 4. Helper Functions ===
def load_resources():
    """Loads model and vocab once per executor."""
    global _model, _vocab, _device
    
    if _vocab is None:
        if os.path.exists(VOCAB_PATH):
            with open(VOCAB_PATH, "r", encoding="utf-8") as f:
                _vocab = json.load(f)
        else:
            print(f"[WARN] Vocab file not found at {VOCAB_PATH}. Using empty vocab.")
            _vocab = {"<PAD>": 0, "<UNK>": 1}

    if _model is None:
        # Re-check device inside executor
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        vocab_size = len(_vocab)
        _model = TextCNN(vocab_size=vocab_size, embed_dim=EMBED_DIM, num_labels=len(ASPECT_COLUMNS))
        
        # --- NEW LOGIC: Check DB for better model ---
        target_model_path = MODEL_PATH # Default
        
        try:
            # Connect to DB to find best model
            conn = psycopg2.connect(host="postgres", port=5432, database="airflow", user="airflow", password="airflow")
            cur = conn.cursor()
            # Get model with highest F1 score
            cur.execute("SELECT model_name FROM retrain_results ORDER BY macro_f1 DESC, id DESC LIMIT 1")
            row = cur.fetchone()
            
            if row:
                best_model_name = row[0]
                candidate_path = os.path.join(NEWER_MODELS_DIR, best_model_name)
                if os.path.exists(candidate_path):
                    target_model_path = candidate_path
                    print(f"[INFO] 🚀 Found better model in DB. Switching to: {target_model_path}")
                else:
                    print(f"[WARN] Best model {best_model_name} listed in DB but file missing. Using default.")
            
            cur.close()
            conn.close()
        except Exception as e:
            print(f"[WARN] Could not check retrain_results (Error: {e}). Using default model.")

        # --- Load the selected model ---
        if os.path.exists(target_model_path):
            _model.load_state_dict(torch.load(target_model_path, map_location=_device))
            print(f"[INFO] Model loaded from {target_model_path} on {_device}")
        else:
            print(f"[WARN] Model file not found at {target_model_path}. Inference will be random/garbage.")
            
        _model.to(_device)
        _model.eval()

def encode_text(text, vocab):
    return [vocab.get(t, 1) for t in text.split()]

# === 5. PySpark UDF for Inference ===
@pandas_udf(T.ArrayType(T.StringType()))
def cnn_inference_udf(texts: pd.Series) -> pd.Series:
    load_resources()
    
    batch_results = []
    processed_indices = []
    
    # Preprocess & Encode
    for text in texts:
        cleaned = clean(text)
        encoded = encode_text(cleaned, _vocab)
        processed_indices.append(torch.tensor(encoded, dtype=torch.long))
    
    if not processed_indices:
        return pd.Series([[] for _ in texts])
        
    from torch.nn.utils.rnn import pad_sequence
    padded_batch = pad_sequence(processed_indices, batch_first=True, padding_value=0).to(_device)
    
    # Inference
    with torch.no_grad():
        outputs = _model(padded_batch) # Shape: [batch, 8, 4]
        predictions = outputs.argmax(dim=2).cpu().numpy() # Shape: [batch, 8]
        
    # Decode to labels
    for row_preds in predictions:
        row_labels = []
        for pred_idx in row_preds:
            label = PREDICTION_MAP.get(pred_idx, "NONE")
            row_labels.append(label)
        batch_results.append(row_labels)
        
    return pd.Series(batch_results)

# === 6. Writer & Logging ===
def write_to_postgres(batch_df, batch_id):
    sys.stdout.reconfigure(encoding='utf-8')
    total_rows = batch_df.count()
    
    print(f"[Batch {batch_id}] Received {total_rows} rows — preview 5:")
    
    # FIX: Changed ASPECTS to ASPECT_COLUMNS here
    if total_rows > 0:
        preview_data = batch_df.select("ReviewText", *ASPECT_COLUMNS).limit(5).toPandas().to_dict(orient="records")
        print(json.dumps(preview_data, ensure_ascii=False, indent=2))
    else:
        print("[]")

    if batch_id == 7:
        print(f"[Batch {batch_id}] 💥 Simulated crash at batch 7.")
        raise Exception(f"Simulated crash at batch {batch_id}")

    try:
        (batch_df
            .select("ReviewText", *ASPECT_COLUMNS)  # FIX: Changed ASPECTS to ASPECT_COLUMNS here
            .write
            .format("jdbc")
            .option("url", "jdbc:postgresql://postgres:5432/airflow")
            .option("dbtable", "absa_results")
            .option("user", "airflow")
            .option("password", "airflow")
            .option("driver", "org.postgresql.Driver")
            .option("charset", "utf8")
            .mode("append")
            .save()
        )
        print(f"[Batch {batch_id}] ✅ Ghi PostgreSQL thành công ({total_rows} dòng).")
        
    except Exception as e:
        print(f"[Batch {batch_id}] ⚠️ Error writing to Postgres: {e}")
        # Optional: print data on error if needed, using ASPECT_COLUMNS
        # preview_error = batch_df.select("ReviewText", *ASPECT_COLUMNS).limit(5).toPandas().to_dict(orient="records")
        # print(json.dumps(preview_error, ensure_ascii=False, indent=2))

# === 7. Main Execution ===
def main():
    spark = (
        SparkSession.builder
        .appName("CNN_Kafka_ABSA_Postgres")
        .config("spark.jars.packages",
                f"org.apache.spark:spark-sql-kafka-0-10_{SCALA_VERSION}:{SPARK_VERSION},"
                "org.postgresql:postgresql:42.6.0,"
                "org.apache.kafka:kafka-clients:3.6.1")
        .config("spark.sql.streaming.checkpointLocation", "/opt/airflow/checkpoints/absa_streaming_checkpoint")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    df_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", "kafka:9092")
        .option("subscribe", "absa-reviews")
        .option("startingOffsets", "latest")
        .option("maxOffsetsPerTrigger", 10)
        .load()
    )
    
    review_schema = T.StructType([
        T.StructField("id", T.StringType()),
        T.StructField("review", T.StringType())
    ])
    
    df_text = df_stream.selectExpr("CAST(value AS STRING) as json_str")
    df_parsed = df_text.withColumn("data", from_json(col("json_str"), review_schema))
    df_input = df_parsed.withColumn("ReviewText", col("data.review")).filter(col("ReviewText").isNotNull())

    # Apply Inference
    df_preds = df_input.withColumn("predictions_array", cnn_inference_udf(col("ReviewText")))
    
    df_final = df_preds
    for i, aspect in enumerate(ASPECT_COLUMNS):
        df_final = df_final.withColumn(aspect, col("predictions_array")[i])

    query = (
        df_final.writeStream
        .foreachBatch(write_to_postgres)
        .outputMode("append")
        .trigger(processingTime="5 seconds")
        .start()
    )

    print("🚀 CNN Streaming job started — listening to Kafka...")
    query.awaitTermination()

if __name__ == "__main__":
    print("="*40)
    print(f"Checking Compute Resources...")
    print(f"GPU Available: {torch.cuda.is_available()}")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    print("="*40)
    main()