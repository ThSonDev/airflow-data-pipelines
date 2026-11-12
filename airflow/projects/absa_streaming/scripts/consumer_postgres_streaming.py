import torch
import sys
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.sql.functions import pandas_udf, from_json, col
import pandas as pd
import json
import os
import psycopg2

from absa_core import (
    ASPECTS as ASPECT_COLUMNS, EMBED_DIM, PREDICTION_MAP, TextCNN, clean, encode, get_settings,
)

# Configuration & Constants
SCALA_VERSION = "2.12"
SPARK_VERSION = "3.5.1"

# Paths - Updated to cnn_best.pth
cfg = get_settings()
MODEL_PATH = f"{cfg.models_dir}/cnn_best.pth"
NEWER_MODELS_DIR = f"{cfg.models_dir}/newer_models"
VOCAB_PATH = f"{cfg.models_dir}/vocab.json"

# Globals for UDF optimization
_model = None
_vocab = None
_device = None

# Helper Functions
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
        
        # Check DB for better model
        target_model_path = MODEL_PATH # Default
        
        try:
            # Connect to DB to find best model
            conn = psycopg2.connect(
                host=cfg.postgres_host, port=cfg.postgres_port, database=cfg.postgres_db,
                user=cfg.postgres_user, password=cfg.postgres_password,
            )
            cur = conn.cursor()
            # Get model with highest F1 score
            cur.execute("SELECT model_name FROM retrain_results ORDER BY macro_f1 DESC, id DESC LIMIT 1")
            row = cur.fetchone()
            
            if row:
                best_model_name = row[0]
                candidate_path = os.path.join(NEWER_MODELS_DIR, best_model_name)
                if os.path.exists(candidate_path):
                    target_model_path = candidate_path
                    print(f"[INFO] Found better model in DB. Switching to: {target_model_path}")
                else:
                    print(f"[WARN] Best model {best_model_name} listed in DB but file missing. Using default.")
            
            cur.close()
            conn.close()
        except Exception as e:
            print(f"[WARN] Could not check retrain_results (Error: {e}). Using default model.")

        # Load the selected model
        if os.path.exists(target_model_path):
            _model.load_state_dict(torch.load(target_model_path, map_location=_device))
            print(f"[INFO] Model loaded from {target_model_path} on {_device}")
        else:
            print(f"[WARN] Model file not found at {target_model_path}. Inference will be random/garbage.")
            
        _model.to(_device)
        _model.eval()

# === 5. PySpark UDF for Inference ===
@pandas_udf(T.ArrayType(T.StringType()))
def cnn_inference_udf(texts: pd.Series) -> pd.Series:
    load_resources()
    
    batch_results = []
    processed_indices = []
    
    # Preprocess & Encode
    for text in texts:
        cleaned = clean(text)
        encoded = encode(cleaned, _vocab)
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

# Writer & Logging
def write_to_postgres(batch_df, batch_id):
    sys.stdout.reconfigure(encoding='utf-8')
    total_rows = batch_df.count()
    
    print(f"[Batch {batch_id}] Received {total_rows} rows — preview 5:")
    
    if total_rows > 0:
        preview_data = batch_df.select("ReviewText", *ASPECT_COLUMNS).limit(5).toPandas().to_dict(orient="records")
        print(json.dumps(preview_data, ensure_ascii=False, indent=2))
    else:
        print("[]")

    if batch_id == 7:
        print(f"[Batch {batch_id}] Simulated crash at batch 7.")
        raise Exception(f"Simulated crash at batch {batch_id}")

    try:
        (batch_df
            .select("ReviewText", *ASPECT_COLUMNS)
            .write
            .format("jdbc")
            .option("url", f"jdbc:postgresql://{cfg.postgres_host}:{cfg.postgres_port}/{cfg.postgres_db}")
            .option("dbtable", "absa_results")
            .option("user", cfg.postgres_user)
            .option("password", cfg.postgres_password)
            .option("driver", "org.postgresql.Driver")
            .option("charset", "utf8")
            .mode("append")
            .save()
        )
        print(f"[Batch {batch_id}] Write into PostgreSQL sucessfully ({total_rows} rows).")
        
    except Exception as e:
        print(f"[Batch {batch_id}] !!! Error writing to Postgres: {e}")

# Main Execution
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

    print("CNN Streaming job started — listening to Kafka...")
    query.awaitTermination()

if __name__ == "__main__":
    print("="*40)
    print(f"Checking Compute Resources...")
    print(f"GPU Available: {torch.cuda.is_available()}")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    print("="*40)
    main()
