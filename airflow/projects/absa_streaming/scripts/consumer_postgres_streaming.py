# SE363 – Phát triển ứng dụng trên nền tảng dữ liệu lớn
# Khoa Công nghệ Phần mềm – Trường Đại học Công nghệ Thông tin, ĐHQG-HCM
# HopDT – Faculty of Software Engineering, University of Information Technology (FSE-UIT)

# consumer_postgres_streaming.py
# ======================================
# Consumer đọc dữ liệu từ Kafka topic "absa-reviews"
# → chạy inference mô hình ABSA (.pt)
# → ghi kết quả vào PostgreSQL
# → Airflow sẽ giám sát và khởi động lại khi job bị dừng.

from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.sql.functions import pandas_udf, from_json, col
import pandas as pd, torch, torch.nn as nn, torch.nn.functional as tF
from transformers import AutoTokenizer, AutoModel
import random, time, os, sys, json

# === 1. Spark session với Kafka connector ===
scala_version = "2.12"
spark_version = "3.5.1"

spark = (
    SparkSession.builder
    .appName("Kafka_ABSA_Postgres")
    .config("spark.jars.packages",
            f"org.apache.spark:spark-sql-kafka-0-10_{scala_version}:{spark_version},"
            "org.postgresql:postgresql:42.6.0,"
            "org.apache.kafka:kafka-clients:3.6.1")
    .config("spark.sql.streaming.checkpointLocation", "/opt/airflow/checkpoints/absa_streaming_checkpoint")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

# === 2. Đọc dữ liệu streaming từ Kafka ===
df_stream = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "kafka:9092")
    .option("subscribe", "absa-reviews")
    .option("startingOffsets", "latest")
    .option("maxOffsetsPerTrigger", 10)
    .load()
)

df_text = df_stream.selectExpr("CAST(value AS STRING) as Review")

# === 3. Định nghĩa mô hình ABSA ===
ASPECTS = ["Price","Shipping","Outlook","Quality","Size","Shop_Service","General","Others"]
MODEL_NAME = "xlm-roberta-base"
MODEL_PATH = "/opt/airflow/models/best_absa_hardshare.pt"
MAX_LEN = 48
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model, _tokenizer = None, None

class ABSAModel(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_aspects=len(ASPECTS)):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        H = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.head_m = nn.Linear(H, num_aspects)
        self.head_s = nn.Linear(H, num_aspects * 3)
    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h_cls = self.dropout(out.last_hidden_state[:, 0, :])
        return self.head_m(h_cls), self.head_s(h_cls).view(-1, len(ASPECTS), 3)

@pandas_udf(T.ArrayType(T.FloatType()))
def absa_infer_udf(texts: pd.Series) -> pd.Series:
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        _model = ABSAModel()


        NEWER_ROOT = "/opt/airflow/models/newer_models"
        # if os.path.exists(NEWER_ROOT):
        #     valid_models = []
        #     for d in os.listdir(NEWER_ROOT):
        #         sub = os.path.join(NEWER_ROOT, d)
        #         model_file = os.path.join(sub, "model.pt")
        #         if d.startswith("best_absa_") and os.path.isdir(sub) and os.path.isfile(model_file) and os.path.getsize(model_file) > 1024:
        #             valid_models.append(d)
        #     if valid_models:
        #         latest = max(valid_models, key=lambda x: int(x.split("_")[-1]))
        #         candidate = os.path.join(NEWER_ROOT, latest, "model.pt")
        #         print(f"[INFO] ✅ Dùng model mới nhất: {candidate}")
        #         _model.load_state_dict(torch.load(candidate, map_location=DEVICE))
        #     else:
        #         print("[INFO] ⚠️ newer_models trống — dùng model mặc định.")
        #         _model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        # else:
        #     print("[INFO] ⚠️ Không có thư mục newer_models — dùng model mặc định.")
        #     _model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

        from sqlalchemy import create_engine, text
        latest_model_name = None
        try:
            engine = create_engine("postgresql+psycopg2://airflow:airflow@postgres:5432/airflow")
            with engine.connect() as conn:
                q = text("""
                    SELECT model_name FROM retrain_results
                    ORDER BY f1_macro DESC NULLS LAST, created_at DESC
                    LIMIT 1
                """)
                result = conn.execute(q).fetchone()
                if result and result[0]:
                    latest_model_name = result[0]
                    print(f"[INFO] Model tốt nhất trong PostgreSQL: {latest_model_name}")
                else:
                    print("[INFO] PostgreSQL rỗng — chưa có model retrain nào, dùng model mặc định.")
        except Exception as e:
            print(f"[WARN] Không thể đọc PostgreSQL: {e}")
            latest_model_name = None

        # -----------------------------
        # 🧠 Chọn model để load
        # -----------------------------
        candidate = None
        if latest_model_name:
            candidate_dir = os.path.join(NEWER_ROOT, latest_model_name)
            candidate_file = os.path.join(candidate_dir, "model.pt")
            if os.path.isfile(candidate_file) and os.path.getsize(candidate_file) > 1024:
                candidate = candidate_file
                print(f"[INFO] ✅ Dùng model mới nhất từ PostgreSQL: {candidate}")
            else:
                print(f"[WARN] Model {latest_model_name} trong PostgreSQL không tồn tại hoặc rỗng — dùng model mặc định.")
        
        if candidate:
            _model.load_state_dict(torch.load(candidate, map_location=DEVICE))
        else:
            print(f"[INFO] 🔁 Dùng model mặc định: {MODEL_PATH}")
            _model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        
        _model.half()
        _model.to(DEVICE).eval()

    results = []
    for t in texts:
        enc = _tokenizer(t, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits_m, logits_s = _model(enc["input_ids"], enc["attention_mask"])
            p_m = torch.sigmoid(logits_m)[0].cpu().numpy().tolist()
            p_s = tF.softmax(logits_s, dim=-1)[0].cpu().numpy().flatten().tolist()
        results.append(p_m + p_s)
    torch.cuda.empty_cache()
    return pd.Series(results)

df_pred = df_text.withColumn("predictions", absa_infer_udf(F.col("Review")))

# === 4. Giải mã kết quả ra nhãn POS/NEG/NEU ===
@pandas_udf("string")
def decode_sentiment(preds: pd.Series) -> pd.Series:
    SENTIMENTS = ["NEG", "POS", "NEU"]
    res = []
    for p in preds:
        if not p:
            res.append("?")
            continue
        p = list(p)
        p_m, p_s = p[:len(ASPECTS)], p[len(ASPECTS):]
        decoded = []
        for i, asp in enumerate(ASPECTS):
            triplet = p_s[i*3:(i+1)*3]
            s = SENTIMENTS[int(max(range(3), key=lambda j: triplet[j]))]
            decoded.append(f"{asp}:{s}")
        res.append(", ".join(decoded))
    return pd.Series(res)

df_final = df_pred.withColumn("decoded", decode_sentiment(F.col("predictions")))
for asp in ASPECTS:
    df_final = df_final.withColumn(asp, F.regexp_extract("decoded", f"{asp}:(\\w+)", 1))

# === Giải mã Review JSON thành text tiếng Việt trước khi stream ===
review_schema = T.StructType([
    T.StructField("id", T.StringType()),
    T.StructField("review", T.StringType())
])
df_final = df_final.withColumn("ReviewText", from_json(col("Review"), review_schema).getField("review"))

# === 5. Ghi kết quả vào PostgreSQL (chuẩn UTF-8, log đầy đủ, xử lý lỗi an toàn) ===
def write_to_postgres(batch_df, batch_id):
    sys.stdout.reconfigure(encoding='utf-8')
    total_rows = batch_df.count()

    if total_rows == 0:
        print(f"[Batch {batch_id}] ⚠️ Không có dữ liệu mới.")
        return

    preview = batch_df.select("ReviewText", *ASPECTS).limit(5).toPandas().to_dict(orient="records")
    print(f"\n[Batch {batch_id}] Nhận {total_rows} dòng, hiển thị 5 dòng đầu:")
    print(json.dumps(preview, ensure_ascii=False, indent=2))

    # Giả lập lỗi mô phỏng để test Airflow restart
    if batch_id % 5 == 0:
        print(f"[Batch {batch_id}] 💥 Giả lập sự cố: crash mô phỏng.")
        raise Exception(f"Simulated crash at batch {batch_id}")

    try:
        (batch_df
            .select("ReviewText", *ASPECTS)
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
        subset = batch_df.select("ReviewText", *ASPECTS).limit(3).toPandas().to_dict(orient="records")
        print(f"[Batch {batch_id}] Dữ liệu đã ghi (mẫu):")
        print(json.dumps(subset, ensure_ascii=False, indent=2))

    except Exception as e:
        print(f"[Batch {batch_id}] ⚠️ Không thể ghi vào PostgreSQL, ghi log ra console thay thế.")
        print(f"Lỗi: {str(e)}")
        subset = batch_df.select("ReviewText", *ASPECTS).limit(5).toPandas().to_dict(orient="records")
        print(json.dumps(subset, ensure_ascii=False, indent=2))

# === 6. Bắt đầu stream ===
query = (
    df_final.writeStream
    .foreachBatch(write_to_postgres)
    .outputMode("append")
    .trigger(processingTime="5 seconds")
    .start()
)

print("🚀 Streaming job started — đang lắng nghe dữ liệu từ Kafka...")
query.awaitTermination()
