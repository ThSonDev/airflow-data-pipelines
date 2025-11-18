import psycopg2
import json
import shutil
import os

# CONFIG
TMP_DIR = "/opt/airflow/models/tmp"
NEWER_MODELS_DIR = "/opt/airflow/models/newer_models"
METRICS_FILE = f"{TMP_DIR}/metrics.json"
MODEL_FILE = f"{TMP_DIR}/model_temp.pth"

# DB CONFIG (Hardcoded based on previous context)
DB_HOST = "postgres"
DB_PORT = 5432
DB_NAME = "airflow"
DB_USER = "airflow"
DB_PASS = "airflow"

# === LOGIC ===
def main():
    if not os.path.exists(METRICS_FILE):
        print("Error: metrics.json not found.")
        exit(1)

    with open(METRICS_FILE, "r") as f:
        metrics = json.load(f)
    
    new_f1 = metrics["macro_f1"]
    new_acc = metrics["balanced_acc"]

    try:
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
        )
        cur = conn.cursor()
        
        # Ensure table exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS retrain_results (
                id SERIAL PRIMARY KEY,
                model_name TEXT NOT NULL,
                macro_f1 DOUBLE PRECISION NOT NULL,
                balanced_acc DOUBLE PRECISION NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
            );
        """)
        
        # Get best previous model
        cur.execute("SELECT macro_f1 FROM retrain_results ORDER BY macro_f1 DESC LIMIT 1")
        row = cur.fetchone()
        
        current_best_f1 = row[0] if row else 0.0
        
        print(f"Current Best F1: {current_best_f1}")
        print(f"New Candidate F1: {new_f1}")

        if new_f1 > current_best_f1:
            print("🚀 New model is BETTER. Deploying...")
            
            # 1. Get Next ID for naming
            cur.execute("SELECT nextval('retrain_results_id_seq')")
            next_id = cur.fetchone()[0]
            
            new_model_name = f"cnn_v{next_id}.pth"
            dest_path = f"{NEWER_MODELS_DIR}/{new_model_name}"
            
            # 2. Move file
            os.makedirs(NEWER_MODELS_DIR, exist_ok=True)
            shutil.move(MODEL_FILE, dest_path)
            print(f"Model moved to: {dest_path}")
            
            # 3. Insert record
            cur.execute("""
                INSERT INTO retrain_results (id, model_name, macro_f1, balanced_acc)
                VALUES (%s, %s, %s, %s)
            """, (next_id, new_model_name, new_f1, new_acc))
            
            conn.commit()
            print("✅ Database updated.")
            
        else:
            print("📉 New model is NOT better. Discarding.")
    
    except Exception as e:
        print(f"Database error: {e}")
        if conn: conn.rollback()
    finally:
        if conn: conn.close()

if __name__ == "__main__":
    main()