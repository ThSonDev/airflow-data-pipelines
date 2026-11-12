import json
import os
import shutil

import psycopg2
from absa_core import get_logger, get_settings

log = get_logger("save_postgres")
cfg = get_settings()
TMP_DIR = cfg.tmp_dir
NEWER_MODELS_DIR = f"{cfg.models_dir}/newer_models"
METRICS_FILE = f"{TMP_DIR}/metrics.json"
MODEL_FILE = f"{TMP_DIR}/model_temp.pth"


def main():
    if not os.path.exists(METRICS_FILE):
        log.error("metrics.json not found")
        exit(1)

    with open(METRICS_FILE, "r") as f:
        metrics = json.load(f)
    new_f1 = metrics["macro_f1"]
    new_acc = metrics["balanced_acc"]

    try:
        conn = psycopg2.connect(
            host=cfg.postgres_host, port=cfg.postgres_port, dbname=cfg.postgres_db,
            user=cfg.postgres_user, password=cfg.postgres_password
        )
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS retrain_results (
                id SERIAL PRIMARY KEY,
                model_name TEXT NOT NULL,
                macro_f1 DOUBLE PRECISION NOT NULL,
                balanced_acc DOUBLE PRECISION NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
            );
        """)

        cur.execute("SELECT macro_f1 FROM retrain_results ORDER BY macro_f1 DESC LIMIT 1")
        row = cur.fetchone()
        current_best_f1 = row[0] if row else 0.0
        log.info("Current best F1: %s, new candidate F1: %s", current_best_f1, new_f1)

        if new_f1 > current_best_f1:
            log.info("New model is better, deploying")
            cur.execute("SELECT nextval('retrain_results_id_seq')")
            next_id = cur.fetchone()[0]

            new_model_name = f"cnn_v{next_id}.pth"
            dest_path = f"{NEWER_MODELS_DIR}/{new_model_name}"
            os.makedirs(NEWER_MODELS_DIR, exist_ok=True)
            shutil.move(MODEL_FILE, dest_path)
            log.info("Model moved to %s", dest_path)

            cur.execute("""
                INSERT INTO retrain_results (id, model_name, macro_f1, balanced_acc)
                VALUES (%s, %s, %s, %s)
            """, (next_id, new_model_name, new_f1, new_acc))
            conn.commit()
            log.info("Database updated")
        else:
            log.info("New model is not better, discarding")

    except Exception as e:
        log.error("Database error: %s", e)
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    main()
