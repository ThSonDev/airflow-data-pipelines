import json
import os
import shutil
import sys

from absa_core import get_logger, get_settings
from model_registry import connect, should_promote

log = get_logger("save_postgres")
FILES = ["model_temp.pth", "vocab.json", "config.json"]


def build_folder(tmp_dir, final_dir, metrics):
    """Copy the candidate into <final_dir>.tmp, then rename: the folder appears complete or not at all."""
    staging = final_dir + ".tmp"
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging)
    try:
        for name in FILES:
            shutil.copy(f"{tmp_dir}/{name}", f"{staging}/{'model.pth' if name == 'model_temp.pth' else name}")
        with open(f"{staging}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        os.rename(staging, final_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    cfg = get_settings()
    metrics_file = f"{cfg.tmp_dir}/metrics.json"
    if not os.path.exists(metrics_file):
        log.error("metrics.json not found")
        sys.exit(1)
    with open(metrics_file) as f:
        metrics = json.load(f)
    cand, prod = metrics["candidate"], metrics["production"]
    log.info("Candidate macro F1 %.4f, production (%s) macro F1 %.4f", cand["macro_f1"], prod["name"], prod["macro_f1"])
    if not should_promote(cand["macro_f1"], prod["macro_f1"]):
        log.info("Candidate is not better, discarding")
        return

    conn = None
    final_dir = None
    try:
        conn = connect()
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
        cur.execute("SELECT nextval('retrain_results_id_seq')")
        next_id = cur.fetchone()[0]
        name = f"cnn_v{next_id}"
        target = f"{cfg.models_dir}/newer_models/{name}"
        build_folder(cfg.tmp_dir, target, metrics)
        final_dir = target
        cur.execute(
            "INSERT INTO retrain_results (id, model_name, macro_f1, balanced_acc) VALUES (%s, %s, %s, %s)",
            (next_id, name, cand["macro_f1"], cand["balanced_acc"]),
        )
        conn.commit()
        log.info("Promoted %s", name)
    except Exception:
        log.exception("Promotion failed")
        if conn:
            conn.rollback()
        if final_dir:
            shutil.rmtree(final_dir, ignore_errors=True)
        sys.exit(1)
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    main()
