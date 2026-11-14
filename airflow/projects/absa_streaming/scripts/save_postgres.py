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
    for label, m in (("candidate", cand), ("production " + prod["name"], prod)):
        log.info("%s: val mean aspect F1 %.4f (decision), test mean aspect F1 %.4f (reported)",
                 label, m["val_mean_aspect_f1"], m["mean_aspect_f1"])
    if not should_promote(cand["val_mean_aspect_f1"], prod["val_mean_aspect_f1"]):
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
        # Old rows were scored on val with flattened F1, so they have no eval_split and are ignored
        cur.execute("ALTER TABLE retrain_results ADD COLUMN IF NOT EXISTS mean_aspect_f1 DOUBLE PRECISION")
        cur.execute("ALTER TABLE retrain_results ADD COLUMN IF NOT EXISTS eval_split TEXT")
        cur.execute("SELECT nextval('retrain_results_id_seq')")
        next_id = cur.fetchone()[0]
        name = f"cnn_v{next_id}"
        target = f"{cfg.models_dir}/newer_models/{name}"
        build_folder(cfg.tmp_dir, target, metrics)
        final_dir = target
        cur.execute(
            "INSERT INTO retrain_results (id, model_name, macro_f1, balanced_acc, mean_aspect_f1, eval_split) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (next_id, name, cand["macro_f1_flat"], cand["balanced_acc"], cand["mean_aspect_f1"], cand["split"]),
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
