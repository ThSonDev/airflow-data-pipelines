"""Find, load and compare ABSA models. A promoted model is a folder with its own vocab and config."""
import json
import os
import re

import numpy as np
import psycopg2
import torch
from absa_core import (
    ASPECTS,
    EMBED_DIM,
    TextCNN,
    clean,
    encode,
    get_logger,
    get_settings,
    load_model_config,
    pad_batch,
)

log = get_logger("model_registry")
# Only names we create ourselves; also blocks "../x" and absolute paths from a bad DB row
NAME_PATTERN = re.compile(r"cnn_v[0-9]+")

# Old rows like 'cnn_v1.pth' must not hide a valid folder
BEST_MODEL_QUERY = (
    "SELECT model_name FROM retrain_results WHERE model_name ~ '^cnn_v[0-9]+$' "
    "ORDER BY macro_f1 DESC, id DESC LIMIT 1"
)


def connect():
    cfg = get_settings()
    return psycopg2.connect(
        host=cfg.postgres_host, port=cfg.postgres_port, dbname=cfg.postgres_db,
        user=cfg.postgres_user, password=cfg.postgres_password, connect_timeout=5,
    )


def _fetch_best_from_db():
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute(BEST_MODEL_QUERY)
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _model_info(name, model_path, vocab_path):
    return {"name": name, "model_path": model_path, "vocab_path": vocab_path,
            "config": load_model_config(model_path)}


def resolve_production_model(fetch_best=None):
    """Best promoted model from the DB if its folder is complete, else the legacy cnn_best.pth."""
    models_dir = get_settings().models_dir
    legacy = _model_info("cnn_best", f"{models_dir}/cnn_best.pth", f"{models_dir}/vocab.json")
    try:
        name = (fetch_best or _fetch_best_from_db)()
    except Exception as e:
        log.warning("Could not read retrain_results (%s), using cnn_best", e)
        return legacy
    if name is None:
        return legacy
    if not NAME_PATTERN.fullmatch(name):
        log.warning("Ignoring DB model name %r (not cnn_v<number>), using cnn_best", name)
        return legacy
    folder = f"{models_dir}/newer_models/{name}"
    model_path, vocab_path = f"{folder}/model.pth", f"{folder}/vocab.json"
    if not (os.path.isfile(model_path) and os.path.isfile(vocab_path)):
        log.warning("Folder %s is incomplete, using cnn_best", folder)
        return legacy
    return _model_info(name, model_path, vocab_path)


def load_model(info, device):
    """Return (model in eval mode, vocab) for a resolve_production_model style dict."""
    with open(info["vocab_path"], encoding="utf-8") as f:
        vocab = json.load(f)
    mask = load_model_config(info["model_path"])["mask_padding"]
    model = TextCNN(len(vocab), EMBED_DIM, len(ASPECTS), mask_padding=mask)
    model.load_state_dict(torch.load(info["model_path"], map_location=device, weights_only=True))
    return model.to(device).eval(), vocab


def predict_classes(model, vocab, texts, device, batch_size=None):
    """Class ids (N, 8) for raw texts. batch_size None means one batch."""
    texts = list(texts)
    step = batch_size or max(len(texts), 1)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), step):
            ids = pad_batch([encode(clean(t), vocab) for t in texts[i:i + step]]).to(device)
            out.append(model(ids).argmax(dim=2).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(ASPECTS)), dtype=int)


def should_promote(new_score, prod_score):
    return new_score > prod_score
