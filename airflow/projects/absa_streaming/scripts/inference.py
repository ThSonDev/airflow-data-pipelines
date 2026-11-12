"""Model loading and the pandas UDF used by the Spark consumer."""
import json
import os

import pandas as pd
import psycopg2
import torch
from absa_core import (
    ASPECTS,
    EMBED_DIM,
    PREDICTION_MAP,
    TextCNN,
    clean,
    encode,
    get_logger,
    get_settings,
    pad_batch,
)
from pyspark.sql import types as T
from pyspark.sql.functions import pandas_udf

log = get_logger("inference")
cfg = get_settings()
MODEL_PATH = f"{cfg.models_dir}/cnn_best.pth"
NEWER_MODELS_DIR = f"{cfg.models_dir}/newer_models"
VOCAB_PATH = f"{cfg.models_dir}/vocab.json"

# Loaded once per Python worker
_model = None
_vocab = None
_device = None


def _best_model_path():
    """Path of the best promoted model in the DB, else the default model."""
    try:
        conn = psycopg2.connect(
            host=cfg.postgres_host, port=cfg.postgres_port, database=cfg.postgres_db,
            user=cfg.postgres_user, password=cfg.postgres_password,
        )
        cur = conn.cursor()
        cur.execute("SELECT model_name FROM retrain_results ORDER BY macro_f1 DESC, id DESC LIMIT 1")
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            candidate = os.path.join(NEWER_MODELS_DIR, row[0])
            if os.path.exists(candidate):
                log.info("Using promoted model: %s", candidate)
                return candidate
            log.warning("Best model %s is in the DB but the file is missing, using default", row[0])
    except Exception as e:
        log.warning("Could not check retrain_results (%s), using default model", e)
    return MODEL_PATH


def load_resources():
    global _model, _vocab, _device
    if _vocab is None:
        if os.path.exists(VOCAB_PATH):
            with open(VOCAB_PATH, "r", encoding="utf-8") as f:
                _vocab = json.load(f)
        else:
            log.warning("Vocab file not found at %s, using empty vocab", VOCAB_PATH)
            _vocab = {"<PAD>": 0, "<UNK>": 1}

    if _model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = TextCNN(vocab_size=len(_vocab), embed_dim=EMBED_DIM, num_labels=len(ASPECTS))
        path = _best_model_path()
        if os.path.exists(path):
            _model.load_state_dict(torch.load(path, map_location=_device))
            log.info("Model loaded from %s on %s", path, _device)
        else:
            log.warning("Model file not found at %s, predictions will be random", path)
        _model.to(_device)
        _model.eval()


def _predict(texts: pd.Series) -> pd.Series:
    load_resources()
    batch = [encode(clean(t), _vocab) for t in texts]
    if not batch:
        return pd.Series([[] for _ in texts])

    padded = pad_batch(batch).to(_device)
    with torch.no_grad():
        predictions = _model(padded).argmax(dim=2).cpu().numpy()  # (batch, 8)
    return pd.Series([[PREDICTION_MAP.get(p, "NONE") for p in row] for row in predictions])


# Wrap after defining: a decorator would hide _predict, and cloudpickle would then copy the
# function with its own globals (model stays None) instead of pickling it by reference.
cnn_inference_udf = pandas_udf(_predict, T.ArrayType(T.StringType()))
