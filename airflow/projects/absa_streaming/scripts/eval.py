import json

import pandas as pd
import torch
from absa_core import ASPECTS, get_logger, get_settings
from model_registry import load_model, predict_classes, resolve_production_model
from sklearn.metrics import balanced_accuracy_score, f1_score

log = get_logger("eval")
cfg = get_settings()
TMP_DIR = cfg.tmp_dir
SPLIT = "val"  # T4 moves this to the test set
BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv(f"{cfg.data_dir}/{SPLIT}_data.csv")
y_true = (df[ASPECTS].values + 1).flatten()  # CSV labels -1..2 become class ids 0..3


def score(info):
    # Each model encodes the raw text with its own vocab
    model, vocab = load_model(info, device)
    y_pred = predict_classes(model, vocab, df["Review"], device, BATCH_SIZE).flatten()
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2, 3], zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }


candidate = {"model_path": f"{TMP_DIR}/model_temp.pth", "vocab_path": f"{TMP_DIR}/vocab.json"}
production = resolve_production_model()
log.info("Scoring candidate and production model (%s) on the %s set", production["name"], SPLIT)
metrics = {"candidate": score(candidate), "production": {"name": production["name"], **score(production)}}
log.info("Evaluation results: %s", metrics)

with open(f"{TMP_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
