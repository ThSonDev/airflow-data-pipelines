import json

import pandas as pd
import torch
from absa_core import ASPECTS, get_logger, get_settings
from metrics import compute_metrics
from model_registry import load_model, predict_classes, resolve_production_model

log = get_logger("eval")
cfg = get_settings()
TMP_DIR = cfg.tmp_dir
REPORT_SPLIT = "test"  # reported numbers only
SELECT_SPLIT = "val"  # promotion decision: never select models on the test set
BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = {}
for split in (REPORT_SPLIT, SELECT_SPLIT):
    df = pd.read_csv(f"{cfg.data_dir}/{split}_data.csv")
    data[split] = (df["Review"], df[ASPECTS].values + 1)  # CSV labels -1..2 become class ids 0..3


def score(info):
    # Each model encodes the raw text with its own vocab
    model, vocab = load_model(info, device)
    preds = {s: predict_classes(model, vocab, texts, device, BATCH_SIZE) for s, (texts, _) in data.items()}
    return {
        **compute_metrics(data[REPORT_SPLIT][1], preds[REPORT_SPLIT]),
        "split": REPORT_SPLIT,
        "val_mean_aspect_f1": compute_metrics(data[SELECT_SPLIT][1], preds[SELECT_SPLIT])["mean_aspect_f1"],
    }


candidate = {"model_path": f"{TMP_DIR}/model_temp.pth", "vocab_path": f"{TMP_DIR}/vocab.json"}
production = resolve_production_model()
log.info("Scoring candidate and production model (%s) on %s and %s", production["name"], REPORT_SPLIT, SELECT_SPLIT)
metrics = {"candidate": score(candidate), "production": {"name": production["name"], **score(production)}}
log.info("Evaluation results: %s", metrics)

with open(f"{TMP_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
