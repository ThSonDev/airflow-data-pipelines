import json

import numpy as np
import torch
from absa_core import (
    ASPECTS,
    EMBED_DIM,
    TextCNN,
    collate_batch,
    get_logger,
    get_settings,
    load_model_config,
)
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

log = get_logger("eval")
TMP_DIR = get_settings().tmp_dir
BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(f"{TMP_DIR}/vocab.json", "r") as f:
    vocab = json.load(f)

X_val = torch.load(f"{TMP_DIR}/val_tensor_x.pt")
y_val = torch.load(f"{TMP_DIR}/val_tensor_y.pt")  # (N, 8), class ids 0-3
loader = DataLoader(list(zip(X_val, y_val)), batch_size=BATCH_SIZE, collate_fn=collate_batch)

MODEL_FILE = f"{TMP_DIR}/model_temp.pth"
mask = load_model_config(MODEL_FILE)["mask_padding"]
model = TextCNN(len(vocab), EMBED_DIM, len(ASPECTS), mask_padding=mask).to(device)
model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
model.eval()

all_preds = []
all_labels = []

log.info("Running inference")
with torch.no_grad():
    for X_b, y_b in loader:
        outputs = model(X_b.to(device))  # (batch, 8, 4)
        all_preds.extend(outputs.argmax(dim=2).cpu().numpy())
        all_labels.extend(y_b.numpy())

y_true_flat = np.array(all_labels).flatten()
y_pred_flat = np.array(all_preds).flatten()

f1 = f1_score(y_true_flat, y_pred_flat, average='macro', labels=[0, 1, 2, 3], zero_division=0)
acc = balanced_accuracy_score(y_true_flat, y_pred_flat)
metrics = {"macro_f1": float(f1), "balanced_acc": float(acc)}
log.info("Evaluation results: %s", metrics)

with open(f"{TMP_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f)
