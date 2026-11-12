import torch
import json
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, balanced_accuracy_score

from absa_core import ASPECTS, EMBED_DIM, TextCNN, get_settings

# CONFIG
TMP_DIR = get_settings().tmp_dir
BATCH_SIZE = 64

# === EVALUATION ===
print("Loading resources for evaluation...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(f"{TMP_DIR}/vocab.json", "r") as f:
    vocab = json.load(f)

X_val = torch.load(f"{TMP_DIR}/val_tensor_x.pt")
y_val = torch.load(f"{TMP_DIR}/val_tensor_y.pt") # Shape: (N, 8) labels 0-3
loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

model = TextCNN(len(vocab), EMBED_DIM, len(ASPECTS)).to(device)
model.load_state_dict(torch.load(f"{TMP_DIR}/model_temp.pth", map_location=device))
model.eval()

all_preds = []
all_labels = []

print("Running inference...")
with torch.no_grad():
    for X_b, y_b in loader:
        X_b = X_b.to(device)
        outputs = model(X_b) # (Batch, 8, 4)
        preds = outputs.argmax(dim=2).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_b.numpy())

# Flatten arrays to calculate global metrics
y_true_flat = np.array(all_labels).flatten()
y_pred_flat = np.array(all_preds).flatten()
known_labels = [0, 1, 2, 3]

f1 = f1_score(y_true_flat, y_pred_flat, average='macro', labels=known_labels, zero_division=0)
acc = balanced_accuracy_score(y_true_flat, y_pred_flat)

metrics = {
    "macro_f1": float(f1),
    "balanced_acc": float(acc)
}

print(f"Evaluation Results: {metrics}")

# Save metrics for the next step
with open(f"{TMP_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f)