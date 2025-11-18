import torch
import torch.nn as nn
import json
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, balanced_accuracy_score

# CONFIG
TMP_DIR = "/opt/airflow/models/tmp"
EMBED_DIM = 128
NUM_FILTERS = 192
WORD_WINDOW = 5
NUM_CLASSES = 4
ASPECT_COLS_LEN = 8
BATCH_SIZE = 64

# === MODEL DEFINITION (Must match training) ===
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_labels):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, NUM_FILTERS, kernel_size=WORD_WINDOW, padding=int(WORD_WINDOW // 2))
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.shared_dense = nn.Linear(NUM_FILTERS, 128)
        self.dropout = nn.Dropout(0.5)
        self.output_heads = nn.ModuleList([
            nn.Linear(128, NUM_CLASSES) for _ in range(num_labels)
        ])

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = torch.relu(self.conv(x))
        x = self.pool(x).squeeze(2)
        x = torch.relu(self.shared_dense(x))
        x = self.dropout(x)
        outputs = []
        for head in self.output_heads:
            outputs.append(head(x))
        return torch.stack(outputs, dim=1)

# === EVALUATION ===
print("Loading resources for evaluation...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(f"{TMP_DIR}/vocab.json", "r") as f:
    vocab = json.load(f)

X_val = torch.load(f"{TMP_DIR}/val_tensor_x.pt")
y_val = torch.load(f"{TMP_DIR}/val_tensor_y.pt") # Shape: (N, 8) labels 0-3
loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

model = TextCNN(len(vocab), EMBED_DIM, ASPECT_COLS_LEN).to(device)
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