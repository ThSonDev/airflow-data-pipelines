import pandas as pd
import torch
import json
import os
from collections import Counter
from torch.nn.utils.rnn import pad_sequence

from absa_core import ASPECTS, clean, encode, get_settings

# CONFIG
cfg = get_settings()
DATA_DIR = cfg.data_dir
TMP_DIR = cfg.tmp_dir
TRAIN_CSV = f"{DATA_DIR}/train_data.csv"
VAL_CSV = f"{DATA_DIR}/val_data.csv"
MAX_VOCAB = 6000

# Ensure tmp exists
os.makedirs(TMP_DIR, exist_ok=True)

# === MAIN EXECUTION ===
print("Loading CSVs...")
train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

# Clean
print("Cleaning text...")
train_df["clean_text"] = train_df["Review"].apply(clean)
val_df["clean_text"] = val_df["Review"].apply(clean)

# Build Vocab
print("Building Vocabulary...")
counter = Counter()
for text in train_df["clean_text"]:
    counter.update(text.split())

vocab = {w: i+2 for i, (w, _) in enumerate(counter.most_common(MAX_VOCAB))}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

# Save Vocab
with open(f"{TMP_DIR}/vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

# Encode and Pad
def encode_and_pad(df, vocab):
    encoded_list = []
    for text in df["clean_text"]:
        encoded_list.append(torch.tensor(encode(text, vocab), dtype=torch.long))
    return pad_sequence(encoded_list, batch_first=True, padding_value=0)

print("Encoding tensors...")
X_train = encode_and_pad(train_df, vocab)
X_val = encode_and_pad(val_df, vocab)

# Process Labels
# Map: -1->0, 0->1, 1->2, 2->3
y_train = torch.tensor(train_df[ASPECTS].values, dtype=torch.long) + 1
y_val = torch.tensor(val_df[ASPECTS].values, dtype=torch.long) + 1

# Save Artifacts
print(f"Saving artifacts to {TMP_DIR}...")
torch.save(X_train, f"{TMP_DIR}/train_tensor_x.pt")
torch.save(y_train, f"{TMP_DIR}/train_tensor_y.pt")
torch.save(X_val, f"{TMP_DIR}/val_tensor_x.pt")
torch.save(y_val, f"{TMP_DIR}/val_tensor_y.pt")

print("Preprocessing Complete.")