import pandas as pd
import torch
import json
import os
import re
import unicodedata
from collections import Counter
from torch.nn.utils.rnn import pad_sequence

# CONFIG
DATA_DIR = "/opt/airflow/projects/absa_streaming/data"
TMP_DIR = "/opt/airflow/models/tmp"
TRAIN_CSV = f"{DATA_DIR}/train_data.csv"
VAL_CSV = f"{DATA_DIR}/val_data.csv"
MAX_VOCAB = 6000

# Ensure tmp exists
os.makedirs(TMP_DIR, exist_ok=True)

# === PREPROCESSING LOGIC (From Notebook) ===
REMOVE_TONE = False
REMOVE_NOISE = True

def remove_vietnamese_tone(text):
    text = unicodedata.normalize('NFD', text)
    text = re.sub(r'[\u0300-\u036f]', '', text)
    text = unicodedata.normalize('NFC', text)
    return text

def remove_non_vietnamese_chars(text):
    return re.sub(
        r"[^a-zàáạảãăắằặẳẵâầấậẩẫèéẹẻẽêềếệểễ"
        r"ìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữ"
        r"ỳýỵỷỹđ0-9\s]",
        " ",
        text
    )

def clean(text):
    text = str(text).lower().strip()
    if REMOVE_TONE:
        text = remove_vietnamese_tone(text)
    if REMOVE_NOISE:
        text = remove_non_vietnamese_chars(text)
    text = re.sub(r"\s+", " ", text)
    return text

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
        encoded_list.append(torch.tensor([vocab.get(t, 1) for t in text.split()], dtype=torch.long))
    return pad_sequence(encoded_list, batch_first=True, padding_value=0)

print("Encoding tensors...")
X_train = encode_and_pad(train_df, vocab)
X_val = encode_and_pad(val_df, vocab)

# Process Labels
ASPECT_COLUMNS = ['Price', 'Shipping', 'Outlook', 'Quality', 'Size', 'Shop_Service', 'General', 'Others']
# Map: -1->0, 0->1, 1->2, 2->3
y_train = torch.tensor(train_df[ASPECT_COLUMNS].values, dtype=torch.long) + 1
y_val = torch.tensor(val_df[ASPECT_COLUMNS].values, dtype=torch.long) + 1

# Save Artifacts
print(f"Saving artifacts to {TMP_DIR}...")
torch.save(X_train, f"{TMP_DIR}/train_tensor_x.pt")
torch.save(y_train, f"{TMP_DIR}/train_tensor_y.pt")
torch.save(X_val, f"{TMP_DIR}/val_tensor_x.pt")
torch.save(y_val, f"{TMP_DIR}/val_tensor_y.pt")

print("Preprocessing Complete.")