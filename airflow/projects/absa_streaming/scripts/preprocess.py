import json
import os
from collections import Counter

import pandas as pd
import torch
from absa_core import ASPECTS, clean, encode, get_logger, get_settings
from text_utils import drop_overlap

log = get_logger("preprocess")
cfg = get_settings()
TMP_DIR = cfg.tmp_dir
MAX_VOCAB = 6000

os.makedirs(TMP_DIR, exist_ok=True)

log.info("Loading and cleaning CSVs")
train_df = pd.read_csv(f"{cfg.data_dir}/train_data.csv")
val_df = pd.read_csv(f"{cfg.data_dir}/val_data.csv")
train_df["clean_text"] = train_df["Review"].apply(clean)
val_df["clean_text"] = val_df["Review"].apply(clean)
test_df = pd.read_csv(f"{cfg.data_dir}/test_data.csv")
test_df["clean_text"] = test_df["Review"].apply(clean)
# The same review in train and val/test would inflate the scores
train_df, dropped = drop_overlap(train_df, val_df, test_df)
log.info("Dropped %d train rows that also appear in val or test", dropped)

counter = Counter()
for text in train_df["clean_text"]:
    counter.update(text.split())

vocab = {w: i + 2 for i, (w, _) in enumerate(counter.most_common(MAX_VOCAB))}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1
with open(f"{TMP_DIR}/vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)
log.info("Vocabulary size: %d", len(vocab))


# Unpadded id lists: train and eval pad each batch to its own length
X_train = [encode(t, vocab) for t in train_df["clean_text"]]
X_val = [encode(t, vocab) for t in val_df["clean_text"]]

# CSV labels -1..2 become class ids 0..3
y_train = torch.tensor(train_df[ASPECTS].values, dtype=torch.long) + 1
y_val = torch.tensor(val_df[ASPECTS].values, dtype=torch.long) + 1

log.info("Saving artifacts to %s", TMP_DIR)
torch.save(X_train, f"{TMP_DIR}/train_tensor_x.pt")
torch.save(y_train, f"{TMP_DIR}/train_tensor_y.pt")
torch.save(X_val, f"{TMP_DIR}/val_tensor_x.pt")
torch.save(y_val, f"{TMP_DIR}/val_tensor_y.pt")
log.info("Preprocessing complete")
