import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from absa_core import (
    ASPECTS,
    EMBED_DIM,
    NUM_CLASSES,
    TextCNN,
    collate_batch,
    get_logger,
    get_settings,
    set_seed,
)
from metrics import compute_metrics
from model_registry import predict_logits
from torch.utils.data import DataLoader

log = get_logger("train")
cfg = get_settings()
TMP_DIR = cfg.tmp_dir
EPOCHS = 100
PATIENCE = 10
BATCH_SIZE = 64

set_seed(cfg.seed)
X_train = torch.load(f"{TMP_DIR}/train_tensor_x.pt")
y_train = torch.load(f"{TMP_DIR}/train_tensor_y.pt")
X_val = torch.load(f"{TMP_DIR}/val_tensor_x.pt")
y_val = torch.load(f"{TMP_DIR}/val_tensor_y.pt")
with open(f"{TMP_DIR}/vocab.json", "r") as f:
    vocab = json.load(f)

shuffle_rng = torch.Generator().manual_seed(cfg.seed)  # same batch order for the same seed
train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_batch, generator=shuffle_rng)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Using device: %s, seed: %d", device, cfg.seed)

model = TextCNN(len(vocab), EMBED_DIM, len(ASPECTS), mask_padding=True).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

best_f1, best_epoch, no_improve = -1.0, 0, 0

with open(f"{TMP_DIR}/config.json", "w") as f:
    json.dump({"mask_padding": True}, f)

log.info("Starting training")
for epoch in range(1, EPOCHS + 1):
    start = time.time()
    model.train()
    train_loss = 0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_b).view(-1, NUM_CLASSES), y_b.view(-1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    logits = predict_logits(model, X_val, device, BATCH_SIZE)
    val_loss = criterion(logits.view(-1, NUM_CLASSES), y_val.reshape(-1)).item()
    val_f1 = compute_metrics(y_val.numpy(), logits.argmax(dim=2).numpy())["mean_aspect_f1"]
    log.info("Epoch %d | train loss %.4f | val loss %.4f | val mean aspect F1 %.4f | %.2fs",
             epoch, avg_train_loss, val_loss, val_f1, time.time() - start)

    if val_f1 > best_f1:
        best_f1, best_epoch, no_improve = val_f1, epoch, 0
        torch.save(model.state_dict(), f"{TMP_DIR}/model_temp.pth")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            log.info("Early stopping at epoch %d", epoch)
            break

log.info("Training complete: best epoch %d, best val mean aspect F1 %.4f, saved to model_temp.pth",
         best_epoch, best_f1)
