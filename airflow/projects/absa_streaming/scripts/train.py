import json

import torch
import torch.nn as nn
import torch.optim as optim
from absa_core import ASPECTS, EMBED_DIM, NUM_CLASSES, TextCNN, get_logger, get_settings
from torch.utils.data import DataLoader, TensorDataset

log = get_logger("train")
TMP_DIR = get_settings().tmp_dir
EPOCHS = 50
PATIENCE = 10
BATCH_SIZE = 64

X_train = torch.load(f"{TMP_DIR}/train_tensor_x.pt")
y_train = torch.load(f"{TMP_DIR}/train_tensor_y.pt")
X_val = torch.load(f"{TMP_DIR}/val_tensor_x.pt")
y_val = torch.load(f"{TMP_DIR}/val_tensor_y.pt")
with open(f"{TMP_DIR}/vocab.json", "r") as f:
    vocab = json.load(f)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Using device: %s", device)

model = TextCNN(len(vocab), EMBED_DIM, len(ASPECTS)).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

best_val_loss = float('inf')
no_improve = 0

log.info("Starting training")
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        output = model(X_b)
        loss = criterion(output.view(-1, NUM_CLASSES), y_b.view(-1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            output = model(X_b)
            loss = criterion(output.view(-1, NUM_CLASSES), y_b.view(-1))
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    log.info("Epoch %d | train loss %.4f | val loss %.4f", epoch, avg_train_loss, avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        no_improve = 0
        torch.save(model.state_dict(), f"{TMP_DIR}/model_temp.pth")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            log.info("Early stopping at epoch %d", epoch)
            break

log.info("Training complete, best model saved to model_temp.pth")
