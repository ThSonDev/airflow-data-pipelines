import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import json

from absa_core import ASPECTS, EMBED_DIM, NUM_CLASSES, TextCNN, get_settings

# CONFIG
TMP_DIR = get_settings().tmp_dir
EPOCHS = 50  # Reduced for example, adjust as needed
PATIENCE = 10
BATCH_SIZE = 64

# === LOAD DATA ===
print("Loading data...")
X_train = torch.load(f"{TMP_DIR}/train_tensor_x.pt")
y_train = torch.load(f"{TMP_DIR}/train_tensor_y.pt")
X_val = torch.load(f"{TMP_DIR}/val_tensor_x.pt")
y_val = torch.load(f"{TMP_DIR}/val_tensor_y.pt")

with open(f"{TMP_DIR}/vocab.json", "r") as f:
    vocab = json.load(f)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

# === TRAINING SETUP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = TextCNN(len(vocab), EMBED_DIM, len(ASPECTS)).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

best_val_loss = float('inf')
no_improve = 0

# === TRAINING LOOP ===
print("Starting training...")
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

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            output = model(X_b)
            loss = criterion(output.view(-1, NUM_CLASSES), y_b.view(-1))
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

    # Early Stopping & Save
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        no_improve = 0
        torch.save(model.state_dict(), f"{TMP_DIR}/model_temp.pth")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

print("Training complete. Best model saved to model_temp.pth")