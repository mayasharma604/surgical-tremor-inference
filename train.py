import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from data.dataset import TrajectoryDataset
from model.intent_lstm import IntentLSTM

# -------------------------
# Config
# -------------------------
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3

# -------------------------
# Data
# -------------------------
dataset = TrajectoryDataset(
    "data/noisy_mouse_data.npy",
    "data/clean_mouse_data.npy"
)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -------------------------
# Model
# -------------------------
model = IntentLSTM(input_dim=2, hidden_dim=64)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -------------------------
# Training
# -------------------------
for epoch in range(EPOCHS):
    total_loss = 0.0

    for x, y in loader:
        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.6f}")

# -------------------------
# Save
# -------------------------
torch.save(model.state_dict(), "intent_model.pt")
print("✅ Model saved")
