import os

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from mamba_ssm import Mamba
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import seaborn as sns

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False
})

SEQ_LEN = 10
PRED_LEN = 1
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-4
LAMBDA_PHY = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df = pd.read_csv("data.csv", parse_dates=["timestamp"])

for lag in range(1, SEQ_LEN + 1):
    df[f'PV_Lag_{lag}'] = df['PV_Active_Power'].shift(lag)

df = df.dropna().reset_index(drop=True)

features = [
    'Temperature', 'Humidity',
    'Global_Horizontal_Radiation', 'Diffuse_Horizontal_Radiation',
    'Tilted_Global_Radiation', 'Tilted_Diffuse_Radiation',
    'imfs0', 'imfs1', 'imfs2', 'imfs3', 'imfs4'
]+ [f'PV_Lag_{lag}' for lag in range(1, SEQ_LEN + 1)]
target = 'PV_Active_Power'

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
x_scaled = scaler_x.fit_transform(df[features])
y_scaled = scaler_y.fit_transform(df[[target]])

split_idx = int(len(x_scaled) * 0.8)
x_train, x_val = x_scaled[:split_idx], x_scaled[split_idx:]
y_train, y_val = y_scaled[:split_idx], y_scaled[split_idx:]

class TimeSeriesDataset(Dataset):
    def __init__(self, x, y, seq_len=10, pred_len=1):
        self.x, self.y = [], []
        for i in range(len(x) - seq_len - pred_len):
            self.x.append(x[i:i+seq_len])
            self.y.append(y[i+seq_len:i+seq_len+pred_len])
        self.x = torch.tensor(np.array(self.x), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

train_dataset = TimeSeriesDataset(x_train, y_train, SEQ_LEN, PRED_LEN)
val_dataset = TimeSeriesDataset(x_val, y_val, SEQ_LEN, PRED_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

class MambaForecastModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.mamba_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                Mamba(d_model=hidden_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.mamba_layers:
            x = x + layer(x)
        x = x[:, -1, :]
        return self.output(x)

model = MambaForecastModel(input_dim=len(features)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

def physics_loss(xb, pred):
    idx_g_tilted = features.index("Tilted_Global_Radiation")
    idx_temp = features.index("Temperature")
    idx_humidity = features.index("Humidity")

    G_tilted = xb[:, -1, idx_g_tilted].unsqueeze(1)
    T = xb[:, -1, idx_temp].unsqueeze(1)
    H = xb[:, -1, idx_humidity].unsqueeze(1)
    alpha_temp = 0.004
    eta_temp = 1 - alpha_temp * (T - 25)

    beta_humid = 0.002
    eta_humid = 1 - beta_humid * (H - 50).abs()

    pv_theory = G_tilted * eta_temp * eta_humid

    return nn.MSELoss()(pred, pv_theory)

best_rmse = float("inf")
best_model_state = None
no_improve_count = 0
patience = 20
min_delta = 1e-4
save_path = "best_model.pth"

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5,
    threshold=1e-4,
    verbose=True,
    min_lr=1e-7
)


for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss_data = loss_fn(pred, yb.view(-1, 1))
        loss_phy = physics_loss(xb, pred)
        loss = loss_data + LAMBDA_PHY * loss_phy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in DataLoader(val_dataset, batch_size=1, shuffle=False):
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            preds.append(pred.cpu().item())
            trues.append(yb.cpu().item())

    preds_inv = scaler_y.inverse_transform(np.array(preds).reshape(-1, 1))
    trues_inv = scaler_y.inverse_transform(np.array(trues).reshape(-1, 1))
    rmse = np.sqrt(np.mean((preds_inv - trues_inv) ** 2))

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {total_loss / len(train_loader):.6f} - Val RMSE: {rmse:.4f}")

    scheduler.step(rmse)

    if best_rmse - rmse > min_delta:
        best_rmse = rmse
        best_model_state = model.state_dict()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_name = "mamba"
        save_path = f"checkpoints/{model_name}_best_{timestamp}.pth"
        os.makedirs("checkpoints", exist_ok=True)

        torch.save(best_model_state, save_path)
        print(f"Best model saved at epoch {epoch + 1} to {save_path}")
        no_improve_count = 0
    else:
        no_improve_count += 1
        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best RMSE: {best_rmse:.4f}")
            break

model.load_state_dict(torch.load(save_path))
