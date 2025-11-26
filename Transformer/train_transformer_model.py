# train_transformer_model.py

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------------------------------------
# Paths & training hyperparameters
# -----------------------------------------------------------

DATA_FILE = Path(
    "C:/Users/blake/Desktop/Project Panther/Transformer/transformer_training_data.csv"
)
MODEL_FILE = Path(
    "C:/Users/blake/Desktop/Project Panther/Transformer/transformer_model.pt"
)
NORM_FILE = Path(
    "C:/Users/blake/Desktop/Project Panther/Transformer/transformer_norm_stats.npz"
)

FEATURES = [
    "t",
    "days_since_first_payment",
    "payment_amount",
    "cum_project_paid",
    "pct_project_paid",
    "project_value",
]

TARGETS = ["weeks_late", "cost_overrun_percent"]

SEQ_LEN = 64
STRIDE = 8
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
RANDOM_SEED = 42


# -----------------------------------------------------------
# Data utilities
# -----------------------------------------------------------

def build_sequences(df: pd.DataFrame,
                    seq_len: int,
                    stride: int,
                    feature_cols,
                    target_cols):
    """
    Build sliding-window sequences per project.

    Returns
    -------
    X : np.ndarray of shape (num_sequences, seq_len, num_features)
    y : np.ndarray of shape (num_sequences, num_targets)
    """
    sequences = []
    targets = []

    for project_id, g in df.groupby("project_id"):
        g = g.sort_values("t")

        feat = g[feature_cols].values.astype("float32")
        # project-level targets: same for all timesteps
        tgt = g[target_cols].iloc[0].values.astype("float32")

        if len(g) < seq_len:
            continue  # not enough timesteps for even one sequence

        for start in range(0, len(g) - seq_len + 1, stride):
            end = start + seq_len
            seq = feat[start:end]
            sequences.append(seq)
            targets.append(tgt)

    if not sequences:
        return (
            np.empty((0, seq_len, len(feature_cols)), dtype="float32"),
            np.empty((0, len(target_cols)), dtype="float32"),
        )

    X = np.stack(sequences, axis=0)
    y = np.stack(targets, axis=0)
    return X, y


class PaymentSequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.shape[0] == y.shape[0], "X and y must have same number of samples"
        self.X = torch.from_numpy(X)  # (N, T, F)
        self.y = torch.from_numpy(y)  # (N, targets)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -----------------------------------------------------------
# Model: Transformer Encoder for sequences
# -----------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class PaymentTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_targets: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # We'll use the last time-step representation for regression
        self.fc_out = nn.Linear(d_model, num_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim)
        returns: (batch, num_targets)
        """
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # last timestep
        out = self.fc_out(x)
        return out


# -----------------------------------------------------------
# Training loop
# -----------------------------------------------------------

def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 1. Load data
    print(f"[transformer] Loading data from: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)

    missing = [c for c in FEATURES + TARGETS if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"The following required columns are missing in {DATA_FILE.name}: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # 2. Project-level train/val split
    project_ids = sorted(df["project_id"].unique())
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(project_ids)
    split_idx = int(0.8 * len(project_ids))
    train_ids = set(project_ids[:split_idx])
    val_ids = set(project_ids[split_idx:])

    train_df = df[df["project_id"].isin(train_ids)]
    val_df = df[df["project_id"].isin(val_ids)]

    print(f"[transformer] Projects: total={len(project_ids)}, "
          f"train={len(train_ids)}, val={len(val_ids)}")

    # 3. Build sequences
    X_train, y_train = build_sequences(
        train_df, seq_len=SEQ_LEN, stride=STRIDE,
        feature_cols=FEATURES, target_cols=TARGETS
    )
    X_val, y_val = build_sequences(
        val_df, seq_len=SEQ_LEN, stride=STRIDE,
        feature_cols=FEATURES, target_cols=TARGETS
    )

    print(f"[transformer] Built sequences:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val:   X={X_val.shape}, y={y_val.shape}")

    if X_train.shape[0] == 0:
        raise RuntimeError(
            "No training sequences were created (X_train is empty).\n"
            "This usually means SEQ_LEN is too long, STRIDE too large, "
            "or the grouping/feature columns are wrong."
        )

    if X_val.shape[0] == 0:
        print("[transformer] WARNING: no validation sequences were created.")

    # 4. Normalise features using training set only
    flat_train = X_train.reshape(-1, X_train.shape[-1])  # (N*T, F)
    feat_mean = flat_train.mean(axis=0)
    feat_std = flat_train.std(axis=0)
    feat_std = np.where(feat_std == 0, 1.0, feat_std)  # avoid division by zero

    X_train = (X_train - feat_mean) / feat_std
    X_val = (X_val - feat_mean) / feat_std

    # 5. Normalise targets (critical for stable learning)
    target_mean = y_train.mean(axis=0)
    target_std = y_train.std(axis=0)
    target_std = np.where(target_std == 0, 1.0, target_std)

    y_train_norm = (y_train - target_mean) / target_std
    y_val_norm = (y_val - target_mean) / target_std

    # 6. Save normalisation stats
    np.savez(
        NORM_FILE,
        mean=feat_mean,
        std=feat_std,
        target_mean=target_mean,
        target_std=target_std,
        feature_cols=np.array(FEATURES),
        target_cols=np.array(TARGETS),
    )
    print(f"[transformer] Saved norm stats to: {NORM_FILE}")

    # 7. Datasets & loaders (using normalised targets)
    train_ds = PaymentSequenceDataset(X_train, y_train_norm)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )

    val_loader = None
    if X_val.shape[0] > 0:
        val_ds = PaymentSequenceDataset(X_val, y_val_norm)
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
        )

    print(f"[transformer] Train samples: {len(train_ds)}")
    if val_loader is not None:
        print(f"[transformer] Val samples:   {len(val_loader.dataset)}")

    # 8. Model, optimizer, loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PaymentTransformer(
        input_dim=len(FEATURES),
        num_targets=len(TARGETS),
    ).to(device)

    criterion = nn.MSELoss()  # on normalised target space
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # 9. Training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)

        train_loss = running_loss / len(train_ds)

        # Validation
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_running = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    preds = model(X_batch)
                    loss = criterion(preds, y_batch)
                    val_running += loss.item() * X_batch.size(0)

            val_loss = val_running / len(val_loader.dataset)

        if val_loss is not None:
            print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}")
        else:
            print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f}")

    # 10. Save model
    torch.save(model.state_dict(), MODEL_FILE)
    print(f"[transformer] Saved model to: {MODEL_FILE}")


if __name__ == "__main__":
    main()
