# inference_transformer.py

import math
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# -----------------------------------------------------------
# Paths & config (must match train script)
# -----------------------------------------------------------

DATA_FILE_DEFAULT = Path(
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

SEQ_LEN_DEFAULT = 64
STRIDE_DEFAULT = 8


# -----------------------------------------------------------
# Model definition (must match training)
# -----------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
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
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.fc_out = nn.Linear(d_model, num_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        out = self.fc_out(x)
        return out


# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------

def build_sequences_for_project(
    df: pd.DataFrame,
    project_id: str,
    seq_len: int,
    stride: int,
    feature_cols,
):
    """
    Build sliding-window sequences for a single project.
    """
    g = df[df["project_id"] == project_id].sort_values("t")

    if g.empty:
        raise ValueError(f"Project_id {project_id} not found in data file")

    feat = g[feature_cols].values.astype("float32")

    if len(g) < seq_len:
        raise ValueError(
            f"Project_id {project_id} has only {len(g)} timesteps, "
            f"but seq_len={seq_len}"
        )

    sequences = []
    for start in range(0, len(g) - seq_len + 1, stride):
        end = start + seq_len
        sequences.append(feat[start:end])

    X = np.stack(sequences, axis=0)  # (num_seqs, seq_len, num_features)
    return X, g


def load_model_and_norm():
    """
    Load trained Transformer model and normalisation stats (features + targets).
    """
    norm = np.load(NORM_FILE)

    feat_mean = norm["mean"]
    feat_std = norm["std"]
    target_mean = norm["target_mean"]
    target_std = norm["target_std"]

    feature_cols = norm["feature_cols"].tolist()
    target_cols = norm["target_cols"].tolist()

    if list(feature_cols) != FEATURES:
        print("[warning] Feature list in norm file differs from this script.")
        print("  From file:", feature_cols)
        print("  In code:  ", FEATURES)

    if list(target_cols) != TARGETS:
        print("[warning] Target list in norm file differs from this script.")
        print("  From file:", target_cols)
        print("  In code:  ", TARGETS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PaymentTransformer(
        input_dim=len(FEATURES),
        num_targets=len(TARGETS),
    )
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.to(device)
    model.eval()

    return model, device, feat_mean, feat_std, target_mean, target_std


def predict_for_project(
    project_id: str,
    data_file: Path,
    seq_len: int,
    stride: int,
):
    """
    Run Transformer inference for a single project_id, returning
    un-normalised predictions (weeks_late, cost_overrun_percent).
    """
    print(f"[infer] Loading data from: {data_file}")
    df = pd.read_csv(data_file)

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Missing required feature columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    X, g = build_sequences_for_project(df, project_id, seq_len, stride, FEATURES)
    print(
        f"[infer] Project {project_id}: {len(g)} timesteps, "
        f"{X.shape[0]} sequences of length {seq_len}"
    )

    model, device, feat_mean, feat_std, target_mean, target_std = load_model_and_norm()

    # Normalise features using training stats
    X_norm = (X - feat_mean) / feat_std
    X_tensor = torch.from_numpy(X_norm).to(device)

    with torch.no_grad():
        preds_norm = model(X_tensor)  # (num_seqs, num_targets)

    preds_norm_np = preds_norm.cpu().numpy()
    # Un-normalise to get predictions in original units
    preds_raw = preds_norm_np * target_std + target_mean

    pred_weeks_late = preds_raw[:, 0]
    pred_cost_overrun = preds_raw[:, 1]

    # Aggregate to project-level prediction (mean over windows)
    proj_weeks_late_mean = float(pred_weeks_late.mean())
    proj_weeks_late_std = float(pred_weeks_late.std())
    proj_cost_overrun_mean = float(pred_cost_overrun.mean())
    proj_cost_overrun_std = float(pred_cost_overrun.std())

    print("\n[infer] Transformer predictions (original units, aggregated):")
    print(
        f"  weeks_late:      mean={proj_weeks_late_mean:.2f}, "
        f"std={proj_weeks_late_std:.2f}"
    )
    print(
        f"  cost_overrun_%:  mean={proj_cost_overrun_mean:.2f}, "
        f"std={proj_cost_overrun_std:.2f}"
    )

    # If actual targets exist, show for comparison
    if all(col in g.columns for col in TARGETS):
        true_weeks = g["weeks_late"].iloc[0]
        true_over = g["cost_overrun_percent"].iloc[0]
        print("\n[infer] Ground truth from data (project-level):")
        print(f"  true_weeks_late:     {true_weeks:.2f}")
        print(f"  true_cost_overrun%:  {true_over:.2f}")

    return {
        "project_id": project_id,
        "pred_weeks_late_mean": proj_weeks_late_mean,
        "pred_weeks_late_std": proj_weeks_late_std,
        "pred_cost_overrun_mean": proj_cost_overrun_mean,
        "pred_cost_overrun_std": proj_cost_overrun_std,
    }


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Infer delay/overrun with Transformer")
    p.add_argument(
        "--data-file",
        type=str,
        default=str(DATA_FILE_DEFAULT),
        help="CSV with timeseries data",
    )
    p.add_argument(
        "--project-id",
        type=str,   # project IDs like 'P0002'
        required=True,
        help="project_id to predict for (string)",
    )
    p.add_argument(
        "--seq-len",
        type=int,
        default=SEQ_LEN_DEFAULT,
        help="sequence length (must match training for best results)",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=STRIDE_DEFAULT,
        help="stride for sliding window",
    )
    return p.parse_args()


def main():
    args = parse_args()
    result = predict_for_project(
        project_id=args.project_id,
        data_file=Path(args.data_file),
        seq_len=args.seq_len,
        stride=args.stride,
    )
    print("\n[infer] Result dict:", result)


if __name__ == "__main__":
    main()
