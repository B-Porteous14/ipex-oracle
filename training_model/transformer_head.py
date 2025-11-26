# transformer_head.py

from pathlib import Path
import os
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# Base directory for this module (training_model/)
BASE_DIR = Path(__file__).resolve().parent

# Paths can be overridden by environment variables for deployment. By default
# we look for the files next to this module, not in the process working dir.
_default_model_path = BASE_DIR / "transformer_model.pt"
_default_norm_path = BASE_DIR / "transformer_norm_stats.npz"

TRANSFORMER_MODEL_PATH = Path(os.getenv("TRANSFORMER_MODEL_PATH", str(_default_model_path)))
TRANSFORMER_NORM_PATH = Path(os.getenv("TRANSFORMER_NORM_PATH", str(_default_norm_path)))

SEQ_LEN = int(os.getenv("TRANSFORMER_SEQ_LEN", "8"))
STRIDE = int(os.getenv("TRANSFORMER_STRIDE", "4"))

FEATURE_COLS_TS = [
    "t",
    "days_since_first_payment",
    "payment_amount",
    "cum_project_paid",
    "pct_project_paid",
    "project_value",
]

TARGET_COLS = ["weeks_late", "cost_overrun_percent"]


def parse_money(value):
    """Parse money strings like '$1,234.56' or '-$500.00' to float."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if s == "":
        return np.nan
    neg = s.startswith("-")
    s_clean = s.replace("-", "").replace("$", "").replace(",", "")
    try:
        val = float(s_clean)
    except ValueError:
        return np.nan
    return -val if neg else val


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        return self.fc_out(x)


def _load_model_and_norm() -> Optional[Dict[str, Any]]:
    if not TRANSFORMER_MODEL_PATH.exists() or not TRANSFORMER_NORM_PATH.exists():
        print("[transformer_head] Transformer model or norm stats not found. "
              "Set TRANSFORMER_MODEL_PATH / TRANSFORMER_NORM_PATH or place "
              ".pt and .npz files next to ml_core.py")
        return None

    norm = np.load(TRANSFORMER_NORM_PATH)

    feat_mean = norm["mean"]
    feat_std = norm["std"]
    target_mean = norm["target_mean"]
    target_std = norm["target_std"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PaymentTransformer(
        input_dim=len(FEATURE_COLS_TS),
        num_targets=len(TARGET_COLS),
    )
    model.load_state_dict(torch.load(TRANSFORMER_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    return {
        "model": model,
        "device": device,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "target_mean": target_mean,
        "target_std": target_std,
    }


# Lazy-loaded globals
_state: Optional[Dict[str, Any]] = None
TRANSFORMER_AVAILABLE = True


def _ensure_loaded():
    global _state, TRANSFORMER_AVAILABLE
    if _state is not None or not TRANSFORMER_AVAILABLE:
        return
    _state = _load_model_and_norm()
    if _state is None:
        TRANSFORMER_AVAILABLE = False


def build_timeseries_from_df(df_raw: pd.DataFrame, project_id: str) -> pd.DataFrame:
    """Build a time-series dataframe matching transformer_training_data.csv schema
    from a raw single-project payments dataframe (as used in ml_core.make_risk_summary_from_df).
    """
    df = df_raw.copy()

    if "payment_amount" not in df.columns or "payment_date" not in df.columns:
        return pd.DataFrame(columns=FEATURE_COLS_TS + ["project_id"])

    df["payment_amount_num"] = df["payment_amount"].apply(parse_money)
    df["project_value_num"] = df.get("project_value", np.nan).apply(parse_money)
    df["payment_date_parsed"] = pd.to_datetime(
        df["payment_date"], dayfirst=True, errors="coerce"
    )

    df = df[df["payment_amount_num"].notna()].copy()
    df = df[df["payment_date_parsed"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=FEATURE_COLS_TS + ["project_id"])

    # Assume single project in the CSV
    pv = df["project_value_num"].dropna()
    project_value = pv.iloc[0] if not pv.empty else np.nan

    df = df.sort_values("payment_date_parsed")

    first_date = df["payment_date_parsed"].iloc[0]
    df["days_since_first_payment"] = (
        df["payment_date_parsed"] - first_date
    ).dt.days

    df["t"] = np.arange(len(df), dtype=int)
    df["cum_project_paid"] = df["payment_amount_num"].cumsum()
    df["pct_project_paid"] = np.where(
        project_value > 0, df["cum_project_paid"] / project_value, 0.0
    )

    df["payment_amount"] = df["payment_amount_num"]
    df["project_value"] = project_value

    out = df[["t", "days_since_first_payment", "payment_amount",
              "cum_project_paid", "pct_project_paid"]].copy()
    out["project_value"] = project_value
    out["project_id"] = project_id

    return out


def predict_transformer_from_df(df_raw: pd.DataFrame, project_id: str) -> Optional[Dict[str, float]]:
    """Run the Transformer on a raw project dataframe and return
    project-level predictions for weeks_late and cost_overrun_percent.
    """
    _ensure_loaded()
    if not TRANSFORMER_AVAILABLE or _state is None:
        return None

    ts = build_timeseries_from_df(df_raw, project_id)
    if ts.empty or len(ts) < SEQ_LEN:
        print(f"[transformer_head] Not enough timesteps for project {project_id} "
              f"({len(ts)} < SEQ_LEN={SEQ_LEN}); skipping Transformer.")
        return None

    feat = ts[FEATURE_COLS_TS].values.astype("float32")

    sequences = []
    for start in range(0, len(ts) - SEQ_LEN + 1, STRIDE):
        end = start + SEQ_LEN
        sequences.append(feat[start:end])

    if not sequences:
        print(f"[transformer_head] No sequences built for project {project_id}; skipping Transformer.")
        return None

    X = np.stack(sequences, axis=0)  # (num_seqs, seq_len, num_features)

    feat_mean = _state["feat_mean"]
    feat_std = _state["feat_std"]
    target_mean = _state["target_mean"]
    target_std = _state["target_std"]
    model = _state["model"]
    device = _state["device"]

    X_norm = (X - feat_mean) / feat_std
    X_tensor = torch.from_numpy(X_norm).to(device)

    with torch.no_grad():
        preds_norm = model(X_tensor)

    preds_norm_np = preds_norm.cpu().numpy()
    preds_raw = preds_norm_np * target_std + target_mean

    weeks = preds_raw[:, 0]
    cost = preds_raw[:, 1]

    return {
        "weeks_late": float(weeks.mean()),
        "cost_overrun_percent": float(cost.mean()),
    }
