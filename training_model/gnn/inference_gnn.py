"""
Inference helper for the ProjectRiskGNN model.

Given a single project's transaction dataframe (same schema as the
engineered dataframe used in ml_core), build a time–series graph
and run the trained GNN to predict:

    - weeks_late
    - cost_overrun_percent
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch_geometric.data import Data

from training_model.gnn.model import ProjectRiskGNN


# Must match the feature columns used in data.py / training
FEATURE_COLS: List[str] = [
    "days_since_first_payment",
    "payment_amount",
    "cum_project_paid",
    "pct_project_paid",
    "project_value",
]


def _build_graph_for_project(df_project: pd.DataFrame) -> Data:
    """
    Build a single Data graph for one project.

    Assumes df_project already has the FEATURE_COLS and either:
        - a column 't' (time index), or
        - can be sorted by 'payment_date'.
    """

    if not set(FEATURE_COLS).issubset(df_project.columns):
        missing = [c for c in FEATURE_COLS if c not in df_project.columns]
        raise ValueError(f"[GNN] Project dataframe missing columns: {missing}")

    proj_df = df_project.copy()

    if "t" in proj_df.columns:
        proj_df = proj_df.sort_values("t")
    elif "payment_date" in proj_df.columns:
        proj_df = proj_df.sort_values("payment_date")
    else:
        # fallback: just sort by index
        proj_df = proj_df.sort_index()

    x = torch.tensor(proj_df[FEATURE_COLS].values, dtype=torch.float32)
    num_nodes = x.shape[0]

    if num_nodes > 1:
        src = []
        dst = []
        for i in range(num_nodes - 1):
            src.append(i)
            dst.append(i + 1)
            src.append(i + 1)
            dst.append(i)
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    project_id = str(proj_df["project_id"].iloc[0]) if "project_id" in proj_df.columns else ""

    data = Data(
        x=x,
        edge_index=edge_index,
        project_id=project_id,
    )
    return data


def predict_risk_gnn(df_project: pd.DataFrame) -> Dict[str, float]:
    """
    Run the trained GNN for a single project dataframe.

    Returns a dict:
        {
            "weeks_late": float,
            "cost_overrun_percent": float,
        }
    """
    # Build graph
    graph = _build_graph_for_project(df_project)

    # Device & model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Same input feature dim as training
    in_channels = graph.x.size(-1)
    model = ProjectRiskGNN(in_channels=in_channels)

    model_path = Path(__file__).resolve().parent / "gnn_project_risk_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"[GNN] Model file not found at {model_path}")

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    graph = graph.to(device)

    with torch.no_grad():
        out = model(graph)  # shape [1, 2]
    out = out.squeeze(0).cpu().numpy()

    weeks_late_pred = float(out[0])
    cost_overrun_pred = float(out[1])

    return {
        "weeks_late": weeks_late_pred,
        "cost_overrun_percent": cost_overrun_pred,
    }
