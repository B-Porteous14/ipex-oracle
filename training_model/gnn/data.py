"""
GNN data utilities for the IPEX Oracle.

This module builds a graph dataset from the same file used by the Transformer:
    Transformer/transformer_training_data.csv

That file has columns like:
    project_id
    t
    days_since_first_payment
    payment_amount
    cum_project_paid
    pct_project_paid
    project_value
    weeks_late
    cost_overrun_percent

We treat each project as a graph:
    - Nodes = time steps (rows for that project)
    - Node features = numeric time-series columns
    - Edges = chain of time steps (t -> t+1 and optionally t+1 -> t)
    - Graph label y = [weeks_late, cost_overrun_percent] for that project
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


# ---------- Helpers to load the raw CSV ----------


def _get_transformer_csv_path() -> Path:
    """
    Resolve the path to Transformer/transformer_training_data.csv
    starting from this file's location.

    This assumes the repo structure:
        Project Panther/
            Transformer/transformer_training_data.csv
            training_model/gnn/data.py  (this file)
    """
    # .../Project Panther/training_model/gnn/data.py  -> parents[2] = Project Panther
    root = Path(__file__).resolve().parents[2]
    csv_path = root / "Transformer" / "transformer_training_data.csv"
    return csv_path


def _load_raw_transformer_df() -> pd.DataFrame:
    csv_path = _get_transformer_csv_path()
    if not csv_path.exists():
        raise FileNotFoundError(
            f"[GNN] Could not find transformer_training_data.csv at {csv_path}"
        )

    df = pd.read_csv(csv_path)

    required_cols = [
        "project_id",
        "t",
        "days_since_first_payment",
        "payment_amount",
        "cum_project_paid",
        "pct_project_paid",
        "project_value",
        "weeks_late",
        "cost_overrun_percent",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[GNN] transformer_training_data.csv is missing columns: {missing}")

    return df


# ---------- Dataset: one graph per project ----------


class ProjectTimeSeriesGraphDataset(Dataset):
    """
    PyTorch Dataset where each item is a torch_geometric.data.Data object
    representing one project as a time-series graph.

    Nodes = time steps for that project.
    Node features = selected numeric columns.
    Edges = chain edges between consecutive time steps.
    Labels y = [weeks_late, cost_overrun_percent] for the project.
    """

    def __init__(self, graphs: List[Data]):
        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        return self.graphs[idx]


def _build_graphs_from_df(df: pd.DataFrame) -> List[Data]:
    """
    Convert the raw transformer dataframe into a list of Data graphs,
    one per project_id.
    """
    graphs: List[Data] = []

    # Group rows by project
    for project_id, proj_df in df.groupby("project_id"):
        proj_df = proj_df.sort_values("t")

        # Node features: choose a simple, consistent set
        feature_cols = [
            "days_since_first_payment",
            "payment_amount",
            "cum_project_paid",
            "pct_project_paid",
            "project_value",
        ]
        x = torch.tensor(proj_df[feature_cols].values, dtype=torch.float32)

        num_nodes = x.shape[0]

        # Chain edges: t -> t+1 and t+1 -> t
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

        # Labels (same for all rows in the project)
        weeks_late = float(proj_df["weeks_late"].iloc[0])
        cost_overrun = float(proj_df["cost_overrun_percent"].iloc[0])
        y = torch.tensor([weeks_late, cost_overrun], dtype=torch.float32)

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            project_id=str(project_id),
        )
        graphs.append(data)

    return graphs


# ---------- Public API: train/val splits ----------


def load_project_trade_dataset(
    train_ratio: float = 0.8, random_seed: int = 42
) -> Tuple[ProjectTimeSeriesGraphDataset, ProjectTimeSeriesGraphDataset]:
    """
    Load the transformer training CSV and return (train_ds, val_ds)
    as graph datasets.

    This is the function expected by training_model.gnn.train_gnn.
    """
    df = _load_raw_transformer_df()

    print(
        f"[GNN] Loaded transformer_training_data.csv: "
        f"{len(df)} rows, {df['project_id'].nunique()} projects."
    )

    graphs = _build_graphs_from_df(df)

    # Build a stable train/val split by project
    project_ids = sorted(df["project_id"].unique())
    rng = torch.Generator().manual_seed(random_seed)
    perm = torch.randperm(len(project_ids), generator=rng).tolist()
    split_idx = int(len(project_ids) * train_ratio)

    train_graphs: List[Data] = []
    val_graphs: List[Data] = []

    id_to_graphs = {}
    for g in graphs:
        pid = g.project_id
        id_to_graphs.setdefault(pid, []).append(g)

    for i, pid_idx in enumerate(perm):
        pid = project_ids[pid_idx]
        pid_graphs = id_to_graphs[str(pid)]
        # we only have one graph per project, but keep this general
        if i < split_idx:
            train_graphs.extend(pid_graphs)
        else:
            val_graphs.extend(pid_graphs)

    train_ds = ProjectTimeSeriesGraphDataset(train_graphs)
    val_ds = ProjectTimeSeriesGraphDataset(val_graphs)

    print(
        f"[GNN] Built graph dataset: {len(train_ds)} train graphs, "
        f"{len(val_ds)} val graphs."
    )

    return train_ds, val_ds
