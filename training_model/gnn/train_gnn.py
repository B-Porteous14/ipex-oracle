"""
Training script for the ProjectRiskGNN model.

Uses the graph dataset built in gnn.data (one time–series graph per project)
and trains a regression model to predict:
    - weeks_late
    - cost_overrun_percent
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

from training_model.gnn.data import load_project_trade_dataset
from training_model.gnn.model import ProjectRiskGNN


def _ensure_tensor(target) -> torch.Tensor:
    """
    Ensure the target is a tensor, not a tuple/list.

    For our dataset, batch.y should already be a float32 tensor of shape
    [num_graphs, 2], but this helper is defensive in case the collate
    function wraps things oddly.
    """
    if isinstance(target, torch.Tensor):
        return target
    if isinstance(target, (tuple, list)):
        tensors = []
        for t in target:
            if isinstance(t, torch.Tensor):
                tensors.append(t)
            else:
                tensors.append(torch.tensor(t, dtype=torch.float32))
        return torch.stack(tensors, dim=0)
    # Fallback: single value
    return torch.tensor(target, dtype=torch.float32)


def train_gnn(
    num_epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
) -> None:
    print("[GNN] Starting training...")

    # 1) Load datasets
    train_ds, val_ds = load_project_trade_dataset()
    print(
        f"[GNN] Loaded {len(train_ds)} training graphs and "
        f"{len(val_ds)} validation graphs"
    )

    # 2) DataLoaders (use PyG DataLoader)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 3) Infer input feature dimension from a sample batch
    first_batch = next(iter(train_loader))
    in_channels = first_batch.x.size(-1)

    model = ProjectRiskGNN(in_channels=in_channels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[GNN] Training on {device} with {num_params} parameters")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 4) Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            out = model(batch)  # [num_graphs, 2]

            target = _ensure_tensor(batch.y).to(device)

            # ---- IMPORTANT: flatten both so shapes match [N*2] ----
            out_flat = out.view(-1)
            target_flat = target.view(-1)

            loss = criterion(out_flat, target_flat)

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * batch.num_graphs

        train_loss = running_train_loss / len(train_ds)

        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                target = _ensure_tensor(batch.y).to(device)

                out_flat = out.view(-1)
                target_flat = target.view(-1)

                loss = criterion(out_flat, target_flat)
                running_val_loss += loss.item() * batch.num_graphs

        val_loss = running_val_loss / len(val_ds)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
        )

    # 5) Save model
    save_path = Path(__file__).resolve().parent / "gnn_project_risk_model.pt"
    torch.save(model.state_dict(), save_path)
    print(f"[GNN] Saved model to: {save_path}")


if __name__ == "__main__":
    train_gnn()
