"""
Graph Neural Network model for the IPEX Oracle.

Each project is represented as a time–series graph:

    - Nodes: time steps for the project
    - Node features: numeric features per time step
    - Edges: chain connections between consecutive time steps
    - Graph label: [weeks_late, cost_overrun_percent]

This module defines `ProjectRiskGNN`, which takes batched PyG `Data` objects
and returns a prediction per project.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch_geometric.nn import GraphConv, global_mean_pool
from torch_geometric.data import Data


class ProjectRiskGNN(nn.Module):
    """
    Simple GraphConv-based GNN for project risk prediction.

    Args:
        in_channels:  number of node feature dimensions
        hidden_channels: hidden layer width
        num_layers:   number of graph conv layers
        out_channels: output dimension (2: weeks_late, cost_overrun_percent)
        dropout:      dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        out_channels: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        convs = []
        # First layer: in_channels -> hidden_channels
        convs.append(GraphConv(in_channels, hidden_channels))

        # Middle layers: hidden -> hidden
        for _ in range(num_layers - 1):
            convs.append(GraphConv(hidden_channels, hidden_channels))

        self.convs = nn.ModuleList(convs)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        # MLP head on pooled graph embedding
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass.

        Args:
            data: a PyG Data object or a Batch with attributes:
                - x: node features [num_nodes, in_channels]
                - edge_index: [2, num_edges]
                - batch: graph indices [num_nodes] (added by DataLoader)

        Returns:
            Tensor of shape [num_graphs, out_channels]
        """
        x, edge_index = data.x, data.edge_index

        # When using DataLoader, `batch` is provided. For a single graph,
        # create a dummy batch of zeros.
        if hasattr(data, "batch"):
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # GraphConv stack
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.act(x)
            x = self.dropout(x)

        # Global pooling to get one embedding per graph
        x = global_mean_pool(x, batch)  # [num_graphs, hidden]

        # Prediction head
        out = self.head(x)  # [num_graphs, out_channels]
        return out
