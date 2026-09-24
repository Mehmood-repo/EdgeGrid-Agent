"""Baseline forecasting architectures for wind turbine networks.

Includes:
1. TemporalGRU: Node-independent recurrent temporal baseline.
2. StaticSTGCN: Spatio-temporal graph neural network with static distance adjacency.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from src.graph.topology import WindTurbineGraph


class TemporalGRU(nn.Module):
    """Node-independent multi-layer GRU baseline.

    Processes each wind turbine's SCADA time series independently without spatial message passing.
    """

    def __init__(
        self,
        in_features: int = 10,
        hidden_dim: int = 64,
        num_layers: int = 2,
        out_len: int = 144,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_len = out_len

        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, out_len),
        )

    def forward(
        self,
        x: torch.Tensor,
        wind_dir: torch.Tensor | None = None,
        wind_spd: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, T_in, N, F].
            wind_dir: Optional wind direction (unused by temporal-only baseline).
            wind_spd: Optional wind speed (unused by temporal-only baseline).

        Returns:
            y_pred: Forecasted active power [B, T_out, N].
        """
        B, T_in, N, F = x.shape

        # Reshape to [B * N, T_in, F]
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, T_in, F)
        h = self.input_proj(x_flat)  # [B * N, T_in, hidden_dim]

        out, _ = self.gru(h)  # [B * N, T_in, hidden_dim]
        last_hidden = out[:, -1, :]  # [B * N, hidden_dim]

        y_flat = self.head(last_hidden)  # [B * N, T_out]
        # Reshape back to [B, T_out, N]
        y_pred = y_flat.view(B, N, self.out_len).permute(0, 2, 1)
        return y_pred


class StaticSTGCN(nn.Module):
    """Spatio-Temporal Graph Convolutional Network using static distance adjacency.

    Interleaves static GCN spatial convolutions with temporal GRU layers.
    """

    def __init__(
        self,
        graph: WindTurbineGraph,
        in_features: int = 10,
        hidden_dim: int = 64,
        out_len: int = 144,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_nodes = graph.num_nodes
        self.hidden_dim = hidden_dim
        self.out_len = out_len

        # Register static graph buffers
        self.register_buffer("edge_index", graph.edge_index)
        self.register_buffer("edge_weight", graph.edge_weight)

        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        self.temporal_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, out_len),
        )

    def forward(
        self,
        x: torch.Tensor,
        wind_dir: torch.Tensor | None = None,
        wind_spd: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, T_in, N, F].

        Returns:
            y_pred: Forecasted active power [B, T_out, N].
        """
        B, T_in, N, F = x.shape

        h = self.input_proj(x)  # [B, T_in, N, hidden_dim]

        # Apply spatial GCN across nodes at each timestep or reshaped batch
        # Reshape [B * T_in, N, hidden_dim] -> [B * T_in * N, hidden_dim] for GCN
        h_spatial = h.view(B * T_in, N, self.hidden_dim)

        # Batch GCN over timesteps: we can iterate or use block-diagonal batching
        # For moderate B * T_in, vectorized graph conv:
        spatial_outs = []
        for t in range(T_in):
            # h[:, t]: [B, N, hidden_dim]
            ht = h[:, t]  # [B, N, hidden_dim]
            # Process each batch item or flatten
            ht_flat = ht.reshape(B * N, self.hidden_dim)
            # Expand edge_index for batch
            batch_edges = []
            for b in range(B):
                batch_edges.append(self.edge_index + b * N)
            batch_edge_index = torch.cat(batch_edges, dim=1)
            batch_edge_weight = self.edge_weight.repeat(B)

            s1 = torch.relu(self.gcn1(ht_flat, batch_edge_index, batch_edge_weight))
            s2 = self.gcn2(s1, batch_edge_index, batch_edge_weight)
            spatial_outs.append(s2.view(B, N, self.hidden_dim))

        # [B, T_in, N, hidden_dim]
        h_spatial = torch.stack(spatial_outs, dim=1)

        # Temporal GRU over time dimension per node
        h_temp_in = h_spatial.permute(0, 2, 1, 3).reshape(B * N, T_in, self.hidden_dim)
        gru_out, _ = self.temporal_gru(h_temp_in)
        last_hidden = gru_out[:, -1, :]  # [B * N, hidden_dim]

        y_flat = self.head(last_hidden)  # [B * N, T_out]
        y_pred = y_flat.view(B, N, self.out_len).permute(0, 2, 1)
        return y_pred
