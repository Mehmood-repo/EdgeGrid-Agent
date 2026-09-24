"""EdgeGridNet: Physics- and Wake-Informed Spatio-Temporal Graph Neural Network.

Integrates dynamic wind-directed graph attention with multi-scale temporal convolutions
for industrial wind turbine power forecasting.
"""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

from src.graph.topology import WindTurbineGraph
from src.graph.dynamic_weights import DynamicGraphWeighter


class WakeConditionedSpatialLayer(nn.Module):
    """Spatial Graph Attention layer conditioned on aerodynamic edge features

    (distance, wind-angle alignment, downwind indicator, and propagation delay).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int = 6,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv = TransformerConv(
            in_channels=in_dim,
            out_channels=out_dim // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            beta=True,
        )
        self.norm = nn.LayerNorm(out_dim)
        self.proj = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features [N_total, in_dim].
            edge_index: Edge indices [2, E_total].
            edge_attr: Aerodynamic edge features [E_total, edge_dim].

        Returns:
            out: Updated node representations [N_total, out_dim].
        """
        residual = x
        h = self.conv(x, edge_index, edge_attr=edge_attr)
        h = self.norm(h + residual)
        h = self.proj(F.gelu(h))
        return self.dropout(h)


class EdgeGridNet(nn.Module):
    """EdgeGrid Spatio-Temporal Graph Neural Network with dynamic aerodynamic wake awareness.

    Pipeline:
    1. Node feature embedding (F -> hidden_dim).
    2. Dynamic edge feature extraction via DynamicGraphWeighter.
    3. Wake-Conditioned Spatial Graph Attention across turbine network.
    4. Temporal Gated Recurrent aggregation across time horizon.
    5. Multi-horizon projection head predicting Patv for each turbine.
    """

    def __init__(
        self,
        graph: WindTurbineGraph,
        in_features: int = 10,
        hidden_dim: int = 64,
        edge_dim: int = 6,
        heads: int = 4,
        temporal_layers: int = 2,
        out_len: int = 144,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_nodes = graph.num_nodes
        self.hidden_dim = hidden_dim
        self.out_len = out_len
        self.edge_dim = edge_dim

        # Dynamic graph weighter module
        self.weighter = DynamicGraphWeighter(
            graph=graph,
            base_epsilon=0.1,
            dist_scale=1000.0,
            rated_speed=12.0,
        )

        # Input feature projection
        self.node_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Wake-conditioned spatial attention layer
        self.spatial_layer = WakeConditionedSpatialLayer(
            in_dim=hidden_dim,
            out_dim=hidden_dim,
            edge_dim=edge_dim,
            heads=heads,
            dropout=dropout,
        )

        # Temporal Recurrent backbone
        self.temporal_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=temporal_layers,
            batch_first=True,
            dropout=dropout if temporal_layers > 1 else 0.0,
        )

        # Multi-horizon forecast head with residual skip from most recent state
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, out_len),
        )

    def forward(
        self,
        x: torch.Tensor,
        wind_dir: Optional[torch.Tensor] = None,
        wind_spd: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, T_in, N, F].
            wind_dir: Historical wind direction in degrees [B, T_in].
                      If None, uses 0.0 default.
            wind_spd: Historical wind speed in m/s [B, T_in].
                      If None, uses 0.0 default.

        Returns:
            y_pred: Forecasted active power [B, T_out, N].
        """
        B, T_in, N, num_feats = x.shape
        device = x.device

        if wind_dir is None:
            wind_dir = torch.zeros((B, T_in), device=device)
        if wind_spd is None:
            wind_spd = torch.full((B, T_in), 5.0, device=device)

        # 1. Project raw features: [B, T_in, N, hidden_dim]
        h = self.node_proj(x)

        # 2. Compute dynamic edge features across the temporal sequence:
        # [B, T_in, num_edges, edge_dim]
        _, edge_attrs = self.weighter(wind_dir, wind_spd)
        num_edges = self.weighter.edge_index.shape[1]

        # 3. Spatial message passing at representative temporal anchors or steps
        # To balance expressiveness and throughput, apply wake attention across timesteps:
        # Flatten (B * T_in, N, hidden_dim) and construct batched graph
        h_flat = h.view(B * T_in * N, self.hidden_dim)
        edge_attrs_flat = edge_attrs.view(B * T_in * num_edges, self.edge_dim)

        # Build batched edge index for B * T_in graphs
        base_edge_index = self.weighter.edge_index  # [2, num_edges]
        offsets = torch.arange(B * T_in, device=device) * N  # [B * T_in]
        batched_src = (base_edge_index[0].unsqueeze(0) + offsets.unsqueeze(1)).view(-1)
        batched_dst = (base_edge_index[1].unsqueeze(0) + offsets.unsqueeze(1)).view(-1)
        batched_edge_index = torch.stack([batched_src, batched_dst], dim=0)

        # Wake-conditioned spatial message passing
        h_spatial = self.spatial_layer(h_flat, batched_edge_index, edge_attrs_flat)

        # 4. Reshape for Temporal GRU: [B * N, T_in, hidden_dim]
        h_spatial = h_spatial.view(B, T_in, N, self.hidden_dim)
        h_temp = h_spatial.permute(0, 2, 1, 3).reshape(B * N, T_in, self.hidden_dim)

        gru_out, _ = self.temporal_gru(h_temp)
        final_state = gru_out[:, -1, :]  # [B * N, hidden_dim]

        # 5. Multi-horizon projection
        y_flat = self.head(final_state)  # [B * N, out_len]

        # Reshape to [B, out_len, N]
        y_pred = y_flat.view(B, N, self.out_len).permute(0, 2, 1)

        # Ensure power output cannot be physically negative
        y_pred = torch.relu(y_pred)

        return y_pred
