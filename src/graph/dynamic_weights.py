"""Dynamic Aerodynamic Edge Weighting and Feature Construction.

Computes time-varying, wind-direction-dependent edge weights and edge attributes
reflecting wake propagation, asymmetric spatial message passing, and angular wake alignment.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from src.graph.topology import WindTurbineGraph


def compute_dynamic_edge_weights(
    edge_index: torch.Tensor,
    edge_bearing: torch.Tensor,
    static_edge_weight: torch.Tensor,
    wind_direction: torch.Tensor,
    base_epsilon: float = 0.1,
    wake_cone_degrees: float = 60.0,
    downwind_exponent: float = 1.0,
) -> torch.Tensor:
    """Computes dynamic, wind-directed edge weights for a batch of timestamps.

    For an edge u -> v with compass bearing theta_uv and wind blowing FROM phi_t,
    the wind travels TOWARDS phi_towards = (phi_t + 180) % 360.
    The angular offset is Delta_theta = theta_uv - phi_towards.

    Args:
        edge_index: Graph edge connectivity [2, num_edges].
        edge_bearing: Edge compass bearing in degrees [num_edges] (0°=N, 90°=E).
        static_edge_weight: Static distance-decay weights [num_edges].
        wind_direction: Instantaneous wind direction in degrees FROM which wind blows.
                        Can be scalar (), 1D [T], 2D [B, T], or per-node [B, T, N].
        base_epsilon: Minimum baseline connection weight for cross-wind/upwind edges.
                      epsilon=0 enforces strict forward downwind message passing.
        wake_cone_degrees: Half-angle aperture for directional wake envelope.
        downwind_exponent: Exponent for cosine sharpening.

    Returns:
        dynamic_weight: Tensor of dynamic edge weights.
                        Shape: [..., num_edges] matching the leading dimensions of wind_direction.
    """
    # Determine wind propagation direction (direction towards which air moves)
    # Wind blowing from phi moves towards (phi + 180) % 360
    phi_towards = (wind_direction + 180.0) % 360.0

    # Handle per-node wind direction [..., N] vs farm-level wind direction [...]
    if wind_direction.ndim >= 1 and wind_direction.shape[-1] == int(edge_index.max().item() + 1):
        # Source node determines the local outgoing wake vector
        src_nodes = edge_index[0]
        # Gather source node wind direction
        phi_towards = phi_towards[..., src_nodes]  # [..., num_edges]
    else:
        # Broadcast farm-level wind direction across edges
        phi_towards = phi_towards.unsqueeze(-1)  # [..., 1]

    # Convert angles to radians
    bearing_rad = torch.deg2rad(edge_bearing)  # [num_edges]
    phi_rad = torch.deg2rad(phi_towards)  # [..., 1] or [..., num_edges]

    # Angular difference: Delta = theta_uv - phi_towards
    delta_angle = bearing_rad - phi_rad

    # Directional alignment: cos(delta_angle) is +1 for exact downwind, -1 for exact upwind
    cos_delta = torch.cos(delta_angle)

    # Rectified directional factor: max(0, cos(delta))
    directional_factor = torch.clamp(cos_delta, min=0.0) ** downwind_exponent

    # Dynamic asymmetric weight: static_weight * [epsilon + (1 - epsilon) * directional_factor]
    modulation = base_epsilon + (1.0 - base_epsilon) * directional_factor
    dynamic_weight = static_edge_weight * modulation

    return dynamic_weight


def compute_dynamic_edge_features(
    edge_index: torch.Tensor,
    edge_dist: torch.Tensor,
    edge_bearing: torch.Tensor,
    wind_direction: torch.Tensor,
    wind_speed: Optional[torch.Tensor] = None,
    dist_scale: float = 1000.0,
    rated_speed: float = 12.0,
) -> torch.Tensor:
    """Constructs multi-dimensional continuous edge feature vectors for GNN layers.

    Feature dimensions:
    - e[:, 0]: Normalized physical distance d_ij / dist_scale
    - e[:, 1]: cos(theta_ij - phi_towards) -> downwind alignment [-1, 1]
    - e[:, 2]: sin(theta_ij - phi_towards) -> cross-wind deviation [-1, 1]
    - e[:, 3]: Binary downwind indicator: I(cos(Delta_theta) > 0)
    - Optional e[:, 4]: Estimated aerodynamic propagation delay d_ij / (v_wind + eps)
    - Optional e[:, 5]: Normalized wind speed v_wind / rated_speed

    Args:
        edge_index: Graph edge connectivity [2, num_edges].
        edge_dist: Physical Euclidean edge distance in meters [num_edges].
        edge_bearing: Edge compass bearing in degrees [num_edges].
        wind_direction: Instantaneous wind direction [..., 1] or [...] in degrees.
        wind_speed: Instantaneous wind speed in m/s (optional).
        dist_scale: Scaling divisor for Euclidean distance (e.g. 1000m).
        rated_speed: Reference rated wind speed for normalization (e.g. 12 m/s).

    Returns:
        edge_features: Tensor of shape [..., num_edges, feature_dim].
    """
    phi_towards = (wind_direction + 180.0) % 360.0

    if wind_direction.ndim >= 1 and wind_direction.shape[-1] == int(edge_index.max().item() + 1):
        src_nodes = edge_index[0]
        phi_towards = phi_towards[..., src_nodes]
    else:
        phi_towards = phi_towards.unsqueeze(-1)

    bearing_rad = torch.deg2rad(edge_bearing)
    phi_rad = torch.deg2rad(phi_towards)
    delta_angle = bearing_rad - phi_rad

    cos_delta = torch.cos(delta_angle)
    sin_delta = torch.sin(delta_angle)
    is_downwind = (cos_delta > 0.0).float()

    # Expand edge_dist to match batch dimensions
    d_norm = (edge_dist / dist_scale).expand_as(cos_delta)

    feature_list = [d_norm, cos_delta, sin_delta, is_downwind]

    if wind_speed is not None:
        if wind_speed.ndim >= 1 and wind_speed.shape[-1] == int(edge_index.max().item() + 1):
            src_nodes = edge_index[0]
            ws_edge = wind_speed[..., src_nodes]
        else:
            ws_edge = wind_speed.unsqueeze(-1).expand_as(cos_delta)

        ws_norm = torch.clamp(ws_edge / rated_speed, 0.0, 2.0)
        # Propagation delay (seconds) = distance / speed
        prop_delay = (edge_dist / torch.clamp(ws_edge, min=1.0)).expand_as(cos_delta)
        prop_delay_norm = prop_delay / 300.0  # normalized by 5 minutes (300s)

        feature_list.extend([ws_norm, prop_delay_norm])

    return torch.stack(feature_list, dim=-1)


class DynamicGraphWeighter(nn.Module):
    """PyTorch Module that maintains the turbine graph and generates dynamic edge

    weights and edge feature embeddings during the forward pass.
    """

    def __init__(
        self,
        graph: WindTurbineGraph,
        base_epsilon: float = 0.1,
        dist_scale: float = 1000.0,
        rated_speed: float = 12.0,
        downwind_exponent: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_nodes = graph.num_nodes
        self.num_edges = graph.edge_index.shape[1]
        self.base_epsilon = base_epsilon
        self.dist_scale = dist_scale
        self.rated_speed = rated_speed
        self.downwind_exponent = downwind_exponent

        # Register fixed geometric buffers
        self.register_buffer("edge_index", graph.edge_index)
        self.register_buffer("edge_dist", graph.edge_dist)
        self.register_buffer("edge_bearing", graph.edge_bearing)
        self.register_buffer("static_edge_weight", graph.edge_weight)
        self.register_buffer("pos", graph.pos)

    def forward(
        self,
        wind_direction: torch.Tensor,
        wind_speed: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates dynamic edge weights and multi-dimensional edge features.

        Args:
            wind_direction: Wind direction in degrees [B, T] or [T] or [B, T, N].
            wind_speed: Optional wind speed in m/s matching wind_direction shape.

        Returns:
            dynamic_weights: Dynamic weights [..., num_edges].
            edge_features: Dynamic edge attributes [..., num_edges, feature_dim].
        """
        weights = compute_dynamic_edge_weights(
            edge_index=self.edge_index,
            edge_bearing=self.edge_bearing,
            static_edge_weight=self.static_edge_weight,
            wind_direction=wind_direction,
            base_epsilon=self.base_epsilon,
            downwind_exponent=self.downwind_exponent,
        )

        features = compute_dynamic_edge_features(
            edge_index=self.edge_index,
            edge_dist=self.edge_dist,
            edge_bearing=self.edge_bearing,
            wind_direction=wind_direction,
            wind_speed=wind_speed,
            dist_scale=self.dist_scale,
            rated_speed=self.rated_speed,
        )

        return weights, features
