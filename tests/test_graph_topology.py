"""Unit tests for spatial graph topology and dynamic aerodynamic weighting."""

import os
import pytest
import numpy as np
import torch
from src.graph.topology import WindTurbineGraph, build_static_graph
from src.graph.dynamic_weights import (
    compute_dynamic_edge_weights,
    compute_dynamic_edge_features,
    DynamicGraphWeighter,
)

LOCATIONS_CSV = "data/raw/sdwpf_baidukddcup2022_turb_location.csv"


@pytest.fixture
def real_graph() -> WindTurbineGraph:
    """Fixture providing a Delaunay graph built from actual farm coordinates."""
    return build_static_graph(LOCATIONS_CSV, method="delaunay")


class TestGraphTopology:
    def test_delaunay_construction(self, real_graph: WindTurbineGraph):
        assert real_graph.num_nodes == 134
        # Delaunay triangulation on 134 2D points produces 387 undirected edges = 774 directed edges
        assert real_graph.edge_index.shape[0] == 2
        assert real_graph.edge_index.shape[1] == 774

        # Verify distance sanity
        min_dist = real_graph.edge_dist.min().item()
        median_dist = real_graph.edge_dist.median().item()
        assert 400.0 < min_dist < 420.0
        assert 950.0 < median_dist < 1050.0

        # Verify weights in (0, 1]
        assert (real_graph.edge_weight > 0.0).all()
        assert (real_graph.edge_weight <= 1.0).all()

    def test_knn_construction(self):
        k = 4
        knn_graph = build_static_graph(LOCATIONS_CSV, method="knn", knn_k=k)
        assert knn_graph.num_nodes == 134
        # Each node has at least k directed outgoing edges
        out_degrees = torch.bincount(knn_graph.edge_index[0], minlength=134)
        assert (out_degrees >= k).all()

    def test_threshold_construction(self):
        threshold_m = 800.0
        thresh_graph = build_static_graph(
            LOCATIONS_CSV, method="threshold", dist_threshold=threshold_m
        )
        assert thresh_graph.num_nodes == 134
        assert (thresh_graph.edge_dist <= threshold_m).all()

    def test_pyg_data_conversion(self, real_graph: WindTurbineGraph):
        data = real_graph.get_pyg_data()
        assert data.num_nodes == 134
        assert data.edge_index.shape == (2, 774)
        assert data.edge_weight.shape == (774,)
        assert data.edge_attr.shape == (774, 2)
        assert data.pos.shape == (134, 2)

    def test_dense_adjacency(self, real_graph: WindTurbineGraph):
        adj = real_graph.get_dense_adjacency()
        assert adj.shape == (134, 134)
        # Diagonal should be zero (no self-loops in Delaunay)
        assert torch.diag(adj).sum().item() == 0.0
        assert (adj >= 0.0).all()


class TestDynamicWeights:
    def test_wake_asymmetry_colinear_pair(self, real_graph: WindTurbineGraph):
        """Tests that a South wind produces strong 1 -> 2 weight and attenuated 2 -> 1 weight."""
        # Find index of edge 1 -> 2 (indices 0 and 1 in 0-indexed layout)
        edge_idx = real_graph.edge_index
        src, dst = edge_idx[0], edge_idx[1]

        # Turbine 1 is (3349.85, 5939.23), Turbine 2 is (3351.00, 6416.65) -> T2 is North of T1
        mask_1_to_2 = (src == 0) & (dst == 1)
        mask_2_to_1 = (src == 1) & (dst == 0)

        assert mask_1_to_2.sum() == 1
        assert mask_2_to_1.sum() == 1

        idx_1_to_2 = torch.where(mask_1_to_2)[0].item()
        idx_2_to_1 = torch.where(mask_2_to_1)[0].item()

        bearing_1_to_2 = real_graph.edge_bearing[idx_1_to_2].item()
        bearing_2_to_1 = real_graph.edge_bearing[idx_2_to_1].item()

        # 1 -> 2 points North (~0°), 2 -> 1 points South (~180°)
        assert abs(bearing_1_to_2 - 0.1) < 2.0
        assert abs(bearing_2_to_1 - 180.1) < 2.0

        # Scenario A: Wind blowing FROM South (180°) -> blowing TOWARDS North (0°)
        wind_from_south = torch.tensor([180.0])
        weights_south = compute_dynamic_edge_weights(
            edge_index=real_graph.edge_index,
            edge_bearing=real_graph.edge_bearing,
            static_edge_weight=real_graph.edge_weight,
            wind_direction=wind_from_south,
            base_epsilon=0.1,
        )

        w_downwind = weights_south[0, idx_1_to_2].item()  # 1 -> 2 (Downwind)
        w_upwind = weights_south[0, idx_2_to_1].item()  # 2 -> 1 (Upwind)

        # Downwind edge should receive full weight (~1.0 * static), upwind should be base_epsilon (0.1 * static)
        static_w = real_graph.edge_weight[idx_1_to_2].item()
        assert w_downwind > 0.95 * static_w
        assert abs(w_upwind - 0.1 * static_w) < 0.05 * static_w

        # Scenario B: Wind blowing FROM North (0°) -> blowing TOWARDS South (180°)
        wind_from_north = torch.tensor([0.0])
        weights_north = compute_dynamic_edge_weights(
            edge_index=real_graph.edge_index,
            edge_bearing=real_graph.edge_bearing,
            static_edge_weight=real_graph.edge_weight,
            wind_direction=wind_from_north,
            base_epsilon=0.1,
        )

        w_downwind_north = weights_north[0, idx_2_to_1].item()  # 2 -> 1 is now downwind
        w_upwind_north = weights_north[0, idx_1_to_2].item()  # 1 -> 2 is now upwind

        assert w_downwind_north > 0.95 * static_w
        assert abs(w_upwind_north - 0.1 * static_w) < 0.05 * static_w

    def test_dynamic_features_batching(self, real_graph: WindTurbineGraph):
        B, T = 4, 24
        num_edges = real_graph.edge_index.shape[1]

        wind_dir = torch.rand(B, T) * 360.0
        wind_spd = torch.rand(B, T) * 15.0

        features = compute_dynamic_edge_features(
            edge_index=real_graph.edge_index,
            edge_dist=real_graph.edge_dist,
            edge_bearing=real_graph.edge_bearing,
            wind_direction=wind_dir,
            wind_speed=wind_spd,
        )

        # 6 features: [d_norm, cos_delta, sin_delta, is_downwind, ws_norm, prop_delay_norm]
        assert features.shape == (B, T, num_edges, 6)
        # cos and sin in [-1, 1]
        assert (features[..., 1] >= -1.01).all() and (features[..., 1] <= 1.01).all()
        assert (features[..., 2] >= -1.01).all() and (features[..., 2] <= 1.01).all()
        # is_downwind in {0, 1}
        assert ((features[..., 3] == 0.0) | (features[..., 3] == 1.0)).all()

    def test_dynamic_graph_weighter_module(self, real_graph: WindTurbineGraph):
        weighter = DynamicGraphWeighter(real_graph, base_epsilon=0.1)

        B, T = 2, 12
        wind_dir = torch.rand(B, T) * 360.0
        wind_spd = torch.rand(B, T) * 10.0

        weights, features = weighter(wind_dir, wind_spd)
        num_edges = real_graph.edge_index.shape[1]

        assert weights.shape == (B, T, num_edges)
        assert features.shape == (B, T, num_edges, 6)
        assert not torch.isnan(weights).any()
        assert not torch.isnan(features).any()
