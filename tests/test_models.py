"""Unit tests for forecasting models (baselines, EdgeGridNet) and masked loss functions."""

import pytest
import torch
from src.graph.topology import WindTurbineGraph, build_static_graph
from src.models.baselines import TemporalGRU, StaticSTGCN
from src.models.edgegrid_net import EdgeGridNet
from src.training.loss import (
    masked_mae,
    masked_rmse,
    compute_kdd_cup_score,
    MaskedSmoothL1Loss,
)

LOCATIONS_CSV = "data/raw/sdwpf_baidukddcup2022_turb_location.csv"


@pytest.fixture(scope="module")
def graph() -> WindTurbineGraph:
    return build_static_graph(LOCATIONS_CSV, method="delaunay")


class TestLossAndMetrics:
    def test_masked_mae_and_rmse(self):
        y_pred = torch.tensor([[100.0, 200.0], [300.0, 400.0]])
        y_true = torch.tensor([[110.0, 200.0], [500.0, 400.0]])  # errors: 10, 0, 200, 0
        mask = torch.tensor([[True, True], [False, True]])  # element [1, 0] with error 200 is masked out!

        # Valid errors: 10, 0, 0 -> MAE = 10 / 3 = 3.3333
        mae = masked_mae(y_pred, y_true, mask)
        assert abs(mae.item() - 10.0 / 3.0) < 1e-4

        # Valid squared errors: 100, 0, 0 -> MSE = 100 / 3 -> RMSE = sqrt(33.333) = 5.7735
        rmse = masked_rmse(y_pred, y_true, mask)
        assert abs(rmse.item() - (100.0 / 3.0) ** 0.5) < 1e-4

        mae_val, rmse_val, combined = compute_kdd_cup_score(y_pred, y_true, mask)
        assert abs(combined - 0.5 * (mae_val + rmse_val)) < 1e-4

    def test_masked_smooth_l1_loss(self):
        loss_fn = MaskedSmoothL1Loss(beta=1.0)
        y_pred = torch.tensor([10.0, 20.0], requires_grad=True)
        y_true = torch.tensor([12.0, 25.0])
        mask = torch.tensor([True, False])

        loss = loss_fn(y_pred, y_true, mask)
        loss.backward()

        assert y_pred.grad is not None
        assert y_pred.grad[0] != 0.0
        assert y_pred.grad[1] == 0.0  # Masked element gets 0 gradient


class TestModelArchitectures:
    def test_temporal_gru_forward_backward(self):
        B, T_in, N, F_in = 2, 12, 134, 10
        out_len = 12

        model = TemporalGRU(
            in_features=F_in,
            hidden_dim=32,
            num_layers=1,
            out_len=out_len,
        )

        x = torch.randn(B, T_in, N, F_in, requires_grad=True)
        y_pred = model(x)

        assert y_pred.shape == (B, out_len, N)
        loss = y_pred.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_static_stgcn_forward_backward(self, graph: WindTurbineGraph):
        B, T_in, N, F_in = 2, 6, 134, 10
        out_len = 12

        model = StaticSTGCN(
            graph=graph,
            in_features=F_in,
            hidden_dim=32,
            out_len=out_len,
        )

        x = torch.randn(B, T_in, N, F_in, requires_grad=True)
        y_pred = model(x)

        assert y_pred.shape == (B, out_len, N)
        loss = y_pred.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_edgegrid_net_forward_backward(self, graph: WindTurbineGraph):
        B, T_in, N, F_in = 2, 6, 134, 10
        out_len = 12

        model = EdgeGridNet(
            graph=graph,
            in_features=F_in,
            hidden_dim=32,
            edge_dim=6,
            heads=2,
            temporal_layers=1,
            out_len=out_len,
        )

        x = torch.randn(B, T_in, N, F_in, requires_grad=True)
        wind_dir = torch.rand(B, T_in) * 360.0
        wind_spd = torch.rand(B, T_in) * 12.0

        y_pred = model(x, wind_dir=wind_dir, wind_spd=wind_spd)

        # Output shape check
        assert y_pred.shape == (B, out_len, N)
        # Power should be non-negative
        assert (y_pred >= 0.0).all()

        # Differentiable masked loss backward check
        loss_fn = MaskedSmoothL1Loss()
        dummy_y = torch.rand(B, out_len, N) * 500.0
        dummy_mask = torch.rand(B, out_len, N) > 0.2

        loss = loss_fn(y_pred, dummy_y, dummy_mask)
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        # Verify gradients exist in model parameters
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad
