"""Masked loss functions and competition evaluation metrics matching Baidu KDD Cup rules."""

from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mae(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Computes Masked Mean Absolute Error (MAE) excluding anomaly timesteps.

    Args:
        y_pred: Predicted values [..., N].
        y_true: Ground truth values [..., N].
        mask: Boolean evaluation mask (True = valid, False = anomaly).
        eps: Small epsilon to prevent zero division.

    Returns:
        Scalar MAE loss tensor.
    """
    mask_float = mask.float()
    abs_err = torch.abs(y_pred - y_true) * mask_float
    return abs_err.sum() / (mask_float.sum() + eps)


def masked_rmse(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Computes Masked Root Mean Squared Error (RMSE) excluding anomaly timesteps."""
    mask_float = mask.float()
    sq_err = ((y_pred - y_true) ** 2) * mask_float
    mse = sq_err.sum() / (mask_float.sum() + eps)
    return torch.sqrt(mse + eps)


def compute_kdd_cup_score(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[float, float, float]:
    """Computes official competition metrics: (MAE, RMSE, CombinedScore).

    CombinedScore = 0.5 * (MAE + RMSE).
    """
    mae_val = masked_mae(y_pred, y_true, mask).item()
    rmse_val = masked_rmse(y_pred, y_true, mask).item()
    combined = 0.5 * (mae_val + rmse_val)
    return mae_val, rmse_val, combined


class MaskedSmoothL1Loss(nn.Module):
    """Differentiable masked Smooth L1 (Huber) loss for robust neural network optimization."""

    def __init__(self, beta: float = 10.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.beta = beta
        self.eps = eps

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_float = mask.float()
        loss = F.smooth_l1_loss(y_pred, y_true, beta=self.beta, reduction="none")
        masked_loss = loss * mask_float
        return masked_loss.sum() / (mask_float.sum() + self.eps)
