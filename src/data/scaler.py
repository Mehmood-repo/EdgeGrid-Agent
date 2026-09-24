"""Feature normalization utilities fitted strictly on training data."""

from __future__ import annotations

import pickle
from typing import Dict, List, Optional, Union
import numpy as np
import torch


class TabularFeatureScaler:
    """Standard (Z-score) or MinMax scaler supporting 2D, 3D, and 4D tensor shapes

    [..., F] across multi-turbine spatial temporal arrays.
    """

    def __init__(
        self,
        feature_names: List[str],
        method: str = "standard",
        eps: float = 1e-6,
    ) -> None:
        self.feature_names = feature_names
        self.method = method
        self.eps = eps
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.min: Optional[np.ndarray] = None
        self.max: Optional[np.ndarray] = None
        self.is_fitted = False

    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> TabularFeatureScaler:
        """Computes mean and std (or min and max) across all leading dimensions.

        Args:
            data: Array of shape [..., num_features].
        """
        if isinstance(data, torch.Tensor):
            arr = data.detach().cpu().numpy()
        else:
            arr = np.asarray(data)

        num_features = len(self.feature_names)
        reshaped = arr.reshape(-1, num_features)

        if self.method == "standard":
            self.mean = np.nanmean(reshaped, axis=0).astype(np.float32)
            self.std = np.nanstd(reshaped, axis=0).astype(np.float32)
            # Avoid divide-by-zero on invariant features
            self.std[self.std < self.eps] = 1.0
        elif self.method == "minmax":
            self.min = np.nanmin(reshaped, axis=0).astype(np.float32)
            self.max = np.nanmax(reshaped, axis=0).astype(np.float32)
            denom = self.max - self.min
            denom[denom < self.eps] = 1.0
            self.std = denom
            self.mean = self.min
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")

        self.is_fitted = True
        return self

    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Applies normalization."""
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before calling transform.")

        is_torch = isinstance(data, torch.Tensor)
        device = data.device if is_torch else None
        dtype = data.dtype if is_torch else None

        arr = data.detach().cpu().numpy() if is_torch else np.asarray(data)

        if self.method in ("standard", "minmax"):
            scaled = (arr - self.mean) / self.std
        else:
            scaled = arr

        if is_torch:
            return torch.from_numpy(scaled).to(device=device, dtype=dtype)
        return scaled

    def inverse_transform_target(
        self, target_data: Union[np.ndarray, torch.Tensor], target_feature: str = "Patv"
    ) -> Union[np.ndarray, torch.Tensor]:
        """Inverts scaling for a single target feature (e.g. Patv)."""
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before inverse transform.")

        target_idx = self.feature_names.index(target_feature)
        mean_val = float(self.mean[target_idx])
        std_val = float(self.std[target_idx])

        is_torch = isinstance(target_data, torch.Tensor)
        device = target_data.device if is_torch else None
        dtype = target_data.dtype if is_torch else None

        arr = target_data.detach().cpu().numpy() if is_torch else np.asarray(target_data)
        unscaled = (arr * std_val) + mean_val

        if is_torch:
            return torch.from_numpy(unscaled).to(device=device, dtype=dtype)
        return unscaled

    def save(self, filepath: str) -> None:
        """Serializes scaler parameters."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> TabularFeatureScaler:
        """Loads serialized scaler parameters."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
