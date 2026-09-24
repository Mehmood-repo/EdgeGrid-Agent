"""PyTorch Spatio-Temporal Dataset for Wind Turbine Networks.

Constructs multi-horizon sliding windows across irregular spatial networks
with strict temporal split boundaries, anomaly masking, and aerodynamic features.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.scaler import TabularFeatureScaler
from src.data.splits import TemporalSplitConfig

DEFAULT_FEATURE_COLS = [
    "Wspd",
    "Wspd_X",
    "Wspd_Y",
    "Ndir",
    "Pab1",
    "Pab2",
    "Pab3",
    "Etmp",
    "Itmp",
    "Patv",
]


class WindTurbineTemporalDataset(Dataset):
    """Spatio-Temporal sliding window dataset for multi-turbine power forecasting.

    Maintains a 3D tensor [T, N, F] in memory for O(1) sliding window slicing.
    """

    def __init__(
        self,
        data_path: Union[str, os.PathLike],
        split: str = "train",
        split_config: Optional[TemporalSplitConfig] = None,
        in_len: int = 144,
        out_len: int = 144,
        stride: int = 1,
        feature_cols: Optional[List[str]] = None,
        target_col: str = "Patv",
        scaler: Optional[TabularFeatureScaler] = None,
        scale_target: bool = False,
    ) -> None:
        """Initialize the sliding-window spatio-temporal dataset.

        Args:
            data_path: Path to cleaned parquet SCADA dataset.
            split: 'train', 'val', 'test', or 'all'.
            split_config: TemporalSplitConfig instance (default: 1-175 train, 176-205 val, 206-243 test).
            in_len: Number of input lookback timesteps (e.g. 144 steps = 24h at 10-min resolution).
            out_len: Number of forecast horizon timesteps (e.g. 144 steps = 24h).
            stride: Step stride between consecutive window origins (default: 1).
            feature_cols: Numeric feature column names to include in input tensor X.
            target_col: Prediction target column name (default: 'Patv').
            scaler: TabularFeatureScaler instance. If None and split=='train', fits a new scaler.
            scale_target: Whether y target is scaled or raw physical kW (default: False for direct kW loss).
        """
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.split_config = split_config or TemporalSplitConfig()
        self.in_len = in_len
        self.out_len = out_len
        self.stride = stride
        self.feature_cols = feature_cols or DEFAULT_FEATURE_COLS
        self.target_col = target_col
        self.scale_target = scale_target

        # 1. Load and filter split
        df = self._load_data()

        # 2. Extract ordered metadata
        self.turb_ids = np.sort(df["TurbID"].unique())
        self.num_nodes = len(self.turb_ids)
        self.id_to_idx = {turb_id: idx for idx, turb_id in enumerate(self.turb_ids)}

        # 3. Reshape into structured 3D arrays: [T, N, F]
        (
            raw_features,
            raw_target,
            mask_arr,
            farm_wind_dir,
            farm_wind_spd,
            self.timesteps_meta,
        ) = self._build_tensor_grid(df)

        # 4. Handle Scaler
        if scaler is None:
            if self.split == "train":
                self.scaler = TabularFeatureScaler(feature_names=self.feature_cols, method="standard")
                self.scaler.fit(raw_features)
            else:
                raise ValueError("Must provide an existing fitted scaler for validation or test splits.")
        else:
            self.scaler = scaler

        scaled_features = self.scaler.transform(raw_features)

        # Convert to PyTorch tensors
        self.features_tensor = torch.from_numpy(scaled_features).float()
        if self.scale_target:
            scaled_target = self.scaler.transform(
                raw_target[:, :, None].repeat(len(self.feature_cols), axis=-1)
            )[:, :, self.feature_cols.index(self.target_col)]
            self.target_tensor = torch.from_numpy(scaled_target).float()
        else:
            self.target_tensor = torch.from_numpy(raw_target).float()

        self.mask_tensor = torch.from_numpy(mask_arr).bool()
        self.farm_wdir_tensor = torch.from_numpy(farm_wind_dir).float()
        self.farm_wspd_tensor = torch.from_numpy(farm_wind_spd).float()

        # 5. Build valid window index list
        total_timesteps = self.features_tensor.shape[0]
        window_size = self.in_len + self.out_len
        if total_timesteps < window_size:
            raise ValueError(
                f"Split '{split}' has {total_timesteps} timesteps, which is smaller than "
                f"window size {window_size} (in_len={in_len} + out_len={out_len})."
            )

        self.window_indices = list(range(0, total_timesteps - window_size + 1, self.stride))

    def _load_data(self) -> pd.DataFrame:
        """Loads parquet file and filters to split day ranges."""
        columns_to_load = list(
            set(self.feature_cols + [self.target_col, "TurbID", "Day", "Tmstamp", "is_anomaly", "Wdir", "Ndir"])
        )
        df = pd.read_parquet(self.data_path, columns=columns_to_load)

        if self.split != "all":
            min_day, max_day = self.split_config.get_day_range(self.split)
            df = df[(df["Day"] >= min_day) & (df["Day"] <= max_day)].copy()

        return df

    def _build_tensor_grid(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, str]]]:
        """Pivots tabular SCADA records into aligned multidimensional arrays."""
        # Sort by Day, Tmstamp, TurbID
        df = df.sort_values(by=["Day", "Tmstamp", "TurbID"]).reset_index(drop=True)

        # Unique time coordinates
        time_index = df[["Day", "Tmstamp"]].drop_duplicates().sort_values(by=["Day", "Tmstamp"])
        timesteps_meta = list(zip(time_index["Day"].values, time_index["Tmstamp"].values))
        num_timesteps = len(timesteps_meta)

        # Pivot features
        p_target = df.pivot(index=["Day", "Tmstamp"], columns="TurbID", values=self.target_col).values
        p_anomaly = df.pivot(index=["Day", "Tmstamp"], columns="TurbID", values="is_anomaly").values
        valid_mask = ~p_anomaly

        # Farm-wide wind direction and wind speed
        p_ndir = df.pivot(index=["Day", "Tmstamp"], columns="TurbID", values="Ndir").values
        p_wdir = df.pivot(index=["Day", "Tmstamp"], columns="TurbID", values="Wdir").values
        p_wspd = df.pivot(index=["Day", "Tmstamp"], columns="TurbID", values="Wspd").values

        # True wind direction azimuth = (Ndir + Wdir) % 360
        true_wind_rad = np.radians(((p_ndir % 360) + p_wdir) % 360)
        mean_sin = np.nanmean(np.sin(true_wind_rad), axis=1)
        mean_cos = np.nanmean(np.cos(true_wind_rad), axis=1)
        farm_wdir = (np.degrees(np.arctan2(mean_sin, mean_cos)) % 360).astype(np.float32)
        farm_wspd = np.nanmean(p_wspd, axis=1).astype(np.float32)

        feature_arrays = []
        for col in self.feature_cols:
            p_feat = df.pivot(index=["Day", "Tmstamp"], columns="TurbID", values=col).values
            feature_arrays.append(p_feat)

        # [T, N, F]
        features_tensor = np.stack(feature_arrays, axis=-1).astype(np.float32)
        target_tensor = p_target.astype(np.float32)
        mask_tensor = valid_mask.astype(bool)

        return (
            features_tensor,
            target_tensor,
            mask_tensor,
            farm_wdir,
            farm_wspd,
            timesteps_meta,
        )

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Extracts a single sliding window sample.

        Returns:
            dict containing:
                'x': [in_len, num_nodes, num_features] historical SCADA tensor.
                'y': [out_len, num_nodes] target power tensor.
                'mask': [out_len, num_nodes] evaluation valid mask (True = valid, False = anomaly).
                'wind_dir_in': [in_len] historical wind direction.
                'wind_spd_in': [in_len] historical wind speed.
                'wind_dir_out': [out_len] target period wind direction.
                'wind_spd_out': [out_len] target period wind speed.
        """
        t_start = self.window_indices[idx]
        t_in_end = t_start + self.in_len
        t_out_end = t_in_end + self.out_len

        return {
            "x": self.features_tensor[t_start:t_in_end],
            "y": self.target_tensor[t_in_end:t_out_end],
            "mask": self.mask_tensor[t_in_end:t_out_end],
            "wind_dir_in": self.farm_wdir_tensor[t_start:t_in_end],
            "wind_spd_in": self.farm_wspd_tensor[t_start:t_in_end],
            "wind_dir_out": self.farm_wdir_tensor[t_in_end:t_out_end],
            "wind_spd_out": self.farm_wspd_tensor[t_in_end:t_out_end],
        }


def build_dataloaders(
    data_path: str = "data/processed/sdwpf_cleaned_243days.parquet",
    batch_size: int = 16,
    in_len: int = 144,
    out_len: int = 144,
    train_stride: int = 6,
    eval_stride: int = 72,
    num_workers: int = 0,
    feature_cols: Optional[List[str]] = None,
    scale_target: bool = False,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, TabularFeatureScaler]:
    """Factory helper to build synchronized Train, Validation, and Test PyTorch DataLoaders."""
    split_config = TemporalSplitConfig()

    # 1. Train dataset & fitted scaler
    train_ds = WindTurbineTemporalDataset(
        data_path=data_path,
        split="train",
        split_config=split_config,
        in_len=in_len,
        out_len=out_len,
        stride=train_stride,
        feature_cols=feature_cols,
        scale_target=scale_target,
    )

    # 2. Validation dataset (using train scaler)
    val_ds = WindTurbineTemporalDataset(
        data_path=data_path,
        split="val",
        split_config=split_config,
        in_len=in_len,
        out_len=out_len,
        stride=eval_stride,
        feature_cols=feature_cols,
        scaler=train_ds.scaler,
        scale_target=scale_target,
    )

    # 3. Test dataset (using train scaler)
    test_ds = WindTurbineTemporalDataset(
        data_path=data_path,
        split="test",
        split_config=split_config,
        in_len=in_len,
        out_len=out_len,
        stride=eval_stride,
        feature_cols=feature_cols,
        scaler=train_ds.scaler,
        scale_target=scale_target,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, train_ds.scaler
