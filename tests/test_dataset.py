"""Unit tests for spatio-temporal dataset, feature scaling, and data loaders."""

import pytest
import numpy as np
import torch
from src.data.scaler import TabularFeatureScaler
from src.data.splits import TemporalSplitConfig
from src.data.dataset import WindTurbineTemporalDataset, DEFAULT_FEATURE_COLS

PARQUET_PATH = "data/processed/sdwpf_cleaned_243days.parquet"


class TestFeatureScaler:
    def test_scaler_fit_transform_inverse(self):
        N_samples, N_nodes, N_feats = 100, 10, 4
        feature_names = ["f1", "f2", "f3", "Patv"]
        data = np.random.randn(N_samples, N_nodes, N_feats).astype(np.float32) * 5.0 + 10.0

        scaler = TabularFeatureScaler(feature_names=feature_names, method="standard")
        scaler.fit(data)

        assert scaler.is_fitted
        scaled = scaler.transform(data)
        assert scaled.shape == data.shape
        assert abs(scaled.mean()) < 0.05
        assert abs(scaled.std() - 1.0) < 0.05

        # Inverse transform target
        patv_data = data[..., 3]
        patv_scaled = scaled[..., 3]
        unscaled_patv = scaler.inverse_transform_target(patv_scaled, target_feature="Patv")
        np.testing.assert_allclose(unscaled_patv, patv_data, rtol=1e-5, atol=1e-5)


class TestTemporalSplits:
    def test_split_bounds_non_overlapping(self):
        cfg = TemporalSplitConfig(train_days=(1, 175), val_days=(176, 205), test_days=(206, 243))
        t_start, t_end = cfg.get_day_range("train")
        v_start, v_end = cfg.get_day_range("val")
        te_start, te_end = cfg.get_day_range("test")

        assert t_start == 1 and t_end == 175
        assert v_start == 176 and v_end == 205
        assert te_start == 206 and te_end == 243
        assert t_end < v_start
        assert v_end < te_start


class TestWindTurbineDataset:
    @pytest.fixture(scope="class")
    def dataset_sample(self):
        """Builds a test split dataset with in_len=24 and out_len=24 for fast testing."""
        cfg = TemporalSplitConfig(train_days=(1, 175), val_days=(176, 205), test_days=(206, 215))
        # Fit scaler on a tiny slice or initialize standard scaler
        scaler = TabularFeatureScaler(feature_names=DEFAULT_FEATURE_COLS, method="standard")
        # Dummy fit for test dataset fixture
        dummy_data = np.random.randn(10, 134, len(DEFAULT_FEATURE_COLS)).astype(np.float32)
        scaler.fit(dummy_data)

        ds = WindTurbineTemporalDataset(
            data_path=PARQUET_PATH,
            split="test",
            split_config=cfg,
            in_len=24,
            out_len=24,
            stride=12,
            scaler=scaler,
            scale_target=False,
        )
        return ds

    def test_dataset_item_shapes_and_types(self, dataset_sample: WindTurbineTemporalDataset):
        assert len(dataset_sample) > 0
        sample = dataset_sample[0]

        assert "x" in sample and "y" in sample and "mask" in sample
        assert "wind_dir_in" in sample and "wind_spd_in" in sample

        # x: [in_len=24, N=134, F=10]
        assert sample["x"].shape == (24, 134, len(DEFAULT_FEATURE_COLS))
        # y: [out_len=24, N=134]
        assert sample["y"].shape == (24, 134)
        # mask: [out_len=24, N=134]
        assert sample["mask"].shape == (24, 134)
        assert sample["mask"].dtype == torch.bool

        # Check for NaNs
        assert not torch.isnan(sample["x"]).any()
        assert not torch.isnan(sample["y"]).any()

    def test_dataloader_batch_collation(self, dataset_sample: WindTurbineTemporalDataset):
        loader = torch.utils.data.DataLoader(dataset_sample, batch_size=4, shuffle=False)
        batch = next(iter(loader))

        assert batch["x"].shape == (4, 24, 134, len(DEFAULT_FEATURE_COLS))
        assert batch["y"].shape == (4, 24, 134)
        assert batch["mask"].shape == (4, 24, 134)
        assert batch["wind_dir_in"].shape == (4, 24)
        assert batch["wind_spd_in"].shape == (4, 24)
