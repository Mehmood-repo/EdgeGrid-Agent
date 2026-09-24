"""Unit tests for ModelTrainer training loops, validation, and checkpointing."""

import os
import shutil
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainer import ModelTrainer


class DummyForecastingModel(nn.Module):
    """Simple linear projection model for fast trainer unit tests."""

    def __init__(self, in_features: int = 4, out_len: int = 6):
        super().__init__()
        self.out_len = out_len
        self.proj = nn.Linear(in_features, out_len)

    def forward(self, x, wind_dir=None, wind_spd=None):
        # x: [B, T_in, N, F] -> mean over T_in -> [B, N, F] -> proj -> [B, N, out_len] -> [B, out_len, N]
        h = x.mean(dim=1)
        out = self.proj(h).permute(0, 2, 1)
        return torch.relu(out)


class DictDataset(torch.utils.data.Dataset):
    def __init__(self, B: int = 8, T_in: int = 6, T_out: int = 6, N: int = 10, F: int = 4):
        self.x = torch.randn(B, T_in, N, F)
        self.y = torch.rand(B, T_out, N) * 100.0
        self.mask = torch.rand(B, T_out, N) > 0.1
        self.wind_dir = torch.rand(B, T_in) * 360.0
        self.wind_spd = torch.rand(B, T_in) * 12.0

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            "x": self.x[idx],
            "y": self.y[idx],
            "mask": self.mask[idx],
            "wind_dir_in": self.wind_dir[idx],
            "wind_spd_in": self.wind_spd[idx],
        }


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    ckpt_dir = tmp_path / "test_checkpoints"
    yield ckpt_dir
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)


class TestModelTrainer:
    def test_trainer_train_and_eval_step(self, temp_checkpoint_dir):
        model = DummyForecastingModel(in_features=4, out_len=6)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        dataset = DictDataset(B=8, T_in=6, T_out=6, N=10, F=4)
        loader = DataLoader(dataset, batch_size=4)

        trainer = ModelTrainer(
            model=model,
            optimizer=optimizer,
            device="cpu",
            checkpoint_dir=temp_checkpoint_dir,
        )

        # 1. Test single training epoch
        loss = trainer.train_epoch(loader)
        assert isinstance(loss, float)
        assert loss > 0.0

        # 2. Test evaluation
        metrics = trainer.evaluate(loader)
        assert "mae" in metrics and "rmse" in metrics and "score" in metrics
        assert metrics["mae"] > 0.0
        assert metrics["rmse"] > 0.0
        assert metrics["score"] > 0.0

    def test_trainer_fit_and_checkpoint_preservation(self, temp_checkpoint_dir):
        model = DummyForecastingModel(in_features=4, out_len=6)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        train_loader = DataLoader(DictDataset(B=12, T_in=6, T_out=6, N=10, F=4), batch_size=4)
        val_loader = DataLoader(DictDataset(B=8, T_in=6, T_out=6, N=10, F=4), batch_size=4)

        trainer = ModelTrainer(
            model=model,
            optimizer=optimizer,
            device="cpu",
            checkpoint_dir=temp_checkpoint_dir,
        )

        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=2,
            early_stopping_patience=2,
            verbose=False,
        )

        assert len(history["train_loss"]) == 2
        assert len(history["val_score"]) == 2
        assert trainer.best_checkpoint_path is not None
        assert trainer.best_checkpoint_path.exists()

        # Test loading saved checkpoint
        trainer.load_best_checkpoint()
