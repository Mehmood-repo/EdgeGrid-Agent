"""Temporal split configuration for wind farm operational time series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class TemporalSplitConfig:
    """Chronological train / validation / test day ranges.

    Avoids future lookahead leakage across wind turbine forecasting horizons.
    """

    train_days: Tuple[int, int] = (1, 175)
    val_days: Tuple[int, int] = (176, 205)
    test_days: Tuple[int, int] = (206, 243)

    def get_day_range(self, split: str) -> Tuple[int, int]:
        if split == "train":
            return self.train_days
        elif split in ("val", "validation"):
            return self.val_days
        elif split == "test":
            return self.test_days
        else:
            raise ValueError(f"Unknown split: '{split}'. Must be 'train', 'val', or 'test'.")
