"""Training and validation engine for wind power forecasting models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.scaler import TabularFeatureScaler
from src.training.loss import (
    MaskedSmoothL1Loss,
    compute_kdd_cup_score,
    masked_mae,
    masked_rmse,
)


class ModelTrainer:
    """Manages training loops, validation evaluations, gradient clipping,

    and best-checkpoint preservation for Spatio-Temporal wind models.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        loss_fn: Optional[nn.Module] = None,
        device: Optional[Union[str, torch.device]] = None,
        checkpoint_dir: Union[str, Path] = "checkpoints",
        scaler: Optional[TabularFeatureScaler] = None,
        max_grad_norm: float = 5.0,
    ) -> None:
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn if loss_fn is not None else MaskedSmoothL1Loss()
        self.scaler = scaler
        self.max_grad_norm = max_grad_norm

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_score = float("inf")
        self.best_checkpoint_path: Optional[Path] = None

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Executes a single training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            x = batch["x"].to(self.device)  # [B, T_in, N, F]
            y = batch["y"].to(self.device)  # [B, T_out, N]
            mask = batch["mask"].to(self.device)  # [B, T_out, N]
            wind_dir = batch.get("wind_dir_in")
            wind_spd = batch.get("wind_spd_in")

            if wind_dir is not None:
                wind_dir = wind_dir.to(self.device)
            if wind_spd is not None:
                wind_spd = wind_spd.to(self.device)

            self.optimizer.zero_grad()
            y_pred = self.model(x, wind_dir=wind_dir, wind_spd=wind_spd)

            loss = self.loss_fn(y_pred, y, mask)
            loss.backward()

            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if self.scheduler is not None:
            self.scheduler.step()

        return total_loss / max(num_batches, 1)

    def evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        """Evaluates model performance against official KDD Cup masked metrics."""
        self.model.eval()
        all_preds = []
        all_trues = []
        all_masks = []

        with torch.no_grad():
            for batch in eval_loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                mask = batch["mask"].to(self.device)
                wind_dir = batch.get("wind_dir_in")
                wind_spd = batch.get("wind_spd_in")

                if wind_dir is not None:
                    wind_dir = wind_dir.to(self.device)
                if wind_spd is not None:
                    wind_spd = wind_spd.to(self.device)

                y_pred = self.model(x, wind_dir=wind_dir, wind_spd=wind_spd)

                all_preds.append(y_pred.detach().cpu())
                all_trues.append(y.detach().cpu())
                all_masks.append(mask.detach().cpu())

        y_preds = torch.cat(all_preds, dim=0)
        y_trues = torch.cat(all_trues, dim=0)
        masks = torch.cat(all_masks, dim=0)

        mae_val, rmse_val, combined_score = compute_kdd_cup_score(y_preds, y_trues, masks)

        return {
            "mae": float(mae_val),
            "rmse": float(rmse_val),
            "score": float(combined_score),
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        early_stopping_patience: int = 3,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Executes full training and validation with early stopping and checkpointing."""
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_mae": [],
            "val_rmse": [],
            "val_score": [],
        }

        patience_counter = 0
        model_name = self.model.__class__.__name__

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_mae"].append(val_metrics["mae"])
            history["val_rmse"].append(val_metrics["rmse"])
            history["val_score"].append(val_metrics["score"])

            if verbose:
                print(
                    f"Epoch {epoch:02d}/{epochs:02d} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val MAE: {val_metrics['mae']:.2f} kW | "
                    f"Val RMSE: {val_metrics['rmse']:.2f} kW | "
                    f"Score: {val_metrics['score']:.2f}"
                )

            # Checkpoint preservation
            current_score = val_metrics["score"]
            if current_score < self.best_val_score:
                self.best_val_score = current_score
                patience_counter = 0
                ckpt_path = self.checkpoint_dir / f"best_{model_name.lower()}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_score": self.best_val_score,
                    },
                    ckpt_path,
                )
                self.best_checkpoint_path = ckpt_path
                if verbose:
                    print(f"  --> Saved new best checkpoint (Score: {self.best_val_score:.2f})")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(
                            f"Early stopping triggered after {patience_counter} epochs without improvement."
                        )
                    break

        return history

    def load_best_checkpoint(self) -> None:
        """Loads weights from the best saved checkpoint."""
        if self.best_checkpoint_path is not None and self.best_checkpoint_path.exists():
            checkpoint = torch.load(self.best_checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded best checkpoint with score: {checkpoint.get('val_score', 'N/A')}")
        else:
            print("No saved checkpoint available to load.")
