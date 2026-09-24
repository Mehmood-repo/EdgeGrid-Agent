"""Automated benchmarking runner comparing baseline and wake-informed models."""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Optional
import pandas as pd
import torch
import torch.optim as torch_optim

from src.data.dataset import build_dataloaders
from src.graph.topology import build_static_graph
from src.models.baselines import StaticSTGCN, TemporalGRU
from src.models.edgegrid_net import EdgeGridNet
from src.training.trainer import ModelTrainer


def count_parameters(model: torch.nn.Module) -> int:
    """Returns total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_benchmark(
    data_path: str = "data/processed/sdwpf_cleaned_243days.parquet",
    locations_path: str = "data/raw/sdwpf_baidukddcup2022_turb_location.csv",
    epochs: int = 3,
    batch_size: int = 8,
    in_len: int = 24,
    out_len: int = 24,
    hidden_dim: int = 32,
    lr: float = 1e-3,
    train_stride: int = 12,
    eval_stride: int = 72,
    device: Optional[str] = None,
    checkpoint_dir: str = "checkpoints/benchmark",
    verbose: bool = True,
) -> pd.DataFrame:
    """Runs an automated benchmark across TemporalGRU, StaticSTGCN, and EdgeGridNet.

    Args:
        data_path: Path to parquet dataset.
        locations_path: Path to turbine locations CSV.
        epochs: Number of training epochs per model.
        batch_size: DataLoader batch size.
        in_len: Input history steps.
        out_len: Forecast horizon steps.
        hidden_dim: Model hidden dimension size.
        lr: AdamW learning rate.
        train_stride: Window stride for training data.
        eval_stride: Window stride for validation and test data.
        device: 'cuda' or 'cpu'.
        checkpoint_dir: Directory to store model checkpoints.
        verbose: Print progress.

    Returns:
        DataFrame summarizing benchmark performance across all architectures.
    """
    device_obj = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if verbose:
        print(f"=== Starting EdgeGrid Benchmark on {device_obj.type.upper()} ===")

    # 1. Build Spatial Graph Topology
    if verbose:
        print("Constructing spatial turbine graph...")
    graph = build_static_graph(locations_path=locations_path, method="delaunay")

    # 2. Build DataLoaders
    if verbose:
        print("Building synchronized Train / Val / Test DataLoaders...")
    train_loader, val_loader, test_loader, scaler = build_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        in_len=in_len,
        out_len=out_len,
        train_stride=train_stride,
        eval_stride=eval_stride,
        scale_target=False,
    )

    models_to_test = {
        "TemporalGRU": TemporalGRU(
            in_features=10,
            hidden_dim=hidden_dim,
            num_layers=2,
            out_len=out_len,
        ),
        "StaticSTGCN": StaticSTGCN(
            graph=graph,
            in_features=10,
            hidden_dim=hidden_dim,
            out_len=out_len,
        ),
        "EdgeGridNet": EdgeGridNet(
            graph=graph,
            in_features=10,
            hidden_dim=hidden_dim,
            edge_dim=6,
            heads=2,
            temporal_layers=1,
            out_len=out_len,
        ),
    }

    results = []

    for name, model in models_to_test.items():
        if verbose:
            print(f"\n--- Benchmarking: {name} ---")

        params = count_parameters(model)
        optimizer = torch_optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        trainer = ModelTrainer(
            model=model,
            optimizer=optimizer,
            device=device_obj,
            checkpoint_dir=f"{checkpoint_dir}/{name.lower()}",
            scaler=scaler,
        )

        start_time = time.time()
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            early_stopping_patience=2,
            verbose=verbose,
        )
        elapsed_time = time.time() - start_time

        # Load best model checkpoint before final test evaluation
        trainer.load_best_checkpoint()
        val_metrics = trainer.evaluate(val_loader)
        test_metrics = trainer.evaluate(test_loader)

        results.append(
            {
                "Model": name,
                "Parameters": params,
                "Train Time (s)": round(elapsed_time, 1),
                "Val MAE (kW)": round(val_metrics["mae"], 2),
                "Val RMSE (kW)": round(val_metrics["rmse"], 2),
                "Val Score": round(val_metrics["score"], 2),
                "Test MAE (kW)": round(test_metrics["mae"], 2),
                "Test RMSE (kW)": round(test_metrics["rmse"], 2),
                "Test Score": round(test_metrics["score"], 2),
            }
        )

    results_df = pd.DataFrame(results)
    if verbose:
        print("\n=== Benchmark Summary Results ===")
        print(results_df.to_string(index=False))

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EdgeGrid Model Benchmark")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--in-len", type=int, default=24)
    parser.add_argument("--out-len", type=int, default=24)
    args = parser.parse_args()

    run_benchmark(
        epochs=args.epochs,
        batch_size=args.batch_size,
        in_len=args.in_len,
        out_len=args.out_len,
    )
