# EdgeGrid-Agent Project Notes & Quick Reference

## Key Documentation
- **Comprehensive System Architecture Report:** [`docs/system_architecture_report.md`](file:///home/ali/projects/EdgeGrid-Agent/docs/system_architecture_report.md)
- **Empirical Wake & Adjacent Turbine Study:** [`docs/spatial_wake_analysis_report.md`](file:///home/ali/projects/EdgeGrid-Agent/docs/spatial_wake_analysis_report.md)

## Executable Notebooks
- [`notebooks/01_data_exploration.ipynb`](file:///home/ali/projects/EdgeGrid-Agent/notebooks/01_data_exploration.ipynb): Data cleaning, outage pruning, spline imputation, and parquet serialization.
- [`notebooks/02_adjacent_turbine_wake_analysis.ipynb`](file:///home/ali/projects/EdgeGrid-Agent/notebooks/02_adjacent_turbine_wake_analysis.ipynb): Pre-executed adjacent turbine correlation and wake deficit study with inline plots.

## Core Packages (`src/`)
- `src/graph/`: [`topology.py`](file:///home/ali/projects/EdgeGrid-Agent/src/graph/topology.py), [`dynamic_weights.py`](file:///home/ali/projects/EdgeGrid-Agent/src/graph/dynamic_weights.py)
- `src/data/`: [`dataset.py`](file:///home/ali/projects/EdgeGrid-Agent/src/data/dataset.py), [`scaler.py`](file:///home/ali/projects/EdgeGrid-Agent/src/data/scaler.py), [`splits.py`](file:///home/ali/projects/EdgeGrid-Agent/src/data/splits.py)
- `src/models/`: [`baselines.py`](file:///home/ali/projects/EdgeGrid-Agent/src/models/baselines.py), [`edgegrid_net.py`](file:///home/ali/projects/EdgeGrid-Agent/src/models/edgegrid_net.py)
- `src/training/`: [`loss.py`](file:///home/ali/projects/EdgeGrid-Agent/src/training/loss.py), [`trainer.py`](file:///home/ali/projects/EdgeGrid-Agent/src/training/trainer.py), [`benchmark.py`](file:///home/ali/projects/EdgeGrid-Agent/src/training/benchmark.py)

## Running Tests & Benchmark
```bash
# Run all 19 unit tests
.venv/bin/pytest tests/ -v

# Run the 3-model benchmark
.venv/bin/python -m src.training.benchmark --epochs 3 --batch-size 8
```
