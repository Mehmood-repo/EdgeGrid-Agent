# EdgeGrid: Physics-Motivated Spatio-Temporal Graph Neural Networks for Industrial Wind Power Forecasting

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5%2B-EE4C2C.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-torch--geometric-3C2179.svg)](https://pyg.org/)
[![Tests](https://img.shields.io/badge/tests-19%20passed-brightgreen.svg)](tests/)
[![Dataset](https://img.shields.io/badge/Baidu%20KDD%20Cup%202022-SDWPF-orange.svg)](https://arxiv.org/abs/2208.04360)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Executive Summary & PhD Research Motivation

Accurate multi-horizon active power forecasting across large-scale wind turbine arrays is a cornerstone of modern power grid stability, dynamic dispatch, and the large-scale integration of intermittent renewables. The **Baidu KDD Cup 2022 Spatial Dynamic Wind Power Forecasting (SDWPF)** challenge provides an industrial dataset spanning **134 wind turbines** over **245 days** at 10-minute intervals.

```
Incident Wind Field (U, θ_wind)
             │
             ▼
      ┌──────────────┐         Directional Aerodynamic Wake Deficit          ┌──────────────┐
      │  Upwind      │ ────────────────────────────────────────────────────> │  Downwind    │
      │  Turbine i   │     w_ij(t) = max(0, cos(θ_wind(t) - φ_ij)) * e^(-d)  │  Turbine j   │
      └──────────────┘                                                       └──────────────┘
             │                                                                      │
             ▼                                                                      ▼
      P_i(t) ~ v_i(t)^3                                                      P_j(t) < P_i(t)
```

### The Scientific Research Gap
Standard Spatio-Temporal Graph Neural Networks (ST-GNNs)—widely deployed in traffic and sensor network forecasting—predominantly assume **static, symmetric, distance-based adjacency matrices** ($\mathbf{A} \in \mathbb{R}^{N \times N}$ where $A_{ij} = A_{ji} = \exp(-d_{ij}^2/\sigma^2)$). In atmospheric fluid flow, this assumption fundamentally violates physical reality:
1. **Directional Asymmetry:** When wind blows from turbine $i$ to turbine $j$ ($\theta_{\text{wind}} \approx \phi_{ij}$), turbine $i$ generates a turbulent velocity deficit on turbine $j$. Turbine $j$ exerts negligible upstream aerodynamic wake influence on turbine $i$.
2. **Temporal Non-Stationarity:** As the regional wind regime rotates, the direction and topology of physical interaction completely invert.
3. **Graph Scalability:** Dense pairwise adjacency ($\mathcal{O}(N^2) = 17,956$ edges) incurs severe computational overhead and introduces spurious spatial smoothing across cross-wind turbines that do not interact.

### The Proposed Paradigm: `EdgeGridNet`
This doctoral research project develops **`EdgeGridNet`**, a physics-motivated dynamic graph architecture that dynamically conditions graph message passing on instantaneous aerodynamic wind vectors across a planar Delaunay triangulation.

---

## 2. Core Research Questions (RQs)

This repository is designed to rigorously answer four fundamental research questions:

* **RQ1 (Value of Spatial Inductive Bias):** Does explicit graph inductive bias provide measurable predictive gains over decoupled, independent temporal models (`TemporalGRU`) when scaling from short-term dispatch (4h) to multi-day planning (48h)?
* **RQ2 (Dynamic Asymmetry vs. Static Homophily):** Does time-varying, wind-directed dynamic edge gating outperform static, isotropic Euclidean graph convolutions (`StaticSTGCN`)?
* **RQ3 (The MAE vs. RMSE Bias-Variance Trade-Off):** How does dynamic edge pruning influence the distribution of prediction errors (reducing farm-wide systematic bias vs. regularizing localized extreme spikes)?
* **RQ4 (Computational Efficiency for Edge SCADA):** Can sparse planar graph formulations ($\mathcal{O}(|\mathcal{E}|)$ where $|\mathcal{E}| = 744 \ll N^2$) deliver real-time inference and training throughput suitable for on-site wind farm edge controllers?

---

## 3. Generalization to Graph Neural Network Literature & Other Problem Domains

Beyond wind energy, the core methodological contribution of this research addresses a fundamental limitation in the broader Graph Neural Network (GNN) literature: **how to perform spatial message passing in systems governed by dynamic, external vector fields**.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        Dynamic Field-Conditioned GNNs (DPEC-GNNs)                      │
├───────────────────────────────┬───────────────────────────────┬────────────────────────┤
│ Domain                        │ Dynamic Field Conditioning    │ Physical Asymmetry     │
├───────────────────────────────┼───────────────────────────────┼────────────────────────┤
│ Wind Farm Power (This Work)   │ Wind velocity vector (U, θ)   │ Wake deficit transport │
│ Atmospheric Air Pollution     │ Regional wind & pressure      │ Advective plume spread │
│ Wildfire Spread Prediction    │ Wind vector + Terrain gradient│ Frontal propagation    │
│ Smart Electric Grids          │ Real-time active power flows  │ Cascading line failure │
│ River Network Hydrology       │ Hydraulic elevation gradients │ Unidirectional runoff  │
│ Tidal Urban Traffic Flows     │ Time-varying commuter rush    │ Directional congestion │
└───────────────────────────────┴───────────────────────────────┴────────────────────────┘
```

### Direct Applications to Broader Research Areas:

1. **Atmospheric & Climate Science (Environmental Geo-AI):**
   - *Air Quality & Pollutant Tracking:* Predicting $PM_{2.5}$, $NO_x$, and greenhouse gas advection across sensor networks by replacing isotropic GCNs with wind-directed dynamic edge kernels.
   - *Wildfire Growth Modeling:* Dynamic graph message passing where edge transmission probabilities are conditioned on local wind velocity vectors and terrain topography slopes.

2. **Smart Electrical Grids & Power Systems:**
   - *Dynamic Line Rating (DLR) & Thermal Limits:* Integrating real-time weather vectors over transmission tower graphs.
   - *Autonomous Multi-Agent Wake Steering:* Coupling `EdgeGridNet` power forecasts with Reinforcement Learning (RL) agents for collaborative, farm-wide active yaw steering to deflect wakes away from downstream turbines.

3. **Computational Fluid Dynamics & Physics-Informed Machine Learning (SciML):**
   - *Mesh-Based Physical Surrogates:* Extending MeshGraphNets by injecting dynamic directional alignment kernels into spatial message passing for high-Reynolds-number turbulent flows.

4. **Decentralized Edge IoT & Resource-Constrained Embedded Systems:**
   - *Edge Computing:* Proving that planar geometric graphs (such as Delaunay triangulations) provide sparse $\mathcal{O}(|\mathcal{E}|)$ representations that run with minimal memory bandwidth overhead on edge SCADA industrial hardware.

---

## 4. Architectural Overview

```
                                 Multi-Turbine Sensor Tensor
                                  X: [B, Tin=24, N=134, F=10]
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       ▼                                               ▼
         ┌───────────────────────────┐                   ┌───────────────────────────┐
         │     Node Projection       │                   │   Dynamic Graph Weighter  │
         │   Linear + LayerNorm      │                   │   Delaunay Planar Graph   │
         │   h: [B, Tin, N, 64]      │                   │   Instantaneous (θ, v)    │
         └─────────────┬─────────────┘                   └─────────────┬─────────────┘
                       │                                               │
                       │             6D Aerodynamic Edge Vector        │
                       │           e_ij: [d, sinφ, cosφ, Δz, v∥, v⊥]   │
                       └───────────────────────┬───────────────────────┘
                                               │
                                               ▼
                               ┌───────────────────────────────┐
                               │ WakeConditionedSpatialLayer   │
                               │ TransformerConv (4 Heads)     │
                               │ Directed Dynamic Attention    │
                               └───────────────┬───────────────┘
                                               │
                                               ▼
                               ┌───────────────────────────────┐
                               │  Temporal Recurrent Backbone  │
                               │  2-Layer Gated GRU (Dim 64)   │
                               └───────────────┬───────────────┘
                                               │
                                               ▼
                               ┌───────────────────────────────┐
                               │    Multi-Horizon Forecast     │
                               │    Head: [B, Tout=288, N=134] │
                               └───────────────────────────────┘
```

### Key Architectural Components

1. **Planar Delaunay Graph ($\mathcal{E}_{\text{planar}}$):**
   - Eliminates arbitrary distance thresholding. The 134 turbines are connected via 2D Delaunay planar triangulation, yielding exactly $372$ undirected edges ($|\mathcal{E}| = 744$ directed edges), achieving a **95.9% graph sparsity ratio**.
2. **Dynamic Aerodynamic Edge Kernel ($w_{ij}(t)$):**
   - Evaluates the directional alignment between incident wind direction $\theta_{\text{wind}}(t)$ and inter-turbine bearing $\phi_{ij}$:
   $$w_{ij}(t) = \max\left(0, \cos(\theta_{\text{wind}}(t) - \phi_{ij})\right) \cdot \exp\left(-\frac{d_{ij}}{\sigma_{\text{wake}}}\right)$$
3. **6D Aerodynamic Edge Attribute Vector ($\mathbf{e}_{ij}$):**
   $$\mathbf{e}_{ij} = \left[ d_{ij}, \, \sin(\phi_{ij}), \, \cos(\phi_{ij}), \, \Delta z_{ij}, \, v_{\parallel}, \, v_{\perp} \right]$$
   where $v_{\parallel}$ and $v_{\perp}$ represent parallel wake advection and perpendicular cross-wind velocity components.
4. **Spatial Transformer Convolutions:**
   - Multi-head edge-conditioned attention scales node representations before sequence-to-sequence temporal recurrent decoding.

---

## 5. Benchmark Performance & Key Empirical Findings

Models were evaluated on NVIDIA Tesla T4 GPUs across both intra-day dispatch ($T_{\text{out}}=24$ steps = 4h) and day-ahead planning ($T_{\text{out}}=288$ steps = 48h).

### Primary 48-Hour Full Horizon Benchmark ($T_{\text{in}}=24$, $T_{\text{out}}=288$)

| Model Architecture | Parameters | Train Time | Test MAE (kW) | Test MAE (MW) | Test RMSE (kW) | Test RMSE (MW) | Test Score (kW) | Test Score (MW) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baidu Baseline (Zhou et al.)** | — | — | 280.29 | **37.56** | 351.35 | **47.08** | 315.82 | **42.32** |
| **KDD Cup Top-1 Winner (HIK)** | — | — | — | ~39.20 | — | ~50.60 | — | **44.90** |
| **TemporalGRU** | 96,224 | **71.6 s** | 271.76 | 36.42 | **361.86** | **48.49** | 316.81 | 42.45 |
| **StaticSTGCN** | 79,584 | 471.1 s | 268.12 | 35.93 | 362.93 | 48.63 | **315.53** | **42.28** |
| **EdgeGridNet (Ours)** | 92,768 | 159.2 s | **267.92** | **35.90** | 363.73 | 48.74 | 315.82 | **42.32** |

*Note: In the official competition formulation, $\text{Score}_{\text{farm}} (\text{MW}) = 0.134 \times \text{Score}_{\text{per-turbine}} (\text{kW})$ via $\alpha = \frac{134\text{ turbines}}{1000\text{ kW/MW}}$.*

### Rigorous Scientific Findings

1. **Lowest Test MAE:** `EdgeGridNet` achieves the lowest Test MAE among all evaluated architectures at **267.92 kW (35.90 MW)**, achieving a **1.66 MW (4.4%) reduction** over the published Baidu baseline ($37.56\text{ MW}$).
2. **The Bias-Variance Trade-Off:**
   - `EdgeGridNet` optimizes **median farm-wide accuracy** (lowest MAE) by pruning uncoupled cross-wind message paths.
   - `StaticSTGCN` achieves a slightly lower combined score ($42.28\text{ MW}$ vs $42.32\text{ MW}$) because isotropic Euclidean smoothing acts as a global variance regularizer, dampening extreme localized prediction errors (lower RMSE) during abrupt frontal wind shifts.
3. **$2.96\times$ Computational Speedup:**
   - Sparse Delaunay message passing ($|\mathcal{E}|=744$) trains **$2.96\times$ faster** than dense Chebyshev graph convolutions ($159.2\text{s}$ vs $471.1\text{s}$) with substantially lower GPU memory bandwidth consumption.

---

## 6. Repository Structure

```
EdgeGrid-Agent/
├── data/
│   ├── raw/
│   │   ├── sdwpf_baidukddcup2022_turb_location.csv  # Turbine spatial coordinates (x, y, z)
│   │   ├── benchmark_results.csv                    # 4h benchmark numerical results
│   │   ├── benchmark_48h_results.csv                # 48h benchmark numerical results
│   │   ├── benchmark_review.md                      # Peer-review academic critique
│   │   └── benchmark_forensic.md                    # Line-by-line forensic methodology review
│   └── processed/
│       └── sdwpf_cleaned_243days.parquet            # Cleaned, imputed, 243-day SCADA dataset
├── docs/
│   ├── kdd_cup_benchmark_research_report.md         # Comprehensive research & ablation paper
│   ├── system_architecture_report.md                # System engineering & pipeline design
│   ├── spatial_wake_analysis_report.md              # Empirical wake deficit & spatial decay study
│   ├── kaggle_execution_guide.md                    # Cloud GPU execution instructions
│   └── notes.md                                     # Project notes and documentation index
├── notebooks/
│   ├── 01_data_exploration.ipynb                    # Outage pruning, spline imputation
│   └── 02_adjacent_turbine_wake_analysis.ipynb      # Empirical wake alignment & deficit plots
├── src/
│   ├── data/
│   │   ├── dataset.py                               # Sliding window temporal PyTorch Dataset
│   │   ├── scaler.py                                # Multivariate TabularFeatureScaler
│   │   └── splits.py                                # Chronological train/val/test splits
│   ├── graph/
│   │   ├── topology.py                              # Delaunay triangulation & azimuth bearings
│   │   └── dynamic_weights.py                       # Wind-directed dynamic edge feature generator
│   ├── models/
│   │   ├── baselines.py                             # TemporalGRU and StaticSTGCN baselines
│   │   └── edgegrid_net.py                          # EdgeGridNet architecture
│   └── training/
│       ├── loss.py                                  # Masked competition score & Huber loss
│       ├── trainer.py                               # ModelTrainer with checkpoint management
│       └── benchmark.py                             # Automated cross-architecture benchmark CLI
└── tests/
    ├── test_dataset.py                              # Data pipeline & scaler unit tests
    ├── test_graph_topology.py                       # Delaunay & dynamic weight unit tests
    ├── test_models.py                               # Forward/backward pass gradient tests
    └── test_trainer.py                              # Trainer & evaluation step tests
```

---

## 7. Quickstart & Reproducibility

### 7.1 Environment Setup

This project uses modern Python packaging via `pyproject.toml` and virtual environments:

```bash
# Clone the repository
git clone https://github.com/Mehmood-repo/EdgeGrid-Agent.git
cd EdgeGrid-Agent

# Create and activate virtual environment (Python 3.12 recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and PyTorch Geometric
pip install -e .
pip install torch-geometric
```

### 7.2 Run the Unit Test Suite

All core modules are verified by 19 automated unit tests covering graph construction, dynamic edge weighting, masked metrics, and model forward/backward gradient flows:

```bash
pytest tests/ -v
# Output: 19 passed in ~9.2s
```

### 7.3 Reproduce the 48-Hour Benchmark

To train and evaluate `TemporalGRU`, `StaticSTGCN`, and `EdgeGridNet` on GPU:

```bash
python -m src.training.benchmark \
    --data-path "data/processed/sdwpf_cleaned_243days.parquet" \
    --locations-path "data/raw/sdwpf_baidukddcup2022_turb_location.csv" \
    --in-len 24 \
    --out-len 288 \
    --epochs 5 \
    --batch-size 16 \
    --hidden-dim 64 \
    --lr 0.001 \
    --device cuda \
    --output-csv "benchmark_48h_results.csv"
```

*For step-by-step instructions on reproducing results using free cloud GPUs, see the [`docs/kaggle_execution_guide.md`](docs/kaggle_execution_guide.md).*

---

## 8. Ongoing Doctoral Research & Component Ablation Roadmap

In response to peer-review feedback ([`data/raw/benchmark_review.md`](data/raw/benchmark_review.md)), the next phase of this doctoral dissertation explores a multi-tiered component isolation matrix:

1. **Topology Isolation:** Comparing Delaunay planar graphs vs. static $k$-NN graphs vs. random Erdős-Rényi controls.
2. **Edge Feature Ablation:** Systematically stripping 6D vectors down to 1D distance decay and 1D directional cosines.
3. **Falsification Control:** Inverting wind direction ($\theta_{\text{wind}} + 180^\circ$) to confirm that reversing aerodynamic flow strictly degrades predictive accuracy, proving physical causality.
4. **Regime Stratification:** Evaluating models conditioned on wind speed tiers (cut-in $3\text{–}6\text{ m/s}$, transitional $6\text{–}10\text{ m/s}$, and rated $>12\text{ m/s}$) to prove where spatial graph dynamics deliver their primary statistical advantage.

---

## 9. References & Citations

If you use this codebase or benchmark methodology in your research, please cite the foundational competition paper and this repository:

```bibtex
@article{zhou2022sdwpf,
  title={SDWPF: A Dataset for Spatial Dynamic Wind Power Forecasting Challenge at KDD Cup 2022},
  author={Zhou, Yuyan and Chen, Zhiyu and Lu, Junkai and Wang, Dongdong and Chen, Jiamin and Liu, Zhaoyang and others},
  journal={arXiv preprint arXiv:2208.04360},
  year={2022}
}

@article{yu2018spatio,
  title={Spatio-temporal graph convolutional networks: a deep learning framework for traffic forecasting},
  author={Yu, Bing and Yin, Haiyang and Zhu, Zhanxing},
  booktitle={Proceedings of the 27th International Joint Conference on Artificial Intelligence},
  pages={3634--3640},
  year={2018}
}

@article{bastankhah2014new,
  title={A new analytical model for wind-turbine wakes},
  author={Bastankhah, Majid and Port{\'e}-Agel, Fernando},
  journal={Renewable Energy},
  volume={70},
  pages={116--123},
  year={2014}
}
```

---

## 10. License

This project is licensed under the [MIT License](LICENSE).
