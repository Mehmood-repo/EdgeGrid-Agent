# Running EdgeGrid-Agent on Kaggle (GPU Guide)

This guide provides the exact, step-by-step instructions to execute the `EdgeGrid-Agent` benchmark on Kaggle using free cloud GPUs (NVIDIA T4 or P100).

---

## Step 1: Upload Data as a Kaggle Dataset

Because `sdwpf_cleaned_243days.parquet` (~114 MB) is ignored by Git, upload it directly to Kaggle:

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) and click **"New Dataset"**.
2. Title your dataset: `edgegrid-sdwpf-data`.
3. Upload the two files:
   - `data/processed/sdwpf_cleaned_243days.parquet`
   - `data/raw/sdwpf_baidukddcup2022_turb_location.csv`
4. Click **Create** (takes ~15 seconds).

---

## Step 2: Create a Kaggle Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code) and click **"New Notebook"**.
2. In the right-hand **Notebook Options** panel:
   - **Accelerator:** Set to **GPU T4 x 2** or **GPU P100**.
   - **Internet:** Toggle to **On** (required for `git clone` and `pip install`).
3. Click **"Add Input"** at the top right of the notebook, search for your dataset `edgegrid-sdwpf-data`, and attach it.

---

## Step 3: Run the Code Cells

### Cell 1: Clone Repository & Install PyG
```python
import os

# Clone latest code from GitHub
!git clone https://github.com/Mehmood-repo/EdgeGrid-Agent.git
%cd EdgeGrid-Agent

# Install PyTorch Geometric
!pip install -q torch-geometric
```

### Cell 2: Verify GPU & Detect Attached Dataset
```python
import glob
import torch

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

# Locate attached dataset files
parquet_matches = glob.glob("/kaggle/input/**/*.parquet", recursive=True)
locations_matches = glob.glob("/kaggle/input/**/*location*.csv", recursive=True)

assert len(parquet_matches) > 0, "Parquet dataset not found! Check your attached input."
assert len(locations_matches) > 0, "Turbine location CSV not found! Check your attached input."

data_path = parquet_matches[0]
locations_path = locations_matches[0]

print(f"Data Path: {data_path}")
print(f"Locations Path: {locations_path}")
```

### Cell 3: Execute the Full Model Benchmark on GPU
```python
# Runs the automated benchmark across TemporalGRU, StaticSTGCN, and EdgeGridNet
!python -m src.training.benchmark \
    --data-path "{data_path}" \
    --locations-path "{locations_path}" \
    --epochs 5 \
    --batch-size 16 \
    --in-len 24 \
    --out-len 24 \
    --hidden-dim 64 \
    --lr 0.001 \
    --train-stride 6 \
    --eval-stride 48 \
    --device cuda \
    --checkpoint-dir /kaggle/working/checkpoints \
    --output-csv /kaggle/working/benchmark_results.csv
```

### Cell 4: View and Plot Benchmark Results
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results table
df_results = pd.read_csv("/kaggle/working/benchmark_results.csv")
print("=== Benchmark Summary Table ===")
display(df_results)

# Generate comparison bar chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Test KDD Cup Combined Score (Lower is Better)
bars1 = ax1.bar(df_results["Model"], df_results["Test Score"], color=["#3498db", "#2ecc71", "#e74c3c"])
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')
ax1.set_ylabel("KDD Cup Combined Score [½(MAE + RMSE)]")
ax1.set_title("Test Generalization Score (Lower is Better)")
ax1.grid(True, linestyle=":", alpha=0.6)

# Panel 2: Training Throughput Time
bars2 = ax2.bar(df_results["Model"], df_results["Train Time (s)"], color=["#2980b9", "#27ae60", "#c0392b"])
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.1f}s", ha='center', va='bottom', fontweight='bold')
ax2.set_ylabel("Training Time (seconds)")
ax2.set_title("GPU Computational Efficiency")
ax2.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig("/kaggle/working/benchmark_comparison.png", dpi=300)
plt.show()
```

---

## Step 4: Download Outputs

Once execution finishes:
1. In the right-hand panel under **Output** (`/kaggle/working/`):
   - `benchmark_results.csv`: Complete numerical performance table.
   - `benchmark_comparison.png`: Generated comparison visualization.
   - `checkpoints/`: Best model `.pt` checkpoint weights for each architecture.
2. Click **Download** on any file or save version to commit the notebook.
