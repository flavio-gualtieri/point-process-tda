# scripts/analyze_experiments.py
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import pathlib
import numpy as np

torch.serialization.add_safe_globals([
    pathlib.PosixPath,
    np._core.multiarray._reconstruct,
    np.ndarray,
    np.dtype,
    np.dtypes.Float64DType,
])

# Each entry: (label, path-to-results.pt)
RUNS = [
    ("Raw cloud", Path("/Users/qp252676/Desktop/point-process-tda/results/params/thomas/raw_pc/results.pt")),
    ("Pairwise", Path("/Users/qp252676/Desktop/point-process-tda/results/params/thomas/pairwise/results.pt")),
    ("PI H0", Path("/Users/qp252676/Desktop/point-process-tda/results/params/thomas/pi_0/results.pt")),
    ("PI H1", Path("/Users/qp252676/Desktop/point-process-tda/results/params/thomas/pi_1/results.pt")),
]

# Load every run's results, skipping any that haven't been run yet.
loaded = []
for name, path in RUNS:
    if not path.exists():
        print(f"  [skip] {name}: no results at {path}")
        continue
    res = torch.load(path, weights_only=True)
    loaded.append((name, res))

if not loaded:
    raise SystemExit("No results found — run at least one experiment first.")

# Training curves — overlay all runs.
fig, axes = plt.subplots(1, 1, figsize=(6, 4))
axes = [axes]
colors = plt.cm.tab10.colors  # distinct color per run

""" for i, (name, res) in enumerate(loaded):
    history = res["history"]
    c = colors[i]
    # Loss: solid = train, dashed = val, same color per run.
    axes[0].plot(history["train_loss"], color=c, linestyle="-",
                 label=f"{name} (train)")
    axes[0].plot(history["val_loss"], color=c, linestyle="--",
                 label=f"{name} (val)")
    axes[1].plot(history["train_acc"], color=c, linestyle="-",
                 label=f"{name} (train)")
    axes[1].plot(history["val_acc"], color=c, linestyle="--",
                 label=f"{name} (val)") """

for i, (name, res) in enumerate(loaded):
    history = res["history"]
    c = colors[i]
    # Loss: solid = train, dashed = val, same color per run.
    axes[0].plot(history["train_loss"], color=c, linestyle="-",
                 label=f"{name} (train)")
    axes[0].plot(history["val_loss"], color=c, linestyle="--",
                 label=f"{name} (val)")


axes[0].set(xlabel="epoch", ylabel="loss", title="Loss")
axes[0].legend(fontsize=8)

""" axes[0].set(xlabel="epoch", ylabel="loss", title="Loss")
axes[0].legend(fontsize=8)
axes[1].set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
axes[1].legend(fontsize=8) """

plt.tight_layout()
plt.savefig("figs/params_comparison.png", dpi=150)