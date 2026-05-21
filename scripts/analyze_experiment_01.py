# scripts/analyze_experiment_01.py
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import pathlib

torch.serialization.add_safe_globals([pathlib.PosixPath])

# Each entry: (label, path-to-results.pt)
RUNS = [
    ("point cloud", Path("results/experiment_01/results.pt")),
    ("persistence image", Path("results/experiment_01_pi/results.pt")),
]

# Load every run's results.
loaded = []
for name, path in RUNS:
    res = torch.load(path, weights_only=True)
    loaded.append((name, res))

# Training curves — overlay all runs.
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
colors = plt.cm.tab10.colors  # distinct color per run

for i, (name, res) in enumerate(loaded):
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
                 label=f"{name} (val)")

axes[0].set(xlabel="epoch", ylabel="loss", title="Loss")
axes[0].legend(fontsize=8)
axes[1].set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("results/experiment_01/training_curves_comparison.png", dpi=150)

# Test accuracies side by side.
print("Test accuracy by run:")
for name, res in loaded:
    print(f"  {name:20s} {res['test_acc']:.3f}")