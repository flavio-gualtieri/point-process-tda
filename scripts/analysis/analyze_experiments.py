# scripts/analyze_experiments.py
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import pathlib

torch.serialization.add_safe_globals([pathlib.PosixPath])

# Each entry: (label, path-to-results.pt)
RUNS = [
    ("PI H0 2D", Path("/Users/qp252676/Desktop/point-process-tda/results/classify/2d/pi/h0/results.pt")),
    ("PI H0 4D", Path("/Users/qp252676/Desktop/point-process-tda/results/classify/4d/pi/h0//results.pt")),
    ("PI H0 6D", Path("/Users/qp252676/Desktop/point-process-tda/results/classify/6d/pi/h0//results.pt")),
    ("PI H0 8D", Path("/Users/qp252676/Desktop/point-process-tda/results/classify/8d/pi/h0//results.pt")),
    ("PI H0 10D", Path("/Users/qp252676/Desktop/point-process-tda/results/classify/10d/pi/h0//results.pt")),
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
plt.savefig("figs/pi_h0_comparison.png", dpi=150)

# Test accuracies side by side.
print("\nTest accuracy by run:")
for name, res in loaded:
    print(f"  {name:20s} {res['test_acc']:.3f}")