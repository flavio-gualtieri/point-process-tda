import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from cloudforger.processes.matern import MaternHardCoreProcess
from cloudforger.processes.poisson import PoissonProcess
from cloudforger.processes.thomas import ThomasProcess
from cloudforger.core.region import Box

PROCESSES = {
    "Matern": MaternHardCoreProcess(parent_intensity=1000, hardcore_radius=0.03),
}

SEED = 20260519
REGION = Box(low=[0, 0], high=[1, 1])
N_POINTS = 500

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, (name, proc) in zip(axes, PROCESSES.items()):
    cloud = proc.sample(n=N_POINTS, region=REGION, seed=SEED)
    pts = cloud.points
    ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.6, linewidths=0)
    ax.set_title(name, fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.02, 0.97, f"n={len(pts)}", transform=ax.transAxes,
            fontsize=9, va="top", color="gray")

fig.suptitle("Example point clouds by process type", fontsize=14, y=1.01)
fig.tight_layout()

out = Path("results/example_clouds.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
