# scripts/plot_distance_distributions.py
"""Per-class pairwise-distance densities. Doubles as generator validation
and as the answer to 'what do the distance distributions look like?'."""
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = Path("data/clouds.pkl")
OUT_PATH = Path("results/distance_distributions.png")

N_CLOUDS_PER_CLASS = 50   # clouds to pool per class
N_PAIRS_PER_CLOUD = 5000  # matches the experiment-3 sample count
N_BINS = 80
SEED = 0

# Unit square -> diameter sqrt(2). Same normalization as PairDistanceCDF,
# so this figure speaks the same coordinates as the model input.
L = np.sqrt(2)


def sample_pair_distances(points: np.ndarray, n_pairs: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Sample n_pairs distances between distinct points. Mirrors the
    PairDistanceCDF sampling exactly."""
    n = len(points)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    collision = i == j
    while collision.any():
        j[collision] = rng.integers(0, n, size=collision.sum())
        collision = i == j
    diffs = points[i] - points[j]
    return np.sqrt(np.sum(diffs ** 2, axis=1)) / L


def main():
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    clouds = data["clouds"]
    labels = np.asarray(data["labels"])
    label_names = data["label_names"]

    rng = np.random.default_rng(SEED)

    # Pool distances by class.
    distances_by_class: dict[str, np.ndarray] = {}
    for class_idx, name in enumerate(label_names):
        idxs_for_class = np.where(labels == class_idx)[0][:N_CLOUDS_PER_CLASS]
        pooled = np.concatenate([
            sample_pair_distances(clouds[i].points, N_PAIRS_PER_CLOUD, rng)
            for i in idxs_for_class
        ])
        distances_by_class[name] = pooled

    # Shared x-range and bin edges so all panels are visually comparable.
    bins = np.linspace(0.0, 1.0, N_BINS + 1)
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)

    # Per-class panels.
    for k, (name, dists) in enumerate(distances_by_class.items()):
        axes[k].hist(dists, bins=bins, density=True, color=colors[k],
                     alpha=0.85, edgecolor="none")
        axes[k].set(title=name, xlabel="distance / sqrt(2)")
        axes[k].set_xlim(0, 1)
    axes[0].set_ylabel("density")

    # Overlay panel.
    for k, (name, dists) in enumerate(distances_by_class.items()):
        axes[3].hist(dists, bins=bins, density=True, color=colors[k],
                     alpha=0.4, label=name, edgecolor="none")
    axes[3].set(title="overlay", xlabel="distance / sqrt(2)")
    axes[3].set_xlim(0, 1)
    axes[3].legend(fontsize=9)

    fig.suptitle(
        f"Pairwise-distance densities ",
        fontsize=11,
    )
    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()