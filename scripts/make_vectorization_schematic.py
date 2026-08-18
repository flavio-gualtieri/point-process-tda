#!/usr/bin/env python3
# scripts/make_vectorization_schematic.py
"""Build the "reader who does not know TDA" schematic requested by the
\\fig{} TODO in writeup/writeup.tex Section ~7 (persistence-image
definition block): one point cloud -> its H0/H1 diagrams -> the five
vectorizations used elsewhere in this repo (PI raster, landscape stack,
silhouette curve, Betti curve, persistence-statistics bar), for one
tight-cluster cloud and one near-Poisson cloud.

The two designs are NOT invented for this figure: they are the exact
(kappa, mu, sigma) triples already named "A_tight_cluster" and
"C_near_poisson" in audit/validate_generator.py's TRIPLES dict, so the
schematic depicts the same generator configurations already validated
there rather than a fresh, uncalibrated pair.

Every vectorizer used below is the real production class from
src/cloudforger/vectorization -- this script only wires them together and
plots; it does not reimplement any TDA math.

Usage:
    python scripts/make_vectorization_schematic.py

Output:
    figs/vectorization_schematic.png
    figs/vectorization_schematic.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.calibration import axis_bounds, axis_bounds_1d  # noqa: E402
from cloudforger.core.region import Box  # noqa: E402
from cloudforger.data_generation.filtration.rips import RipsFiltration  # noqa: E402
from cloudforger.data_generation.point_processes.thomas import ThomasProcess  # noqa: E402
from cloudforger.vectorization.landscapes.landscape_silhouette import (  # noqa: E402
    LandscapeTransformer,
    SilhouetteTransformer,
)
from cloudforger.vectorization.persistence_images.persistence_image import (  # noqa: E402
    PersistenceImager,
)
from cloudforger.vectorization.scalar_features.betti_curve import BettiCurve  # noqa: E402
from cloudforger.vectorization.scalar_features.persistence_statistics import (  # noqa: E402
    STAT_NAMES,
    _stats_from_pairs,
)

FIG_DIR = ROOT / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Same two named designs as audit/validate_generator.py's TRIPLES -- reused
# verbatim (not re-derived) so this schematic and the generator audit agree
# on what "tight cluster" / "near Poisson" mean.
DESIGNS = {
    "A_tight_cluster": dict(kappa=50.0, mu=4.0, sigma=0.01, title="Tight cluster"),
    "C_near_poisson": dict(kappa=4.0, mu=5.0, sigma=0.5, title="Near-Poisson"),
}
SEED = 20260818
W = Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]))

# dataviz reference palette (validated categorical slots 1/2), matching
# audit/validate_generator.py's convention.
H0_COLOR = "#2a78d6"
H1_COLOR = "#eb6834"
GRAY = "#8a8a86"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#c9c8c3",
    "axes.grid": False,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

N_LANDSCAPE_LAYERS = 3
STAT_SUBSET = ("count", "total_persistence", "persistent_entropy")
STAT_IDX = {name: i for i, name in enumerate(STAT_NAMES)}


def sample_cloud(label: str, params: dict):
    proc = ThomasProcess(parent_intensity=params["kappa"], mean_offspring=params["mu"],
                          cluster_scale=params["sigma"])
    cloud = proc.sample(region=W, seed=SEED + hash(label) % 10_000)
    return cloud


def plot_cloud(ax, cloud, title):
    ax.scatter(cloud.points[:, 0], cloud.points[:, 1], s=10, color=GRAY, edgecolor="black", linewidth=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)


def plot_diagram(ax, diagram, title):
    h0 = diagram.finite_pairs(0)
    h1 = diagram.finite_pairs(1)
    all_pairs = np.concatenate([h0, h1], axis=0) if len(h0) + len(h1) > 0 else np.zeros((1, 2))
    hi = float(all_pairs.max()) * 1.1 if all_pairs.max() > 0 else 1.0
    ax.plot([0, hi], [0, hi], color=GRAY, lw=1, ls="--", zorder=0)
    if len(h0):
        ax.scatter(h0[:, 0], h0[:, 1], s=14, color=H0_COLOR, label="$H_0$", zorder=2)
    if len(h1):
        ax.scatter(h1[:, 0], h1[:, 1], s=14, color=H1_COLOR, marker="^", label="$H_1$", zorder=2)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("birth", fontsize=8)
    ax.set_ylabel("death", fontsize=8)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=7, loc="lower right", handletextpad=0.2)


def plot_pi(ax, diagram, title):
    """Both dims as one raster: H0 stacked above H1, separated by a line --
    a single 'PI raster' panel that still shows both channels."""
    resolution = 48
    ims = []
    for dim in (0, 1):
        pairs = diagram.finite_pairs(dim)
        if len(pairs) == 0:
            ims.append(np.zeros((resolution, resolution)))
            continue
        births = pairs[:, 0]
        perss = pairs[:, 1] - pairs[:, 0]
        birth_range = (float(births.min()), float(births.max()) * 1.1 + 1e-9)
        pers_range = (0.0, float(perss.max()) * 1.1 + 1e-9)
        # H0 births are all exactly 0 for a Rips filtration on a point cloud
        # (every vertex exists from the start) -- that birth axis is
        # degenerate by construction, not a bug, so widen it into a visible
        # window rather than feeding PersistenceImager a near-zero-width range.
        if birth_range[1] - birth_range[0] < 1e-6:
            birth_range = (birth_range[0] - 0.5, birth_range[0] + 0.5)
        imager = PersistenceImager(birth_range=birth_range, pers_range=pers_range,
                                    resolution=resolution, sigma_pixels=2.0)
        ims.append(imager.transform(diagram, dim))

    stack = np.concatenate([ims[0] / (ims[0].max() or 1.0), ims[1] / (ims[1].max() or 1.0)], axis=0)
    ax.imshow(stack, cmap="magma", aspect="auto")
    ax.axhline(resolution - 0.5, color="white", lw=1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.02, 0.97, "$H_0$", color="white", fontsize=8, transform=ax.transAxes, va="top", ha="left")
    ax.text(0.02, 0.47, "$H_1$", color="white", fontsize=8, transform=ax.transAxes, va="top", ha="left")
    ax.set_title(title)


def plot_landscape(ax, diagram, title):
    pairs = diagram.finite_pairs(1)
    if len(pairs) == 0:
        ax.text(0.5, 0.5, "no $H_1$ features", ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.set_title(title)
        return
    t_min, T = axis_bounds_1d([diagram], homology_dim=1, q=1.0, pad_factor=1.05)
    transformer = LandscapeTransformer(t_min=t_min, T=T, G=200, K=N_LANDSCAPE_LAYERS)
    landscapes = transformer.transform(diagram, dim=1)
    shades = np.linspace(0.9, 0.35, N_LANDSCAPE_LAYERS)
    for k in range(N_LANDSCAPE_LAYERS):
        ax.plot(transformer.grid, landscapes[k], color=H1_COLOR, alpha=shades[k], lw=1.5,
                label=f"$\\lambda_{k+1}$")
    ax.fill_between(transformer.grid, landscapes[0], color=H1_COLOR, alpha=0.15)
    ax.set_xlabel("$t$", fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc="upper right", handletextpad=0.2)
    ax.set_title(title)


def plot_silhouette(ax, diagram, title):
    pairs = diagram.finite_pairs(1)
    if len(pairs) == 0:
        ax.text(0.5, 0.5, "no $H_1$ features", ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.set_title(title)
        return
    t_min, T = axis_bounds_1d([diagram], homology_dim=1, q=1.0, pad_factor=1.05)
    transformer = SilhouetteTransformer(t_min=t_min, T=T, G=200, p=1.0)
    out = transformer.transform(diagram, dim=1)
    ax.plot(transformer.grid, out["silhouette"], color=H1_COLOR, lw=2)
    ax.fill_between(transformer.grid, out["silhouette"], color=H1_COLOR, alpha=0.2)
    ax.set_xlabel("$t$", fontsize=8)
    ax.set_title(title)


def plot_betti(ax, diagram, title):
    all_pairs = np.concatenate([diagram.finite_pairs(0), diagram.finite_pairs(1)], axis=0)
    hi = float(all_pairs[:, 1].max()) * 1.05 if len(all_pairs) else 1.0
    betti = BettiCurve(homology_dims=(0, 1), grid_size=200, grid_range=(0.0, hi))
    feat = betti.compute(diagram)
    ax.plot(betti.grid, feat.curves[0], color=H0_COLOR, lw=2, label="$\\beta_0$")
    ax.plot(betti.grid, feat.curves[1], color=H1_COLOR, lw=2, label="$\\beta_1$")
    ax.set_xlabel("$t$", fontsize=8)
    ax.set_ylabel("# features alive", fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title(title)


def plot_stats(ax, diagram, title):
    """Bar heights are per-statistic min-max normalized (each stat's own H0/H1
    pair rescaled to [0, 1]) because count (~tens-hundreds), total persistence
    (~O(1)), and entropy (~O(1)) live on incomparable raw scales -- sharing one
    linear axis would make the persistence/entropy bars invisible next to
    count. Raw values are printed above each bar so nothing is hidden, only
    rescaled for legibility."""
    h0_stats = _stats_from_pairs(diagram.finite_pairs(0))
    h1_stats = _stats_from_pairs(diagram.finite_pairs(1))
    x = np.arange(len(STAT_SUBSET))
    width = 0.35
    h0_vals = np.array([h0_stats[STAT_IDX[name]] for name in STAT_SUBSET])
    h1_vals = np.array([h1_stats[STAT_IDX[name]] for name in STAT_SUBSET])
    scale = np.maximum(np.maximum(h0_vals, h1_vals), 1e-12)
    h0_norm, h1_norm = h0_vals / scale, h1_vals / scale
    bars0 = ax.bar(x - width / 2, h0_norm, width, color=H0_COLOR, label="$H_0$")
    bars1 = ax.bar(x + width / 2, h1_norm, width, color=H1_COLOR, label="$H_1$")
    for bar, val in zip(list(bars0) + list(bars1), list(h0_vals) + list(h1_vals)):
        label = f"{val:.0f}" if val >= 10 else f"{val:.2f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03, label,
                 ha="center", va="bottom", fontsize=6, rotation=90)
    ax.set_ylim(0, 1.35)
    ax.set_yticks([])
    ax.set_xticks(x)
    ax.set_xticklabels(["count", "tot.\npersist.", "entropy"], fontsize=7)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title(title)


def main():
    fig, axes = plt.subplots(2, 7, figsize=(22, 6.4))
    col_titles = ["Point cloud", "Persistence diagram", "Persistence image",
                  "Landscape ($H_1$)", "Silhouette ($H_1$)", "Betti curve", "Statistics"]

    filtration = RipsFiltration(maxdim=1)

    for row, (label, params) in enumerate(DESIGNS.items()):
        cloud = sample_cloud(label, params)
        diagram = filtration.compute(cloud)

        row_axes = axes[row]
        plot_cloud(row_axes[0], cloud, col_titles[0] if row == 0 else "")
        plot_diagram(row_axes[1], diagram, col_titles[1] if row == 0 else "")
        plot_pi(row_axes[2], diagram, col_titles[2] if row == 0 else "")
        plot_landscape(row_axes[3], diagram, col_titles[3] if row == 0 else "")
        plot_silhouette(row_axes[4], diagram, col_titles[4] if row == 0 else "")
        plot_betti(row_axes[5], diagram, col_titles[5] if row == 0 else "")
        plot_stats(row_axes[6], diagram, col_titles[6] if row == 0 else "")

        row_axes[0].set_ylabel(
            f"{params['title']}\n$\\kappa$={params['kappa']}, $\\mu$={params['mu']}, "
            f"$\\sigma$={params['sigma']}\n(n={cloud.n_points})",
            fontsize=8, rotation=0, ha="right", va="center", labelpad=45,
        )

    fig.suptitle(
        "One point cloud $\\rightarrow$ its $H_0/H_1$ diagram $\\rightarrow$ five vectorizations",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    png_path = FIG_DIR / "vectorization_schematic.png"
    pdf_path = FIG_DIR / "vectorization_schematic.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
