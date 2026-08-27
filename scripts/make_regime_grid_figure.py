#!/usr/bin/env python3
# scripts/make_regime_grid_figure.py
"""Build the 2x4 "regime grid" figure requested by the \\figtodo in
docs/final_report.tex Section 1 (Introduction): simulated realizations of
Thomas (top row) and nested Thomas (bottom row) across the overlap index
nu in {0.1, 0.3, 0.6, 0.9} at fixed parent intensity kappa and fixed
expected count lambda, using the real simulator of
src/cloudforger/data_generation/point_processes (ssec:simulator).

nu here is exactly the "c"/"c1" design coordinate used to build training
designs (see e.g. configs/runs/thomas/logn_only.yaml, .../nested_thomas/
logn_only.yaml): cluster_scale = nu / (2*sqrt(kappa)), matching
ThomasProcess's own c1 = 2*cluster_scale*sqrt(parent_intensity). For nested
Thomas, nu drives the coarse (meta) level the same way; the fine level's
overlap index c2 and offspring count are held fixed across columns so the
second, finer level of structure stays visible throughout the row.

Within each row the same seed is reused across all four columns and every
non-cluster_scale quantity (kappa, mean offspring counts) is held fixed, so
the parent/meta-parent skeleton is identical across a row and only the
cluster spread changes -- an apples-to-apples view of nu alone. This is
intuition-building, not a publication-rigorous figure: one seed per panel,
no repeated draws, no error bars, bare axes.

Usage:
    python scripts/make_regime_grid_figure.py

Output:
    figs/regime_grid.png
    figs/regime_grid.pdf
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.core.region import Box  # noqa: E402
from cloudforger.data_generation.point_processes.thomas import ThomasProcess  # noqa: E402
from cloudforger.data_generation.point_processes.nested_thomas import NestedThomasProcess  # noqa: E402

FIG_DIR = ROOT / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

NU_VALUES = [0.1, 0.3, 0.6, 0.9]
KAPPA = 50.0       # parent intensity, fixed across the whole grid
LAMBDA = 400.0     # expected point count E[N], fixed across the whole grid
META_OFFSPRING = 3.0   # nested Thomas: mean # meta-cluster members, fixed across columns
C2_FINE = 0.15          # nested Thomas: fine-level overlap index, fixed across columns

THOMAS_SEED = 7
NESTED_SEED = 11

REGION = Box(low=[0.0, 0.0], high=[1.0, 1.0])

POINT_COLOR = "#2a78d6"
POINT_SIZE = 4.0


def thomas_process(nu: float) -> ThomasProcess:
    cluster_scale = nu / (2.0 * math.sqrt(KAPPA))
    mean_offspring = LAMBDA / KAPPA
    return ThomasProcess(
        parent_intensity=KAPPA,
        mean_offspring=mean_offspring,
        cluster_scale=cluster_scale,
    )


def nested_thomas_process(nu: float) -> NestedThomasProcess:
    meta_cluster_scale = nu / (2.0 * math.sqrt(KAPPA))
    cluster_scale = C2_FINE / (2.0 * math.sqrt(KAPPA * META_OFFSPRING))
    mean_offspring = LAMBDA / (KAPPA * META_OFFSPRING)
    return NestedThomasProcess(
        meta_parent_intensity=KAPPA,
        meta_offspring=META_OFFSPRING,
        meta_cluster_scale=meta_cluster_scale,
        mean_offspring=mean_offspring,
        cluster_scale=cluster_scale,
    )


def main():
    fig, axes = plt.subplots(2, 4, figsize=(9.6, 4.9))

    for col, nu in enumerate(NU_VALUES):
        cloud = thomas_process(nu).sample(region=REGION, seed=THOMAS_SEED)
        ax = axes[0, col]
        ax.scatter(cloud.points[:, 0], cloud.points[:, 1], s=POINT_SIZE, c=POINT_COLOR, linewidths=0)
        ax.set_title(rf"$\nu={nu:g}$", fontsize=11)

        cloud = nested_thomas_process(nu).sample(region=REGION, seed=NESTED_SEED)
        ax = axes[1, col]
        ax.scatter(cloud.points[:, 0], cloud.points[:, 1], s=POINT_SIZE, c=POINT_COLOR, linewidths=0)

    for ax in axes.flat:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[0, 0].set_ylabel("Thomas", fontsize=11)
    axes[1, 0].set_ylabel("nested Thomas", fontsize=11)

    fig.suptitle(rf"$\lambda={LAMBDA:g}$, $\kappa={KAPPA:g}$ fixed", fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    png_path = FIG_DIR / "regime_grid.png"
    pdf_path = FIG_DIR / "regime_grid.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
