#!/usr/bin/env python3
# scripts/make_calibration_comparison_figure.py
"""Build the "naive min/max vs coverage-quantile calibration" figure
requested by the \\fig{} TODO in writeup/writeup.tex Section 6.2
(Calibration): the same H1 diagram rendered under the two calibration
schemes, side by side, to make the "loss of texture" argument -- currently
carried entirely by prose -- visual.

Population: 4000 clouds drawn from the SAME reparametrized design
distribution used in production (K=kappa, EN=kappa*mu, c=nu, log-uniform
over the exact ranges/derived-fields/constraint of
configs/runs/thomas/thomas_pi_multik.yaml), not a hand-picked pair of
triples -- so the "some diagram has an extreme outlier relative to the rest
of the training population" premise the text makes is demonstrated on a
realistic sample of that population, rather than asserted. The naive
min/max bound and axis_bounds() (the real per-diagram coverage-quantile
calibrator, src/cloudforger/calibration) are both computed on this same
population; PersistenceImager (also production code) renders the one
most-textured non-outlier ("typical") diagram under each resulting bound. No
TDA math is reimplemented here.

Usage:
    python scripts/make_calibration_comparison_figure.py

Output:
    figs/calibration_comparison.png
    figs/calibration_comparison.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.calibration import axis_bounds  # noqa: E402
from cloudforger.core.region import Box  # noqa: E402
from cloudforger.data_generation.design import (  # noqa: E402
    apply_derived,
    passes,
    sample_param_vector,
)
from cloudforger.data_generation.filtration.rips import RipsFiltration  # noqa: E402
from cloudforger.data_generation.point_processes.thomas import ThomasProcess  # noqa: E402
from cloudforger.vectorization.persistence_images.persistence_image import (  # noqa: E402
    PersistenceImager,
)

FIG_DIR = ROOT / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Verbatim from configs/runs/thomas/thomas_pi_multik.yaml's process.design.random
# block -- the actual production design distribution for the Thomas family,
# reused rather than re-derived so this figure's population matches what the
# calibration is really fit on.
RANGES = {
    "K": {"low": 15.0, "high": 120.0, "scale": "log"},
    "EN": {"low": 150.0, "high": 800.0, "scale": "log"},
    "c": {"low": 0.10, "high": 0.90, "scale": "log"},
}
DERIVED = {
    "parent_intensity": "K",
    "mean_offspring": "EN / K",
    "cluster_scale": "c / (2 * sqrt(K))",
}
CONSTRAINTS = ["mean_offspring >= 2.5"]

N_POPULATION = 4000
SEED = 20260818
COVERAGE, PAD = 0.95, 1.05  # the production defaults quoted in the text (q=0.95, alpha=1.05)
RESOLUTION = 64  # matches Definition per_image / production configs

BLUE = "#2a78d6"
ORANGE = "#eb6834"
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


def sample_population():
    rng = np.random.default_rng(SEED)
    W = Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]))
    filtration = RipsFiltration(maxdim=1)
    diagrams, params = [], []
    seed_ctr = 0
    while len(diagrams) < N_POPULATION:
        p = apply_derived(sample_param_vector(RANGES, rng), DERIVED)
        if not passes(p, CONSTRAINTS):
            continue
        proc = ThomasProcess(parent_intensity=p["parent_intensity"],
                              mean_offspring=p["mean_offspring"],
                              cluster_scale=p["cluster_scale"])
        cloud = proc.sample(region=W, seed=600_000 + seed_ctr)
        seed_ctr += 1
        diagrams.append(filtration.compute(cloud))
        params.append(p)
    return diagrams, params


def naive_bounds(diagrams, homology_dim=1):
    """Literal pooled min/max over every point of every diagram -- the
    "naive" scheme the text contrasts with axis_bounds' per-diagram quantile."""
    births, perss = [], []
    for d in diagrams:
        pairs = d.finite_pairs(homology_dim)
        if len(pairs) == 0:
            continue
        births.append(pairs[:, 0])
        perss.append(pairs[:, 1] - pairs[:, 0])
    births, perss = np.concatenate(births), np.concatenate(perss)
    return (float(births.min()), float(births.max())), (0.0, float(perss.max()))


def pick_typical(diagrams, homology_dim=1, bulk_percentile=60.0):
    """Among the non-outlier bulk of the population (own max persistence at
    or below `bulk_percentile` -- i.e. NOT the diagram setting the naive
    bound), pick the one with the most H1 features. A sparse bulk diagram
    would show the axis-compression effect too, but with only 1-2 blobs to
    begin with there is little "texture" left to visibly lose; the
    richest-textured ordinary diagram makes the naive/quantile contrast
    legible at a glance rather than merely present in the numbers."""
    max_pers = np.array([
        (d.finite_pairs(homology_dim)[:, 1] - d.finite_pairs(homology_dim)[:, 0]).max()
        if len(d.finite_pairs(homology_dim)) else 0.0
        for d in diagrams
    ])
    n_pairs = np.array([len(d.finite_pairs(homology_dim)) for d in diagrams])
    cutoff = np.percentile(max_pers, bulk_percentile)
    eligible = np.where(max_pers <= cutoff)[0]
    return int(eligible[np.argmax(n_pairs[eligible])])


def plot_box_panel(ax, diagram, birth_range, pers_range, title, box_color, crop=None):
    pairs = diagram.finite_pairs(1)
    b, pers = pairs[:, 0], pairs[:, 1] - pairs[:, 0]
    ax.scatter(b, pers, s=22, color=BLUE, edgecolor="black", linewidth=0.4, zorder=3)
    rect = mpatches.Rectangle((birth_range[0], pers_range[0]),
                               birth_range[1] - birth_range[0], pers_range[1] - pers_range[0],
                               fill=False, edgecolor=box_color, linewidth=2, zorder=2)
    ax.add_patch(rect)
    if crop is not None:
        (cb_lo, cb_hi), (cp_lo, cp_hi) = crop
        zoom = mpatches.Rectangle((cb_lo, cp_lo), cb_hi - cb_lo, cp_hi - cp_lo,
                                   fill=False, edgecolor=GRAY, linewidth=1.2, linestyle="--", zorder=4)
        ax.add_patch(zoom)
    pad_x = 0.05 * (birth_range[1] - birth_range[0])
    pad_y = 0.05 * (pers_range[1] - pers_range[0])
    ax.set_xlim(birth_range[0] - pad_x, birth_range[1] + pad_x)
    ax.set_ylim(pers_range[0] - pad_y, pers_range[1] + pad_y)
    ax.set_xlabel("birth", fontsize=8)
    ax.set_ylabel("persistence", fontsize=8)
    ax.set_title(title, fontsize=10)


def plot_pi_panel(ax, diagram, birth_range, pers_range, title, crop):
    """Render the full-resolution PI, then zoom the DISPLAY to the same
    physical (birth, persistence) window in both panels (the dashed box in
    the row above), using nearest-neighbor interpolation so actual pixels
    are shown, not smoothed further. Because the naive bound spans a wider
    physical range on the same 64x64 grid, this shared physical crop covers
    fewer, larger pixels there than in the quantile-calibrated image -- the
    resolution loss the text describes, made visible instead of asserted."""
    imager = PersistenceImager(birth_range=birth_range, pers_range=pers_range,
                                resolution=RESOLUTION, sigma_pixels=2.0)
    im = imager.transform(diagram, dim=1)
    # transform() already flips its array so persistence increases upward
    # under imshow's default origin="upper" (row 0 = max persistence); passing
    # origin="lower" here would double-flip it and, combined with `extent`,
    # point the crop window at the wrong physical region of the image.
    ax.imshow(im, cmap="magma", origin="upper", aspect="auto", interpolation="nearest",
              extent=[birth_range[0], birth_range[1], pers_range[0], pers_range[1]])
    (cb_lo, cb_hi), (cp_lo, cp_hi) = crop
    ax.set_xlim(cb_lo, cb_hi)
    ax.set_ylim(cp_lo, cp_hi)
    n_px_shown = (cb_hi - cb_lo) / (birth_range[1] - birth_range[0]) * RESOLUTION
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{title}\n(~{n_px_shown:.0f} px across the crop below)", fontsize=10)


def main():
    diagrams, params = sample_population()
    n_birth, n_pers = naive_bounds(diagrams, homology_dim=1)
    q_birth, q_pers = axis_bounds(diagrams, homology_dim=1, coverage=COVERAGE, pad=PAD)

    typ_idx = pick_typical(diagrams, homology_dim=1)
    typical = diagrams[typ_idx]

    outlier_idx = int(np.argmax([
        (d.finite_pairs(1)[:, 1] - d.finite_pairs(1)[:, 0]).max() if len(d.finite_pairs(1)) else 0.0
        for d in diagrams
    ]))
    outlier_params = params[outlier_idx]

    # Shared physical zoom window: the typical diagram's own footprint plus a
    # margin, identical in both panels below -- so the two rasters differ
    # only in how many of the fixed 64x64 grid's pixels land inside it.
    typ_pairs = typical.finite_pairs(1)
    tb, tp = typ_pairs[:, 0], typ_pairs[:, 1] - typ_pairs[:, 0]
    crop = ((max(0.0, tb.min() - 0.02), tb.max() + 0.03), (0.0, tp.max() + 0.03))

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 8.4))

    plot_box_panel(axes[0, 0], typical, n_birth, n_pers,
                    "Naive min/max\n(pooled over all points, all diagrams)", ORANGE, crop=crop)
    plot_box_panel(axes[0, 1], typical, q_birth, q_pers,
                    f"Coverage-quantile ($q$={COVERAGE}, $\\alpha$={PAD})\n"
                    "(per-diagram statistic, quantiled over diagrams)", BLUE, crop=crop)
    plot_pi_panel(axes[1, 0], typical, n_birth, n_pers,
                   "PI under naive bound, zoomed to dashed box", crop)
    plot_pi_panel(axes[1, 1], typical, q_birth, q_pers,
                   "PI under quantile bound, zoomed to dashed box", crop)

    fig.suptitle(
        "Same $H_1$ diagram, two calibrations: the naive bound is stretched by an\n"
        f"outlier diagram elsewhere in the population "
        f"($\\mu$={outlier_params['mean_offspring']:.1f} offspring/parent, not shown), "
        "compressing this diagram's own texture",
        fontsize=10.5, y=1.02,
    )
    fig.tight_layout()
    png_path = FIG_DIR / "calibration_comparison.png"
    pdf_path = FIG_DIR / "calibration_comparison.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"naive birth={n_birth} pers={n_pers}")
    print(f"quantile birth={q_birth} pers={q_pers}")
    print(f"typical diagram idx={typ_idx} own max persistence="
          f"{(typical.finite_pairs(1)[:, 1] - typical.finite_pairs(1)[:, 0]).max():.4f}")


if __name__ == "__main__":
    main()
