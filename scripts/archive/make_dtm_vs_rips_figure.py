#!/usr/bin/env python3
# scripts/make_dtm_vs_rips_figure.py
"""Build the "Rips vs. DTM_5" figure requested by the \\figtodo in
docs/final_report.tex Section 1.4 (Persistent homology): one hand-picked
clustered cloud -- two dense blobs joined by a single sparse two-point
"bridge" -- shown at one intermediate filtration value under each
filtration. Leftmost panel: the raw cloud itself (r=0, no balls yet), so the
two right-hand columns can be read as "what each filtration has done to
this" rather than requiring the reader to reconstruct it from the balls
panels alone. Middle column: balls around each point at that value; right
column: the real H0 diagram, computed with the repo's own
RipsFiltration/DTMFiltration classes (ripser / gudhi's DTMRipsComplex, the
same code path scripts/featurize.py uses), not reimplemented here.

The cloud is constructed so that the bridge is the *shortest path* between
the two blobs (~1.08) while the direct blob-to-blob gap is far larger
(~2.58): Rips only sees raw distance, so it closes the two-blob merge as
soon as r clears the bridge's longest link. DTM_5 additionally weights each
point by its distance-to-measure (RMS distance to its 5 nearest
neighbours); the two bridge points sit in a sparse neighbourhood so their
weight is large, which delays their balls' growth. At the shared
intermediate r used for both left panels, Rips has already fully merged into one component
(verified below) while DTM_5 still has four: the two blobs plus each bridge
point on its own -- exactly the mechanism section ssec:persistence
describes and Table tab:branches's Rips-vs-DTM gap reflects.

The left-panel ball radius for DTM, rad_i(r) = max(0, r - w_i) / 2 with w_i
the point's DTM_5 weight, is a simplified illustrative convention (linear
in r - w_i), not gudhi's internal weighted-Rips edge formula -- adequate
for a qualitative picture, per the figtodo's own "intuition-building, not
publication-rigorous" framing. The right-panel diagrams are the real thing.

Usage:
    python scripts/make_dtm_vs_rips_figure.py

Output:
    figs/dtm_vs_rips.png
    figs/dtm_vs_rips.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.core.cloud import PointCloud  # noqa: E402
from cloudforger.data_generation.filtration.rips import RipsFiltration  # noqa: E402
from cloudforger.data_generation.filtration.dtm import DTMFiltration  # noqa: E402

FIG_DIR = ROOT / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

K = 5           # DTM_k, matches Table tab:branches's DTM_5 arm
R_BALLS = 1.3   # shared intermediate filtration value for the left panels

BLOB_COLOR = "#2a78d6"
BRIDGE_COLOR = "#8a8a86"
HIGHLIGHT_COLOR = "#d6472a"


def build_cloud() -> tuple[np.ndarray, np.ndarray]:
    """Two Gaussian blobs plus a two-point sparse bridge. Returns (points,
    is_bridge) with is_bridge a boolean mask for coloring."""
    rng = np.random.default_rng(0)
    blob_a = rng.normal(loc=[0.0, 0.0], scale=0.16, size=(22, 2))
    blob_b = rng.normal(loc=[3.0, 0.0], scale=0.16, size=(22, 2))
    bridge = np.array([[1.25, 0.03], [1.72, -0.04]])
    points = np.vstack([blob_a, bridge, blob_b])
    is_bridge = np.zeros(len(points), dtype=bool)
    is_bridge[22:24] = True
    return points, is_bridge


def dtm_weights(points: np.ndarray, k: int) -> np.ndarray:
    """Each point's DTM_k value: RMS distance to its k nearest neighbours
    (matches DTMFiltration's own docstring definition)."""
    dist = cdist(points, points)
    np.fill_diagonal(dist, np.inf)
    nearest_k = np.sort(dist, axis=1)[:, :k]
    return np.sqrt((nearest_k ** 2).mean(axis=1))


def union_find_components(points: np.ndarray, radius_fn) -> np.ndarray:
    """Connected components of the "balls touch" graph at one filtration
    value, radius_fn(i) giving point i's ball radius. Used only to sanity
    check / drive the ball-panel coloring, not for the diagrams themselves."""
    n = len(points)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    dist = cdist(points, points)
    radii = np.array([radius_fn(i) for i in range(n)])
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] <= radii[i] + radii[j]:
                union(i, j)

    return np.array([find(i) for i in range(n)])


def draw_raw_cloud(ax, points: np.ndarray, is_bridge: np.ndarray, title: str):
    """The cloud itself, r=0, no balls -- what both filtrations start from."""
    colors = np.where(is_bridge, BRIDGE_COLOR, BLOB_COLOR)
    ax.scatter(points[:, 0], points[:, 1], s=16, c=colors, zorder=3, linewidths=0)
    ax.scatter([], [], s=16, c=BLOB_COLOR, label="blob")
    ax.scatter([], [], s=16, c=BRIDGE_COLOR, label="bridge")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=2, frameon=False, fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-0.7, 3.7)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_balls(ax, points: np.ndarray, is_bridge: np.ndarray, radius_fn, title: str):
    colors = np.where(is_bridge, BRIDGE_COLOR, BLOB_COLOR)
    for (x, y), c, r in zip(points, colors, [radius_fn(i) for i in range(len(points))]):
        if r > 0:
            ax.add_patch(Circle((x, y), r, facecolor=c, edgecolor=c, alpha=0.22, linewidth=0.8))
    ax.scatter(points[:, 0], points[:, 1], s=10, c=colors, zorder=3, linewidths=0)

    n_components = len(np.unique(union_find_components(points, lambda i: radius_fn(i))))
    ax.set_title(f"{title}\n({n_components} component{'s' if n_components != 1 else ''} at this $r$)", fontsize=10)
    ax.set_xlim(-0.7, 3.7)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_diagram(ax, diagram: np.ndarray, title: str):
    finite = diagram[np.isfinite(diagram[:, 1])]
    infinite = diagram[~np.isfinite(diagram[:, 1])]

    top = float(finite[:, 1].max()) * 1.25 if len(finite) else 1.0
    lo = min(0.0, float(diagram[:, 0].min())) if len(diagram) else 0.0

    ax.plot([lo, top], [lo, top], color="#c9c8c3", lw=1, zorder=1)

    # The A/B merge is the longest-lived finite bar (the last structural
    # merge before the single infinite-persistence component) -- highlight
    # it, the event the left panel is a snapshot of.
    if len(finite):
        merge_idx = np.argmax(finite[:, 1] - finite[:, 0])
        others = np.delete(finite, merge_idx, axis=0)
        ax.scatter(others[:, 0], others[:, 1], s=14, c=BLOB_COLOR, alpha=0.6, zorder=2, linewidths=0)
        b, d = finite[merge_idx]
        ax.scatter([b], [d], s=70, facecolors="none", edgecolors=HIGHLIGHT_COLOR, linewidths=1.8, zorder=4)

    if len(infinite):
        # Plotted at the panel's right edge rather than its true birth x, so
        # it doesn't sit on top of the highlighted merge marker at top-left.
        ax.scatter([top] * len(infinite), [top] * len(infinite), s=40, marker="^", c="black", zorder=3, clip_on=False)

    ax.set_title(title, fontsize=10)
    ax.set_xlim(lo, top)
    ax.set_ylim(lo, top)
    ax.set_xlabel("birth", fontsize=9)
    ax.set_ylabel("death", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    points, is_bridge = build_cloud()
    weights = dtm_weights(points, K)

    cloud = PointCloud(points=points, generator_name="illustrative_bridge")
    rips_diagram = RipsFiltration(maxdim=0).compute(cloud).diagrams[0]
    dtm_diagram = DTMFiltration(maxdim=0, k=K).compute(cloud).diagrams[0]

    fig = plt.figure(figsize=(11.6, 8.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.0, 1.15])

    ax_cloud = fig.add_subplot(gs[0, 0])
    ax_note = fig.add_subplot(gs[1, 0])
    ax_rips_balls = fig.add_subplot(gs[0, 1])
    ax_rips_diag = fig.add_subplot(gs[0, 2])
    ax_dtm_balls = fig.add_subplot(gs[1, 1])
    ax_dtm_diag = fig.add_subplot(gs[1, 2])

    rips_radius = lambda i: R_BALLS / 2.0  # noqa: E731
    dtm_radius = lambda i: max(0.0, R_BALLS - weights[i]) / 2.0  # noqa: E731

    draw_raw_cloud(ax_cloud, points, is_bridge, "initial point cloud ($r=0$)")

    ax_note.axis("off")
    ax_note.text(
        0.5, 0.9,
        "Two dense blobs joined by a sparse\ntwo-point bridge: the shortest path\n"
        "between the blobs, but through a\nneighbourhood far sparser than either\n"
        "blob's interior.",
        transform=ax_note.transAxes, ha="center", va="top", fontsize=9, color="#3a3a38",
    )

    draw_balls(ax_rips_balls, points, is_bridge, rips_radius, "Rips, balls of radius $r/2$")
    draw_diagram(ax_rips_diag, rips_diagram, "Rips $H_0$")

    draw_balls(ax_dtm_balls, points, is_bridge, dtm_radius, rf"DTM$_{K}$, balls of radius $(r-w_i)_+/2$")
    draw_diagram(ax_dtm_diag, dtm_diagram, rf"DTM$_{K}$ $H_0$")

    fig.suptitle(f"Same cloud, same intermediate value $r={R_BALLS:g}$", fontsize=11, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    png_path = FIG_DIR / "dtm_vs_rips.png"
    pdf_path = FIG_DIR / "dtm_vs_rips.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
