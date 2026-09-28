"""F1: example realizations, two per family -- one clearly structured, one near CSR.

Test split, replicate 0, n within 50 of N_TARGET so every panel has a similar count. Structured =
well inside the regime (the 90th percentile of distance past the tau = 0.9 boundary, compare's
frozen cutoffs); near CSR = the median out-of-regime pattern at tau = 0.5. Cell has no coordinate,
so k = 2 (paired points) against k >= 20 (near-lattice); Poisson shows two draws. Deterministic.

    python paper/scripts/f1_examples.py      # -> paper/figs/f1_examples.pdf
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from cloudforger.pipeline.regime import in_regime
from cloudforger.scores.simulate import observed

import common as C

N_TARGET = 300


def closest(g):
    return g.index[np.argmin(np.abs(g.n.to_numpy() - N_TARGET))]


def at_quantile(g, depth, q):
    """The pattern of g whose depth is at quantile q (depth aligned with g)."""
    order = np.argsort(depth)
    return g.index[order[int(q * (len(g) - 1))]]


def pick(r, family, cuts):
    t = r[(r.family == family) & (r.split == "test") & (r.rep == 0)]
    if family == "poisson":
        first = closest(t)
        return [first, closest(t.drop(first))]
    if family == "cell":
        return [closest(t[t.k == 2]), closest(t[t.k >= 20])]
    t = t[(t.n - N_TARGET).abs() <= 50]                         # comparable counts across panels
    c = cuts[family]
    sign = 1 if c["direction"] == "increasing" else -1
    depth = sign * (C.u_of(t, family, cuts) - c["u_boundary"]["0.9"])    # > 0: past the tau = 0.9 boundary
    inside, outside = t[depth > 0], ~in_regime({family: c}, t, 0.5)
    # well inside the regime (90th percentile of depth past u*(0.9)); a typical out-of-regime pattern
    return [at_quantile(inside, depth[depth > 0], 0.9), at_quantile(t[outside], depth[outside], 0.5)]


def main() -> None:
    C.style()
    r, cuts = C.rows(), C.cutoffs()
    chosen = {f: pick(r, f, cuts) for f in C.FAMILIES}
    pts = observed(r.loc[[c for p in chosen.values() for c in p]])
    fig, axes = plt.subplots(2, len(C.FAMILIES), figsize=(7.0, 1.95), gridspec_kw={"wspace": 0.08, "hspace": 0.12})
    for j, f in enumerate(C.FAMILIES):
        for i, cid in enumerate(chosen[f]):
            ax = axes[i, j]
            x = pts[cid]
            ax.scatter(x[:, 0], x[:, 1], s=0.6, c=C.INK, linewidths=0, rasterized=True)
            ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[], aspect="equal")
            for side in ("top", "right", "bottom", "left"):
                ax.spines[side].set_visible(True)
                ax.spines[side].set_color(C.AXIS)
            ax.text(0.02, -0.04, f"n = {len(x)}", transform=ax.transAxes, va="top", fontsize=5.5, color=C.INK2)
        axes[0, j].set_title(C.NAMES[f], pad=3)
    axes[0, 0].set_ylabel("structured", color=C.INK2)
    axes[1, 0].set_ylabel("near CSR", color=C.INK2)
    C.save(fig, "f1_examples")


if __name__ == "__main__":
    main()
