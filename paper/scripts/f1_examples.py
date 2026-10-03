"""F1: one structured realization per family.

Columns in Table 1's order. Every pattern is an observed one (replicate 0) of the main evaluation's clouds
with n in N_RANGE, so panels have similar counts, chosen by rule, not by eye: the 90th percentile of the
regime coordinate x (common.x_of; cell: k) over the clouds beyond the tau = 0.9 boundary (cell: k >= 5).
Poisson: the cloud with n closest to the middle of N_RANGE.

    python paper/scripts/f1_examples.py      # -> paper/figs/f1_examples.pdf
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C  # first: it sets the bank (paper config `data`) before cloudforger reads it

from cloudforger.scores.simulate import observed  # noqa: E402

ORDER = ["poisson", "thomas", "nested", "ring", "lgcp", "matern2", "strauss", "cell"]
STRUCTURED = {"poisson": "all", "cell": "k 5-30"}
N_RANGE = (200, 450)
Q = 0.9


def pick(cl: pd.DataFrame, family: str) -> pd.Series:
    """The cloud of the family's structured stratum (n in N_RANGE) at quantile Q of x; Poisson: n closest
    to mid-range."""
    g = cl[(cl.family == family) & (cl.stratum == STRUCTURED.get(family, "above 0.9")) & cl.n.between(*N_RANGE)]
    if family == "poisson":
        return g.iloc[np.argmin(np.abs(g.n - np.mean(N_RANGE)))]
    order = np.argsort(g.x.to_numpy(), kind="stable")
    return g.iloc[order[int(round(Q * (len(g) - 1)))]]


def main() -> None:
    C.style()
    r, cuts = C.rows(), C.cutoffs()
    cl = pd.read_csv(C.evaluation("main") / "clouds.csv")
    cl = cl[cl.variant == C.cfg()["headline_variant"]].copy()
    cl["n"] = r.n.reindex(cl.fit_case_id).to_numpy()
    cl["x"] = np.nan
    for f, g in cl.groupby("family"):
        rf = r.loc[g.fit_case_id]
        cl.loc[g.index, "x"] = C.x_of(rf, f, cuts) if f in cuts else rf.k.to_numpy(float)
    fams = [f for f in ORDER if f in C.FAMILIES]
    chosen = {f: pick(cl, f).fit_case_id for f in fams}
    pts = observed(r.loc[list(chosen.values())])
    fig, axes = plt.subplots(1, len(fams), figsize=(C.TEXT_WIDTH, 1.05),
                             gridspec_kw={"wspace": 0.08, "left": 0.0, "right": 1.0, "bottom": 0.1, "top": 0.86})
    for ax, f in zip(axes, fams):
        x = pts[chosen[f]]
        ax.scatter(x[:, 0], x[:, 1], s=0.8, c=C.INK, linewidths=0, rasterized=True)
        ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[], aspect="equal")
        ax.axis("off")                                          # no frame: the pattern is the figure
        ax.text(0.5, -0.05, f"$n = {len(x)}$", transform=ax.transAxes, va="top", ha="center", fontsize=7,
                color=C.INK2)
        ax.set_title(C.PLOT_NAMES[f], pad=3)
    C.save(fig, "f1_examples")


if __name__ == "__main__":
    main()
