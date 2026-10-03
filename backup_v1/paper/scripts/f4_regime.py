"""F4: where each family is called Poisson, with the fitted regime boundaries.

Per non-Poisson family: P(called poisson) -- the share of patterns the reference classifier (the
run's regime.classifier, on every row it has predictions for: out-of-fold train, val, test) calls
poisson -- binned over (nbar, x), x the family's regime coordinate, both on log axes. The regime
rule is a straight line there: log x + a log nbar = u*(tau), drawn for tau = 0.5 (solid) and 0.9
(dashed). Cell has no monotone coordinate, so its panel is P(called poisson) against k.

    python paper/scripts/f4_regime.py        # -> paper/figs/f4_regime.pdf
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from cloudforger.pipeline.regime import coordinate

import common as C

COORD_LABEL = {"omega": r"$\omega$", "omega_inner": r"$\omega_{\rm in}$", "zeta": r"$\zeta$"}


def main() -> None:
    C.style()
    ref = C.run_config()["regime"]["classifier"]
    post = C.classifier(ref)
    called = post[C.FAMILIES].to_numpy().argmax(1) == C.FAMILIES.index("poisson")
    r = C.rows().loc[post.index]
    cuts = C.cutoffs()
    fams = [f for f in C.FAMILIES if f != "poisson"]
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.3), gridspec_kw={"wspace": 0.45, "hspace": 0.55})
    im = None
    for ax, f in zip(axes.flat, fams):
        m = (r.family == f).to_numpy()
        rf, cf = r[m], called[m]
        if f not in cuts:                                       # cell: against k
            k = rf.k.to_numpy()
            ks = np.unique(k)
            ax.plot(ks, [cf[k == v].mean() for v in ks], color=C.SERIES[0], marker="o", ms=2.5)
            ax.set(xlabel="$k$", ylabel="P(called Poisson)", ylim=(0, 1), title=C.NAMES[f])
            continue
        c = cuts[f]
        x, nbar = coordinate(rf, f, c["coordinate"]), rf.nbar.to_numpy(float)
        im = ax.hexbin(nbar, x, C=cf.astype(float), reduce_C_function=np.mean, gridsize=26, mincnt=5,
                       xscale="log", yscale="log", cmap=C.blues(), vmin=0, vmax=1, linewidths=0)
        grid = np.geomspace(nbar.min(), nbar.max(), 50)
        for tau, ls in (("0.5", "-"), ("0.9", "--")):
            ub = c["u_boundary"].get(tau)
            if ub is not None:
                ax.plot(grid, np.exp(ub - c["nbar_exponent"] * np.log(grid)), color=C.INK, lw=0.9, ls=ls)
        ax.set(xlim=(nbar.min(), nbar.max()), ylim=(x.min(), x.max()), title=C.NAMES[f],
               xlabel=r"$\bar n$", ylabel=COORD_LABEL.get(c["coordinate"], c["coordinate"]))
        C.log_ticks(ax)
    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    if im is not None:                                          # colorbar on the left, legend on the right
        cb = fig.colorbar(im, cax=legend_ax.inset_axes([0.0, 0.05, 0.08, 0.9]))
        cb.set_label("P(called Poisson)", color=C.INK2)
        cb.outline.set_visible(False)
    legend_ax.legend([Line2D([], [], color=C.INK, lw=0.9), Line2D([], [], color=C.INK, lw=0.9, ls="--")],
                     [r"boundary $\tau = 0.5$", r"boundary $\tau = 0.9$"], loc="center left",
                     bbox_to_anchor=(0.45, 0.5))
    C.save(fig, "f4_regime")


if __name__ == "__main__":
    main()
