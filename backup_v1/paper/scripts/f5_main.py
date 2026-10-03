"""F5: detection, parameter error and the true model's gain over CSR, against the regime coordinate.

One column per non-Poisson family, x = u = log x + a log nbar (compare's frozen cutoffs; cell: k),
binned by quantile over the family's test clouds:
    (a) detection rate -- share NOT called poisson -- of every classifier of the run
    (b) estimator error RMSE(log theta)/s.d. (s.d. over all the family's test clouds, mean over
        targets) of every estimator of the run
    (c) oracle gain S(csr) - S(true) of the headline evaluation's clouds, mean +/- s.e. per bin
Detection and error share one y-scale across families, the gain has one per panel. Models in
paper.yaml `highlight` are coloured (fixed order); every other model is a thin grey line,
so the figure shows how closely all of them track the same boundary. Vertical lines: u*(0.5), u*(0.9).

    python paper/scripts/f5_main.py          # -> paper/figs/f5_main.pdf
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from cloudforger.pipeline.core import TARGETS

import common as C

BINS, GAIN_BINS = 12, 6


def binned(x: np.ndarray, y: np.ndarray, edges: np.ndarray, stat) -> tuple[np.ndarray, np.ndarray]:
    at = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    centres, vals = [], []
    for b in range(len(edges) - 1):
        m = at == b
        if m.sum() >= 5:
            centres.append(np.median(x[m]))
            vals.append(stat(y[m]))
    return np.array(centres), np.array(vals)


def line(ax, x, y, model, highlight):
    if model in highlight:
        ax.plot(x, y, color=C.SERIES[highlight.index(model)], lw=1.2, zorder=3)
    else:
        ax.plot(x, y, color=C.MUTED, lw=0.5, alpha=0.6, zorder=2)


def main() -> None:
    C.style()
    rc, cuts, hl = C.run_config(), C.cutoffs(), C.cfg()["highlight"][:3]
    r = C.rows()
    test = r[(r.split == "test")]
    fams = [f for f in C.FAMILIES if f != "poisson"]
    xs = {f: (C.u_of(test[test.family == f], f, cuts) if f in cuts else test[test.family == f].k.to_numpy(float))
          for f in fams}
    edges = {f: np.unique(np.quantile(xs[f], np.linspace(0, 1, BINS + 1))) for f in fams}

    # detection and error share a scale across families; the gain differs ~10x between them, so each
    # gain panel keeps its own
    fig, axes = plt.subplots(3, len(fams), figsize=(7.0, 4.0), gridspec_kw={"wspace": 0.35, "hspace": 0.28})
    for i in (0, 1):
        for ax in axes[i, 1:]:
            ax.sharey(axes[i, 0])
            ax.tick_params(labelleft=False)
    # (a) detection
    for model in rc["classify"]:
        post = C.classifier(model)
        post = post[post.split == "test"]
        detected = pd.Series(post[C.FAMILIES].to_numpy().argmax(1) != C.FAMILIES.index("poisson"), index=post.index)
        for j, f in enumerate(fams):
            ids = test.index[test.family == f]
            line(axes[0, j], *binned(xs[f], detected.reindex(ids).to_numpy(float), edges[f], np.mean), model, hl)
    # (b) estimator error
    for model in rc["estimate"]:
        for j, f in enumerate(fams):
            t = test[test.family == f]
            y = np.log(t[TARGETS[f]].to_numpy(float))
            e2 = (np.log(C.estimator(f, model).reindex(t.index)[TARGETS[f]].to_numpy()) - y) ** 2
            sd = y.std(0)
            stat = lambda v: float(np.mean(np.sqrt(np.nanmean(v, 0)) / sd))
            line(axes[1, j], *binned(xs[f], e2, edges[f], stat), model, hl)
    # (c) oracle gain, from the headline evaluation's clouds (one row per cloud)
    ev = pd.read_csv(C.evaluation("main") / "clouds.csv").drop_duplicates("fit_case_id").set_index("fit_case_id")
    for j, f in enumerate(fams):
        g = ev[ev.family == f]
        rf = r.loc[g.index]
        x = C.u_of(rf, f, cuts) if f in cuts else rf.k.to_numpy(float)
        e = np.unique(np.quantile(x, np.linspace(0, 1, GAIN_BINS + 1)))
        cx, mean = binned(x, g.kernel_gain.to_numpy(), e, np.mean)
        _, se = binned(x, g.kernel_gain.to_numpy(), e, lambda v: v.std(ddof=1) / np.sqrt(len(v)))
        ax = axes[2, j]
        ax.axhline(0, color=C.AXIS, lw=0.6, zorder=1)
        ax.errorbar(cx, mean, yerr=se, color=C.SERIES[0], lw=1.0, marker="o", ms=2.5, capsize=0, zorder=3)
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)   # x10^k offset on top
        ax.yaxis.get_offset_text().set_fontsize(5.5)
    # boundaries, labels
    for j, f in enumerate(fams):
        for i in range(3):
            ax = axes[i, j]
            if f in cuts:
                for tau, ls in (("0.5", "-"), ("0.9", "--")):
                    ub = cuts[f]["u_boundary"].get(tau)
                    if ub is not None:
                        ax.axvline(ub, color=C.INK2, lw=0.6, ls=ls, zorder=1)
            if i < 2:
                ax.tick_params(labelbottom=False)
        axes[0, j].set_title(C.NAMES[f], pad=3)
        axes[2, j].set_xlabel("$u$" if f in cuts else "$k$")
    axes[0, 0].set(ylabel="detection rate", ylim=(0, 1.02))
    axes[1, 0].set(ylabel="RMSE / s.d.")
    axes[2, 0].set_ylabel("oracle gain")
    handles = [Line2D([], [], color=C.SERIES[i], lw=1.2) for i in range(len(hl))]
    handles += [Line2D([], [], color=C.MUTED, lw=0.5), Line2D([], [], color=C.INK2, lw=0.6),
                Line2D([], [], color=C.INK2, lw=0.6, ls="--")]
    fig.legend(handles, [*hl, "other models", r"$u^*(0.5)$", r"$u^*(0.9)$"], loc="lower center",
               ncol=len(handles), bbox_to_anchor=(0.5, 0.93))
    C.save(fig, "f5_main")


if __name__ == "__main__":
    main()
