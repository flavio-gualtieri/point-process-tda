"""The regime boundary per family, for the appendix: F4 unpooled, and the boundary in the physical parameters.

s_regime_detect -- one column per non-Poisson family (Table 1's order, cell last).
    top     detection power on the test split against x (common.x_of; cell: k) in quantile bins: the CSR
            test and f4's DETECTORS coloured, every other classifier of the run as a thin grey line, all
            as tests of the cutoffs' size (common.detected).
    bottom  gain over CSR per evaluation stratum, as a share of the true model's in the family's strongest
            stratum (f4_regime.gain_shares), at the stratum's median x (cell: median k). 95% bootstrap.
s_regime_map -- one panel per family with a coordinate: share of test patterns the reference detector
    (the run's regime.classifier, a test of the cutoffs' size) does NOT flag, binned over (nbar, coordinate)
    on log axes, with the boundaries u*(0.5) (solid) and u*(0.9) (dashed), straight lines there. Axes
    are cut to the central 99% of the coordinate.

    python paper/scripts/s_regime.py         # -> paper/figs/s_regime_detect.pdf, s_regime_map.pdf
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

import common as C  # first: it sets the bank (paper config `data`) before cloudforger reads it
import f4_regime as F4

from cloudforger.pipeline.regime import coordinate  # noqa: E402

ORDER = ["thomas", "nested", "ring", "lgcp", "matern2", "strauss", "cell"]
COORD_LABEL = {"omega": r"$\omega$", "omega_inner": r"$\omega_{\rm in}$", "zeta": r"$\zeta$"}
CELL_STRATA = ["k 2-2", "k 3-4", "k 5-30"]
BINS = 16
XLIM = (-5, 4)


def k_axis(ax) -> None:
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([2, 5, 10, 30]))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())


def detect() -> None:
    C.style()
    r, cuts, names = C.rows(), C.cutoffs(), C.cfg().get("names", {})
    fams = [f for f in ORDER if f in C.FAMILIES]
    test = r[r.split == "test"]
    others = [m for m in C.run_config()["classify"] if m not in F4.DETECTORS]
    fig, axes = plt.subplots(2, len(fams), figsize=(C.TEXT_WIDTH, 3.0), layout="constrained")
    for j, f in enumerate(fams):
        t = test[test.family == f]
        x = C.x_of(t, f, cuts) if f in cuts else t.k.to_numpy(float)
        ax = axes[0, j]
        if f in cuts:
            at = np.clip(np.searchsorted(np.quantile(x, np.linspace(0, 1, BINS + 1)), x, side="right") - 1, 0, BINS - 1)
            groups = [at == b for b in range(BINS)]
        else:
            groups = [x == k for k in np.unique(x)]
        centre = [np.median(x[g]) for g in groups]
        for model in others + F4.DETECTORS:
            y = C.detected(model).reindex(t.index).to_numpy(float)
            coloured = model in F4.DETECTORS
            ax.plot(centre, [y[g].mean() for g in groups], zorder=3 if coloured else 2,
                    color=F4.COLOURS.get(model, C.MUTED), lw=1.3 if coloured else 0.5, alpha=1 if coloured else 0.6)
        ax.set(ylim=(0, 1.02), yticks=[0, 0.5, 1], title=C.PLOT_NAMES[f])
        # bottom: gain over CSR per stratum
        strata = F4.STRATA if f in cuts else CELL_STRATA
        series, meta, est, boot = F4.gain_shares({f: strata})
        lo, hi = np.percentile(boot[:, 0], [2.5, 97.5], axis=0)
        xm = (C.x_of(r.loc[meta.fit_case_id], f, cuts) if f in cuts else r.k[meta.fit_case_id].to_numpy(float))
        pos = np.array([np.median(xm[(meta.stratum == s).to_numpy()]) for s in strata])
        ax = axes[1, j]
        for i, name in enumerate(series):
            off = pos * (1 + (i - 1) * 0.08) if f == "cell" else pos + (i - 1) * 0.15
            ax.errorbar(off, est[0, :, i], yerr=[est[0, :, i] - lo[:, i], hi[:, i] - est[0, :, i]],
                        color=F4.GAIN_COLOURS[i], marker="o", ms=3, mec="white", mew=0.4, lw=1.2, elinewidth=0.8,
                        capsize=0, zorder=3 + i)
        ax.axhline(0, color=C.AXIS, lw=0.6)
        for i in (0, 1):
            if f in cuts:
                for v, ls in ((0, "-"), (1, "--")):
                    axes[i, j].axvline(v, color=C.INK2, lw=0.6, ls=ls, zorder=4)
                axes[i, j].set_xlim(XLIM[0], max(XLIM[1], pos.max() + 1))     # Matérn II's strong stratum sits far out
            else:
                k_axis(axes[i, j])
                axes[i, j].set_xlim(1.8, 33)
        axes[0, j].tick_params(labelbottom=False)
        axes[1, j].set_xlabel("$k$" if f == "cell" else F4.XLABEL, labelpad=1)
        if j:
            axes[0, j].tick_params(labelleft=False)
    axes[0, 0].set_ylabel("Detection power")
    axes[1, 0].set_ylabel("Relative gain\nover CSR")
    # top row's series, then the bottom row's; ParamNet is the same blue in both, so it is listed once
    entries = {F4.LABELS.get(m, names.get(m, m)): F4.COLOURS[m] for m in F4.DETECTORS}
    entries["Other classifiers"] = C.MUTED
    entries |= {name: c for name, c in zip(series, F4.GAIN_COLOURS) if name not in entries}
    handles = [Patch(facecolor=c, edgecolor="none") for c in entries.values()]
    fig.legend(handles, list(entries), loc="outside lower center", ncol=len(entries), handlelength=0.8,
               handleheight=0.8, handletextpad=0.4, columnspacing=1.2)
    C.save(fig, "s_regime_detect")


def regime_map() -> None:
    C.style()
    ref = C.run_config()["regime"]["classifier"]
    cuts = C.cutoffs()
    flagged = C.detected(ref)
    r = C.rows().loc[flagged.index]
    fams = [f for f in ORDER if f in cuts]
    fig, axes = plt.subplots(1, len(fams), figsize=(C.TEXT_WIDTH, 1.75), layout="constrained")
    im = None
    for ax, f in zip(axes.flat, fams):
        m = (r.family == f).to_numpy()
        rf, cf = r[m], ~flagged[m].to_numpy()
        c = cuts[f]
        x, nbar = coordinate(rf, f, c["coordinate"]), rf.nbar.to_numpy(float)
        lo, hi = np.quantile(x, [0.005, 0.995])
        im = ax.hexbin(nbar, x, C=cf.astype(float), reduce_C_function=np.mean, gridsize=20, mincnt=5,
                       xscale="log", yscale="log", extent=(np.log10(nbar.min()), np.log10(nbar.max()),
                                                           np.log10(lo), np.log10(hi)),
                       cmap=C.blues(), vmin=0, vmax=1, linewidths=0, rasterized=True)
        grid = np.geomspace(nbar.min(), nbar.max(), 50)
        for tau, ls in (("0.5", "-"), ("0.9", "--")):
            y = np.exp(c["u_boundary"][tau] - c["nbar_exponent"] * np.log(grid))
            ax.plot(grid, y, color="white", lw=2.0, ls="-", alpha=0.8)          # halo: legible on dark bins
            ax.plot(grid, y, color=C.INK, lw=0.9, ls=ls)
        ax.set(xlim=(nbar.min(), nbar.max()), ylim=(lo, hi), title=C.PLOT_NAMES[f], xlabel=r"$\bar n$")
        ax.set_ylabel(COORD_LABEL.get(c["coordinate"], c["coordinate"]), labelpad=1)
        C.log_ticks(ax)
        ax.xaxis.set_major_locator(FixedLocator([100, 500]))   # narrow panels: two ticks, not three
    cb = fig.colorbar(im, ax=axes, location="right", shrink=0.8, aspect=18, pad=0.01, ticks=[0, 0.5, 1])
    cb.set_label("Not detected", color=C.INK2, labelpad=2)
    cb.outline.set_visible(False)
    fig.legend([Line2D([], [], color=C.INK, lw=0.9), Line2D([], [], color=C.INK, lw=0.9, ls="--")],
               [r"$u^\star(0.5)$", r"$u^\star(0.9)$"], loc="outside lower center", ncol=2,
               handlelength=2.2, borderaxespad=0)
    C.save(fig, "s_regime_map")


def main() -> None:
    detect()
    regime_map()


if __name__ == "__main__":
    main()
