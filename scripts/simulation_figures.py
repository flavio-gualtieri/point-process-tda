"""Figures for docs/02_simulation.tex -> docs/figs/."""

from __future__ import annotations

from multiprocessing import get_context
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cloudforger.classical.lfunction import RADII, l_minus_r, l_minus_r_from_excess
from cloudforger.departure.tables import Tables
from cloudforger.simulation.families import FAMILIES, Rules, solve
from cloudforger.simulation.processes import SAMPLERS, lgcp_eigenvalues
from cloudforger.simulation.lgcp_grid import grid_size
from cloudforger.simulation.seeding import PATTERN, case_rng
from cloudforger.simulation.sweep import DATA, Config

OUT = Path(__file__).resolve().parents[1] / "docs" / "figs"
STRUCTURED = ("thomas", "nested", "lgcp", "matern2")
LABEL = {"poisson": "Poisson", "thomas": "Thomas", "nested": "Nested Thomas", "lgcp": "LGCP", "matern2": "Matérn II"}
SERIES = {"thomas": "#2a78d6", "nested": "#eb6834", "lgcp": "#1baf7a", "matern2": "#eda100"}
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
LEVELS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
NBAR = 400.0
SHAPES = {"thomas": {"sigma": 0.03}, "nested": {"sigma2": 0.008, "rho": 5.0, "mu1": 3.0},
          "lgcp": {"s": 0.04}, "matern2": {}}
GALLERY = 2 * 10**9

plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.titlesize": 8, "axes.titlecolor": INK, "legend.frameon": False,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def theta(tables, rules, name, level):
    fam = FAMILIES[name](rules)
    shape = SHAPES[name]
    amp = solve(fam, tables, NBAR, shape, level)
    model = fam.model(NBAR, shape, amp)
    if name == "lgcp":
        model["M"] = grid_size(amp, shape["s"], NBAR, tables)
        model["root_lam"] = lgcp_eigenvalues(amp, shape["s"], model["M"])
    return fam, amp, model


def gallery(tables, rules, root):
    cols = list(reversed(LEVELS)) + [0.0]
    fig, axes = plt.subplots(len(STRUCTURED), len(cols), figsize=(7.2, 4.6))
    for i, name in enumerate(STRUCTURED):
        for j, level in enumerate(cols):
            rng = case_rng(root, "pilot", name if level else "poisson", GALLERY + 10 * j, PATTERN)
            pts = SAMPLERS["poisson"](rng, NBAR) if level == 0 else SAMPLERS[name](rng, **theta(tables, rules, name, level)[2])
            ax = axes[i, j]
            ax.scatter(pts[:, 0], pts[:, 1], s=1.6, c=INK, linewidths=0)
            ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[], aspect="equal")
            ax.grid(False)
            for side in ax.spines.values():
                side.set_visible(True)
                side.set_color(AXIS)
            if i == 0:
                ax.set_title("CSR" if level == 0 else rf"$\tilde\delta={level:g}$")
            if j == 0:
                ax.set_ylabel(LABEL[name])
    fig.savefig(OUT / "02_regime_patterns.pdf")
    plt.close(fig)


def curves(tables, rules):
    _, s0, c95, mask = tables.moments(NBAR)
    r = RADII[mask[0]]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3), sharey=True, layout="constrained")
    for ax, name in zip(axes, STRUCTURED):
        ax.axhspan(-1, 1, color=GRID, alpha=0.6, linewidth=0)
        ax.axhline(0, color=AXIS, linewidth=0.8)
        for level, color in zip(LEVELS, RAMP):
            fam, amp, _ = theta(tables, rules, name, level)
            z = l_minus_r_from_excess(fam.excess(RADII, NBAR, SHAPES[name], amp)) / (s0[0] * c95[0])
            ax.plot(r, z[mask[0]], color=color, linewidth=1.5, label=rf"$\tilde\delta={level:g}$")
        ax.set(xscale="log", title=LABEL[name], xlabel="$r$")
    axes[0].set_ylabel(r"$(L_\theta(r)-r)\,/\,(c_{95}\,s_0(r))$")
    axes[-1].legend(loc="lower right", fontsize=6, handlelength=1.2, borderaxespad=0.2)
    fig.savefig(OUT / "02_regime_curves.pdf")
    plt.close(fig)


def _stat(args):
    name, lo, hi = args
    z = np.load(DATA / name / "points.npz")
    pts, off = z["points"], z["offsets"]
    return lo, np.stack([l_minus_r(pts[off[k]:off[k + 1]]) for k in range(lo, hi)])


def realised_statistic(name, jobs=10, chunk=500):
    path = DATA / name / "S.npy"
    if path.exists():
        return np.load(path)
    m = pd.read_csv(DATA / name / "manifest.csv")
    tasks = [(name, lo, min(lo + chunk, len(m))) for lo in range(0, len(m), chunk)]
    with get_context("spawn").Pool(jobs) as pool:
        parts = dict(pool.map(_stat, tasks))
    curves_ = np.concatenate([parts[lo] for _, lo, _ in tasks])
    s = Tables().statistic(curves_, m.n.to_numpy())
    np.save(path, s)
    return s


def power():
    edges = np.geomspace(0.1, 4, 13)
    mid = np.sqrt(edges[1:] * edges[:-1])
    null = realised_statistic("poisson")
    size = (null > 1).mean()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), layout="constrained")
    frames = {}
    for name in STRUCTURED:
        m = pd.read_csv(DATA / name / "manifest.csv")
        m["S"] = realised_statistic(name)
        frames[name] = m
        b = np.digitize(m.delta, edges) - 1
        rate = np.array([(m.S[b == k] > 1).mean() for k in range(len(mid))])
        se = np.sqrt(rate * (1 - rate) / np.bincount(b, minlength=len(mid)))
        axes[0].errorbar(mid, rate, yerr=1.96 * se, color=SERIES[name], linewidth=1.5, marker="o", markersize=3,
                         label=LABEL[name])
        med = [np.median(m.S[b == k]) for k in range(len(mid))]
        axes[2].plot(mid, med, color=SERIES[name], linewidth=1.5, marker="o", markersize=3, label=LABEL[name])
    axes[0].axhline(size, color=MUTED, linestyle="--", linewidth=1)
    axes[0].text(3.9, size + 0.03, f"CSR size {size:.3f}", color=INK2, fontsize=7, ha="right")
    axes[0].set(xscale="log", xlabel=r"$\tilde\delta$", ylabel="rejection rate, 5% L-test", ylim=(0, 1.02), title="by family")
    axes[0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, 0.93))

    pooled = pd.concat(frames.values())
    thirds = np.quantile(pooled.nbar, [0, 1 / 3, 2 / 3, 1])
    for k, color in enumerate(("#86b6ef", "#2a78d6", "#104281")):
        sub = pooled[(pooled.nbar >= thirds[k]) & (pooled.nbar <= thirds[k + 1])]
        b = np.digitize(sub.delta, edges) - 1
        axes[1].plot(mid, [(sub.S[b == j] > 1).mean() for j in range(len(mid))], color=color, linewidth=1.5,
                     marker="o", markersize=3, label=rf"$\bar n\in[{thirds[k]:.0f},{thirds[k + 1]:.0f}]$")
    axes[1].axhline(size, color=MUTED, linestyle="--", linewidth=1)
    axes[1].text(3.9, size + 0.03, "CSR size", color=INK2, fontsize=7, ha="right")
    axes[1].set(xscale="log", xlabel=r"$\tilde\delta$", ylim=(0, 1.02), title="pooled over families, by $\\bar n$")
    axes[1].legend(fontsize=7, loc="upper left")

    axes[2].plot(mid, mid, color=MUTED, linestyle="--", linewidth=1)
    axes[2].axhline(np.median(null), color=MUTED, linestyle=":", linewidth=1)
    axes[2].text(3.8, np.median(null) * 1.08, "CSR median", color=INK2, fontsize=7, ha="right")
    axes[2].set(xscale="log", yscale="log", xlabel=r"$\tilde\delta$", ylabel="median realised $S$", title="by family")
    fig.savefig(OUT / "02_power.pdf")
    plt.close(fig)
    return size


def prior():
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 1.9), layout="constrained")
    amp = {"thomas": ("mu", r"$\mu$"), "nested": ("mu2", r"$\mu_2$"), "lgcp": ("sigma2", r"$\sigma^2$"),
           "matern2": ("R", r"$R\sqrt{\bar n}$")}
    bins = np.linspace(0.6, 1.4, 60)
    for name in ("poisson",) + STRUCTURED:
        m = pd.read_csv(DATA / name / "manifest.csv")
        axes[0].hist(m.n / m.nbar, bins=bins, histtype="step", linewidth=1.2,
                     color=SERIES.get(name, INK), label=LABEL[name], density=True)
        if name in amp:
            ax = axes[1 + STRUCTURED.index(name)]
            key, label = amp[name]
            v = m[key] * (np.sqrt(m.nbar) if name == "matern2" else 1)
            ax.hist(v, bins=np.geomspace(v.min(), v.max(), 40), color=SERIES[name], linewidth=0)
            ax.set(xscale="log", xlabel=label, yticks=[], title=LABEL[name])
    axes[0].set(xlabel=r"$n/\bar n$", yticks=[], title="realised size")
    axes[0].legend(fontsize=5.5, loc="upper left", handlelength=0.8, borderaxespad=0.1)
    fig.savefig(OUT / "02_prior.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    tables = Tables()
    rules = Rules.load(tables)
    gallery(tables, rules, Config.load().root)
    curves(tables, rules)
    prior()
    print("CSR size on sweep:", power())
