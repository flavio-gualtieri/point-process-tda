#!/usr/bin/env python
"""Visual check that delta-tilde measures distance from CSR (one PNG per family).

For the C ladders of Thomas and Matern II (fixed shape, amplitude walked down
to CSR) the figure shows, top to bottom:

  row 1  one replicate cloud at five rungs of the nbar = 500 ladder, and an
         exact CSR anchor (delta-tilde = 0) at the same nbar;
  (a)    the definition: the closed-form |L(r) - r| / s0(r; nbar) of those
         rungs; its peak over the c95 line is delta-tilde;
  (b)    the observed departure S_L = max_r |L_hat - m0| / s0 / c95 of all
         300 replicates per rung (median and 5-95% band) against
         delta-tilde, both ladders, next to the CSR anchors' own S_L band;
  (c)    the rejection rate P(S_L > 1) of the classical 5% test against
         delta-tilde, which should fall to the anchors' realised size.

    python docs/theory/scripts/make_delta_justification.py            # -> docs/theory/figs/*.png
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cloudforger.baselines.vihrs import _isotropic_l_minus_r  # noqa: E402
from cloudforger.generation.nulls import R_GRID, departure, l_minus_r, load_tables, s0_at  # noqa: E402
from cloudforger.generation.prior import build_priors  # noqa: E402
from cloudforger.generation.spec import load_spec  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "dv3" / "C"
FIGS = ROOT / "docs" / "theory" / "figs"
SPEC = ROOT / "configs" / "generation" / "dv3.yaml"

# same palette and rcParams as make_generation_figures.py
INK, INK2, MUTED, GRID, AXIS, WASH = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f0efec"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 8,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "axes.titlesize": 8.6, "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titlepad": 5,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 7.4, "lines.linewidth": 1.6,
    "lines.solid_capstyle": "round",
})

FAMILIES = {"thomas": "Thomas (s = 1, μ walked down)", "matern2": "Matérn II (τ walked down)"}
SHOW_LEVELS = [15, 11, 7, 3, 0]          # rungs drawn as clouds, far -> near CSR
SHOW_NBAR = 500.0
VIEW = {"thomas": 1.0, "matern2": 0.35}   # side of the sub-window drawn; the hard core is invisible at full size
LADDER_COLORS = {125.0: ORANGE, 500.0: BLUE}


def load_family(family: str):
    rows = list(csv.DictReader(open(DATA / family / "manifest.csv")))
    z = np.load(DATA / family / "points.npz")
    coords, off = z["coords"], z["offsets"]
    assert np.array_equal(z["index"], [int(r["index"]) for r in rows])
    return rows, [coords[off[i]:off[i + 1]] for i in range(len(rows))]


def observed_S(clouds, tabs):
    curves = np.stack([_isotropic_l_minus_r(p, np.zeros(2), np.ones(2), R_GRID) for p in clouds])
    n = np.array([len(p) for p in clouds], float)
    return departure(curves, n, "L", tabs, center=True)


def figure(family, tabs, priors, anchors):
    rows, clouds = load_family(family)
    prior = priors[family]
    key = lambda r: (float(r["nbar"]), int(r["level_id"]))
    groups: dict[tuple[float, int], list[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(key(r), []).append(i)
    S = observed_S(clouds, tabs)
    shape_keys = [k for k in prior.design_keys]

    fig = plt.figure(figsize=(12.5, 6.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.35], hspace=0.3)
    top = gs[0].subgridspec(1, len(SHOW_LEVELS) + 1, wspace=0.08)
    bot = gs[1].subgridspec(1, 3, wspace=0.28)

    # row 1: clouds
    colors = dict(zip(SHOW_LEVELS, SEQ[::-1]))
    for j, lev in enumerate(SHOW_LEVELS + [None]):
        ax = fig.add_subplot(top[j])
        if lev is None:
            pts, col = anchors[SHOW_NBAR]["cloud"], MUTED
            title = "CSR   δ̃ = 0"
        else:
            idx = groups[(SHOW_NBAR, lev)]
            pts, col = clouds[idx[0]], colors[lev]
            r = rows[idx[0]]
            amp = ", ".join(f"{k} = {float(r[k]):.3g}" for k in shape_keys if r[k] and k not in ("s",))
            title = f"δ̃ = {float(r['delta_tilde']):.2f}\n{amp}"
        v = VIEW[family]
        ax.scatter(pts[:, 0], pts[:, 1], s=2.2 / v, color=INK, lw=0)
        ax.set_xlim(0, v), ax.set_ylim(0, v), ax.set_aspect("equal")
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True), sp.set_color(col), sp.set_linewidth(2.2)
        ax.set_title(title, fontsize=8, loc="center")
        ax.text(0.5, -0.06, f"n = {len(pts)}" + ("" if v == 1 else f"  (view [0, {v}]²)"), transform=ax.transAxes, ha="center", va="top",
                fontsize=7, color=INK2)

    # (a) definition: closed-form curve over the null s.d.; peak / c95 = delta-tilde
    ax = fig.add_subplot(bot[0])
    lo_i = np.searchsorted(tabs["n_grid"], SHOW_NBAR) - 1
    w = (np.log(SHOW_NBAR) - np.log(tabs["n_grid"][lo_i])) / np.log(tabs["n_grid"][lo_i + 1] / tabs["n_grid"][lo_i])
    c95 = (1 - w) * tabs["L"]["c95"][lo_i] + w * tabs["L"]["c95"][lo_i + 1]
    mask = tabs["L"]["mask"][lo_i] & tabs["L"]["mask"][lo_i + 1]
    s0 = s0_at(SHOW_NBAR, "L", tabs)
    for lev in SHOW_LEVELS:
        r = rows[groups[(SHOW_NBAR, lev)][0]]
        shape = {k: float(r[k]) for k in shape_keys}
        z = np.where(mask, np.abs(l_minus_r(prior.excess(R_GRID, SHOW_NBAR, shape))) / np.where(mask, s0, 1), np.nan)
        ax.plot(R_GRID, z, color=colors[lev], label=f"δ̃ = {float(r['delta_tilde']):.2f}")
        k = int(np.nanargmax(z))
        ax.plot(R_GRID[k], z[k], "o", ms=5, color=colors[lev], mec="white", mew=1)
    ax.axhline(c95, color=MUTED, lw=1, ls="--")
    ax.text(0.248, c95 * (0.9 if family == "matern2" else 1.08), f"c₀.₉₅ = {c95:.2f}  (null 95% point)", ha="right", va="top" if family == "matern2" else "bottom", fontsize=7, color=INK2)
    ax.axhline(0, color=AXIS, lw=0.6)   # CSR: the curve is identically 0
    ax.set_yscale("log"), ax.set_ylim(0.03, 80), ax.set_xlim(0, 0.25)
    ax.set_xlabel("r"), ax.set_ylabel("|L(r) − r| / s₀(r; n̄)   (closed form)")
    ax.set_title(f"(a) Definition at n̄ = {int(SHOW_NBAR)}: δ̃ = peak / c₀.₉₅\n     CSR is 0 at every r")
    ax.legend(loc="upper right" if family == "matern2" else "lower right", handlelength=1.4)

    # (b) observed departure vs delta-tilde
    ax = fig.add_subplot(bot[1])
    xlo = 0.06
    for nb, col in LADDER_COLORS.items():
        levs = sorted(l for (n_, l) in groups if n_ == nb)
        d = np.array([float(rows[groups[(nb, l)][0]]["delta_tilde"]) for l in levs])
        q = np.array([np.quantile(S[groups[(nb, l)]], [0.05, 0.5, 0.95]) for l in levs])
        o = np.argsort(d)
        ax.fill_between(d[o], q[o, 0], q[o, 2], color=col, alpha=0.15, lw=0)
        ax.plot(d[o], q[o, 1], color=col, marker="o", ms=3.5, label=f"ladder n̄ = {int(nb)}")
        a = anchors[nb]["S"]
        aq = np.quantile(a, [0.05, 0.5, 0.95])
        ax.plot([xlo * 0.8, xlo * 1.25], [aq[1]] * 2, color=col, lw=2.4)
        ax.plot([xlo] * 2, [aq[0], aq[2]], color=col, lw=1)
    ax.axvspan(0.035, 0.085, color=WASH, lw=0, zorder=0)
    ax.text(xlo, 0.2, "CSR\nanchors", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.axhline(1, color=MUTED, lw=1, ls="--")
    g = np.geomspace(0.5, 12, 50)
    ax.plot(g, g, color=AXIS, lw=0.9, ls=":")
    ax.text(11, 13.5, "S = δ̃", fontsize=7, color=INK2, ha="right")
    ax.set_xscale("log"), ax.set_yscale("log"), ax.set_xlim(0.035, 13), ax.set_ylim(0.18, 25)
    ax.set_xlabel("δ̃ (design distance from CSR)"), ax.set_ylabel("observed S_L per cloud")
    ax.set_title("(b) Observed departure on the simulated clouds\n     median and 5–95% band, 300 clouds per rung")
    ax.legend(loc="upper left", handlelength=1.4)

    # (c) rejection rate of the classical 5% test
    ax = fig.add_subplot(bot[2])
    for nb, col in LADDER_COLORS.items():
        levs = sorted(l for (n_, l) in groups if n_ == nb)
        d = np.array([float(rows[groups[(nb, l)][0]]["delta_tilde"]) for l in levs])
        p = np.array([np.mean(S[groups[(nb, l)]] > 1) for l in levs])
        o = np.argsort(d)
        ax.plot(d[o], p[o], color=col, marker="o", ms=3.5, label=f"ladder n̄ = {int(nb)}")
        ax.plot([xlo], [np.mean(anchors[nb]["S"] > 1)], "D", ms=5.5, color=col, mec="white", mew=1)
    ax.axvspan(0.035, 0.085, color=WASH, lw=0, zorder=0)
    ax.axhline(0.05, color=MUTED, lw=1, ls="--")
    ax.text(12, 0.065, "nominal size 5%", ha="right", fontsize=7, color=INK2)
    ax.axvline(1, color=AXIS, lw=0.8)
    ax.text(xlo, 0.9, "CSR\nanchors", ha="center", va="top", fontsize=7, color=INK2)
    ax.set_xscale("log"), ax.set_xlim(0.035, 13), ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("δ̃ (design distance from CSR)"), ax.set_ylabel("P(S_L > 1): classical test rejects CSR")
    ax.set_title("(c) Detectability falls to the null size as δ̃ → 0\n     δ̃ ≈ 1 is the edge of detection")
    ax.legend(loc="center right", handlelength=1.4)

    fig.suptitle(f"{FAMILIES[family]}: clouds approach CSR as δ̃ decreases", x=0.125, ha="left",
                 fontsize=11, color=INK, y=0.93)
    FIGS.mkdir(exist_ok=True)
    out = FIGS / f"delta_justification_{family}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(out)


def main():
    tabs = load_tables(SPEC.parent / "null_tables.npz")
    priors = build_priors(load_spec(SPEC))
    rows, clouds = load_family("poisson")
    anchors = {}
    for nb in LADDER_COLORS:
        idx = [i for i, r in enumerate(rows) if float(r["nbar"]) == nb]
        anchors[nb] = {"S": observed_S([clouds[i] for i in idx], tabs), "cloud": clouds[idx[0]]}
    for family in FAMILIES:
        figure(family, tabs, priors, anchors)


if __name__ == "__main__":
    main()
