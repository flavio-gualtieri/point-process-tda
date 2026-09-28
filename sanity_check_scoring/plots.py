"""Figures for the throwaway scoring sanity check (reads results.json from run.py)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from cloudforger.scores.kernel import Component
from cloudforger.scores.simulate import simulate

from run import KC, SCENARIOS, rng_for

OUT = Path(__file__).parent
RES = json.loads((OUT / "results.json").read_text())["scenarios"]

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
TRUTH, NEAR, WRONG, CSR = "#2a78d6", "#1baf7a", "#eb6834", "#8a8984"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.titlesize": 9.5, "axes.titlecolor": INK, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "savefig.dpi": 170,
})


def role(name):
    return TRUTH if name == "oracle" else CSR if name == "CSR" else WRONG if "wrong" in name else NEAR


def model_of(scen, name, x):
    m = SCENARIOS[scen]["ladder"][name]
    return m if m is not None else ("poisson", {"nbar": float(len(x))})


def ladder_arrays(scen):
    v = RES[scen]
    L = {k: np.array([c["ladder"][k] for c in v]) for k in v[0]["ladder"]}
    gain = (L["CSR"] - L["oracle"]).mean()
    return {k: (a - L["oracle"]) / gain for k, a in L.items()}, gain


def observed(scen, i=0):
    return simulate(*SCENARIOS[scen]["truth"], 1, rng_for("x", scen, i))[0]


def dots(ax, pts, zoom=None, s=3):
    ax.scatter(pts[:, 0], pts[:, 1], s=s, c=INK, lw=0)
    lim = (0, zoom or 1)
    ax.set(xlim=lim, ylim=lim, xticks=[], yticks=[], aspect="equal")
    for sp in ax.spines.values():
        sp.set_visible(True), sp.set_color(GRID)


# ----------------------------------------------------------------- fig 1: what the clouds look like
def fig_gallery():
    rows = [("thomas", None), ("matern2", 0.35)]
    ncol = max(len(SCENARIOS[s]["ladder"]) for s, _ in rows) + 1
    fig, axes = plt.subplots(2, ncol, figsize=(1.75 * ncol, 5.0))
    for r, (scen, zoom) in enumerate(rows):
        x = observed(scen)
        rel, _ = ladder_arrays(scen)
        dots(axes[r, 0], x, zoom, s=4 if zoom else 3)
        axes[r, 0].set_title("OBSERVED cloud x", fontweight="bold")
        axes[r, 0].set_ylabel({"thomas": "Thomas cloud\n(clustered)", "matern2": "Matérn II cloud\n(regular; zoomed 0.35²)"}[scen],
                              color=INK, fontsize=9.5)
        for c, name in enumerate(SCENARIOS[scen]["ladder"], start=1):
            ax = axes[r, c]
            sim = simulate(*model_of(scen, name, x), 1, rng_for("gallery", scen, name))[0]
            dots(ax, sim, zoom, s=4 if zoom else 3)
            ax.set_title(name, color=role(name) if name != "CSR" else INK2)
            m = rel[name].mean()
            ax.text(0.5, -0.09, "regret / gain = " + (f"{m:.2f}" if m < 10 else f"{m:.0f}"),
                    transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=INK2)
        for c in range(len(SCENARIOS[scen]["ladder"]) + 1, ncol):
            axes[r, c].axis("off")
    fig.suptitle("One simulated pattern from each candidate model  (0 = as good as the truth, 1 = as bad as CSR)",
                 color=INK, fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=3.0)
    fig.savefig(OUT / "fig1_gallery.png")
    plt.close(fig)


# ------------------------------------------------------------------ fig 2: the ladder, cloud by cloud
def fig_ladder():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [6, 5]})
    for ax, scen, title, cap in [(axes[0], "thomas", "Observed: clustered Thomas cloud", None),
                                 (axes[1], "matern2", "Observed: regular Matérn II cloud", 1.35)]:
        rel, gain = ladder_arrays(scen)
        rng = np.random.default_rng(0)
        for j, (name, a) in enumerate(rel.items()):
            col = role(name)
            shown = a if cap is None else np.minimum(a, cap)
            ax.scatter(j + rng.uniform(-0.18, 0.18, len(a)), shown, s=9, color=col, alpha=0.45, lw=0)
            m, se = a.mean(), a.std(ddof=1) / np.sqrt(len(a))
            if cap is not None and m > cap:
                ax.annotate(f"{m:.0f}× gain\n(off scale)", (j, cap), xytext=(j, cap - 0.28), ha="center",
                            fontsize=8, color=INK2, arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5))
                continue
            ax.errorbar(j, m, yerr=2 * se, fmt="o", ms=7, color=col, mec="#fcfcfb", mew=1.5, elinewidth=2, capsize=0)
            ax.text(j + 0.24, m, f"{m:.2f}", va="center", fontsize=8, color=INK2)
        ax.axhline(0, color=TRUTH, lw=1, ls=(0, (4, 3)), zorder=0)
        ax.axhline(1, color=CSR, lw=1, ls=(0, (4, 3)), zorder=0)
        ax.set_xticks(range(len(rel)), [n.replace(", ", ",\n").replace(" (", "\n(") for n in rel], fontsize=8)
        ax.set_ylabel("regret / mean gain   (per cloud)")
        ax.set_title(f"{title}\n40 independent clouds; the truth scores best on all 40", loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        if cap is not None:
            ax.set_ylim(-0.1, cap + 0.05)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_ladder.png")
    plt.close(fig)


# -------------------------------------------------------- fig 3: sweeping one parameter away from truth
def fig_sweeps():
    specs = [("thomas", "sigma", "cluster width σ  (× true)", True),
             ("thomas", "clusters", "number of clusters κ  (× true; nbar fixed)", True),
             ("matern2", "core", "hard-core radius R  (× true)", False)]
    fig = plt.figure(figsize=(12, 4.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 3.2], hspace=0.08, wspace=0.28)
    for c, (scen, sw, xlabel, logx) in enumerate(specs):
        v = RES[scen]
        ms = list(v[0]["sweeps"][sw])
        o = np.array([cl["ladder"]["oracle"] for cl in v])
        gain = (np.array([cl["ladder"]["CSR"] for cl in v]) - o).mean()
        R = np.array([[cl["sweeps"][sw][m] for m in ms] for cl in v]) - o[:, None]
        R /= gain
        mv = np.array(ms, float)
        mean, se = R.mean(0), R.std(0, ddof=1) / np.sqrt(len(R))
        ax = fig.add_subplot(gs[1, c])
        ax.fill_between(mv, mean - 2 * se, mean + 2 * se, color=NEAR, alpha=0.2, lw=0)
        ax.plot(mv, mean, color=NEAR, lw=2)
        ax.scatter(mv, mean, s=28, color=NEAR, ec="#fcfcfb", lw=1.2, zorder=3)
        ax.scatter([1], [0], s=70, color=TRUTH, ec="#fcfcfb", lw=1.5, zorder=4)
        ax.axhline(1, color=CSR, lw=1, ls=(0, (4, 3)))
        ax.text(mv[0], 1.03, "CSR", color=INK2, fontsize=8, ha="left", va="bottom")
        wrong = [k for k in v[0]["ladder"] if "wrong" in k][0]
        w = (np.array([cl["ladder"][wrong] for cl in v]) - o).mean() / gain
        if w < 3:
            ax.axhline(w, color=WRONG, lw=1, ls=(0, (4, 3)))
            ax.text(mv[-1], w + 0.03, wrong, ha="right", color=INK2, fontsize=8, va="bottom")
        ax.axvline(1, color=TRUTH, lw=0.8, alpha=0.5)
        ax.text(1, ax.get_ylim()[1] if False else -0.02, " truth", color=TRUTH, fontsize=8, va="top")
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        if c == 0:
            ax.set_ylabel("regret / mean gain\n(mean ± 2 s.e., 40 clouds)")
        argmin = np.bincount(R.argmin(1), minlength=len(ms))[ms.index("1.0")]
        ax.set_title(f"minimum at the truth on {argmin}/40 single clouds", loc="left", fontsize=8.5, color=INK2)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        # thumbnails of the swept model above the curve
        picks = {"sigma": ["0.25", "1.0", "4.0", "16.0"], "clusters": ["0.125", "1.0", "4.0", "20.0"],
                 "core": ["0.25", "0.5", "1.0", "1.1"]}[sw]
        sub = gs[0, c].subgridspec(1, len(picks), wspace=0.08)
        for k, m in enumerate(picks):
            a = fig.add_subplot(sub[0, k])
            pts = simulate(*SCENARIOS[scen]["sweeps"][sw][float(m)], 1, rng_for("thumb", scen, sw, m))[0]
            dots(a, pts, 0.35 if scen == "matern2" else None, s=1.2 if scen == "thomas" else 2.5)
            a.set_title(f"×{float(m):g}", fontsize=8, color=TRUTH if m == "1.0" else INK2, pad=2)
    fig.suptitle("Intermediate models: moving one parameter away from the truth raises the score smoothly toward (or past) CSR",
                 color=INK, fontsize=10.5, x=0.01, ha="left", y=0.99)
    fig.savefig(OUT / "fig3_sweeps.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------- fig 4: what the score actually compares
def fig_mechanism():
    scen = "thomas"
    x = observed(scen)
    rng = np.random.default_rng(1)
    comp = Component(x, KC["R"], KC["kind"], KC["h"], KC["grid"], KC["centres"], rng)
    names = ["observed", "oracle", "thomas, sigma x2", "thomas, sigma x4", "CSR", "matern2 (wrong)"]
    embs, counts = {}, {}
    for nm in names:
        pats = [x] if nm == "observed" else simulate(*model_of(scen, nm, x), 16, rng_for("mech", nm))
        E = np.concatenate([comp.embed(p, rng, 60) for p in pats])
        embs[nm] = E
        cnt = []
        for p in pats:
            X = p * np.sqrt(len(p))
            t = cKDTree(X)
            cnt += [len(n) - 1 for n in t.query_ball_point(X, KC["R"])]
        counts[nm] = np.array(cnt)
    fig = plt.figure(figsize=(12.5, 6.4))
    gs = fig.add_gridspec(3, len(names), height_ratios=[1, 1, 1.1], hspace=0.35, wspace=0.12)
    vmax = max(embs[n].mean(0).max() for n in names)
    ex = embs["observed"]
    for c, nm in enumerate(names):
        col = INK if nm == "observed" else role(nm)
        # one typical configuration: the median-crowded one
        E = embs[nm]
        a = fig.add_subplot(gs[0, c])
        mass = E.sum(1)
        a.imshow(E[np.argsort(mass)[int(0.8 * len(E))]].reshape(21, 21), origin="lower", cmap="Blues",
                 vmin=0, vmax=np.percentile(ex.max(1), 90), extent=(-1, 1, -1, 1))
        a.set(xticks=[], yticks=[])
        a.set_title("observed x" if nm == "observed" else nm, color=col if nm != "CSR" else INK2)
        if c == 0:
            a.set_ylabel("a busy configuration\n(80th pct. crowding)", fontsize=8.5)
        b = fig.add_subplot(gs[1, c])
        b.imshow(E.mean(0).reshape(21, 21), origin="lower", cmap="Blues", vmin=0, vmax=vmax, extent=(-1, 1, -1, 1))
        b.set(xticks=[], yticks=[])
        if c == 0:
            b.set_ylabel("MEAN configuration\n(what `lin` sees)", fontsize=8.5)
        if nm != "observed":
            d = np.linalg.norm(E.mean(0) - ex.mean(0))
            b.set_xlabel(f"distance to observed: {d:.1f}", fontsize=8, color=INK2)
    ax = fig.add_subplot(gs[2, :])
    bins = np.arange(0, 45) - 0.5
    for nm in names:
        col = INK if nm == "observed" else role(nm)
        h, _ = np.histogram(counts[nm], bins=bins, density=True)
        ls = {"thomas, sigma x2": (0, (5, 2)), "thomas, sigma x4": (0, (1.5, 1.5))}.get(nm, "-")
        ax.plot(bins[:-1] + 0.5, h, color=col, lw=2.4 if nm == "observed" else 1.6, ls=ls, label="observed x" if nm == "observed" else nm,
                drawstyle="steps-mid")
    ax.set_xlabel("neighbours within R = 1.5 mean spacings of a point  (a 1-D shadow of the configuration law the score compares)")
    ax.set_ylabel("share of points")
    ax.legend(frameon=False, ncol=3, fontsize=8.5)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.suptitle("What the kernel score compares: the local configurations around each point (radius 1.5 spacings, rescaled to the unit disk)",
                 color=INK, fontsize=10.5, x=0.01, ha="left")
    fig.savefig(OUT / "fig4_mechanism.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_gallery()
    fig_ladder()
    fig_sweeps()
    fig_mechanism()
