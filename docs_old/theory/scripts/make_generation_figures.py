"""Figures for docs/theory/generation.tex (DV3 generation spec).

Reads only what is already on disk: out/null.npz and out/data_departure.npz
(SLURM job 26430628), out/design_preview.{json,npz} (design_preview.py, closed
forms), out/diagram_tasks.npz and out/inspect_generation.json
(inspect_generation.py, SLURM accounting). Simulates nothing.

    python docs/theory/scripts/make_generation_figures.py [--preview DIR]
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

import departure as dep  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
FIGS = HERE.parent / "figs"

# reference palette (dataviz skill, references/palette.md), light mode
INK, INK2, MUTED, GRID, AXIS, WASH = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f0efec"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"          # categorical slots 1-3
SEQ = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # ordinal blue, steps 250..650
TEXTW = 6.38                                                    # A4 text width in inches

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 7.2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "axes.titlesize": 7.8, "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titlepad": 5,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 6.8, "lines.linewidth": 1.4,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "pdf.fonttype": 42, "savefig.dpi": 300,
})

TABS = dep.null_tables(OUT / "null.npz")
R = TABS["r_grid"]


def save(fig, name, preview):
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    if preview:
        fig.savefig(Path(preview) / f"{name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def null_at(nbar):
    """s0(r; nbar), c95(nbar) and the L mask, interpolated linearly in log n (as in departure.py)."""
    t, ng = TABS["L"], np.log(TABS["n_grid"])
    ln = np.log(nbar)
    hi = int(np.clip(np.searchsorted(ng, ln, side="right"), 1, len(ng) - 1))
    lo = hi - 1
    w = (ln - ng[lo]) / (ng[hi] - ng[lo])
    s0 = (1 - w) * t["s0"][lo] + w * t["s0"][hi]
    c95 = (1 - w) * t["c95_max"][lo] + w * t["c95_max"][hi]
    return s0, c95, t["mask"][lo] & t["mask"][hi]


# ----------------------------------------------------------------------------- F1 pipeline
def fig_pipeline(preview):
    fig, ax = plt.subplots(figsize=(TEXTW, 2.25))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")
    boxes = [
        ("1  Identify", "case id\n→ spawn key\n(DV, set,\nfamily, case,\nrole)"),
        ("2  Draw θ", "n̄: log-uniform\nshape | n̄ ~ π\n(reject shape,\nnever n̄)\nB, C: θ fixed"),
        ("3  Invert", "design coords\n→ model\nparameters\n(Strauss:\nfitted lookup)"),
        ("4  Simulate", "exact sampler\non a buffered\nwindow, then\ncrop to [0,1]²"),
        ("5  Store", "points → .npz\nrow →\nmanifest.csv\n+ diagnostics"),
        ("6  Check", "case: flags\ncell: V-checks\npass / fail\nfixed in\nadvance"),
    ]
    w, gap, y0, h = 15.2, 1.55, 7.0, 17.0
    xs = [0.4 + i * (w + gap) for i in range(len(boxes))]
    for x, (title, body) in zip(xs, boxes):
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc=WASH, ec="none"))
        ax.text(x + 1.0, y0 + h - 2.0, title, ha="left", va="top", fontsize=7.4, color=INK, weight="bold")
        ax.text(x + 0.9, y0 + h - 5.2, body, ha="left", va="top", fontsize=5.9, color=INK2, linespacing=1.3)
    for x in xs[:-1]:
        ax.annotate("", xy=(x + w + gap - 0.2, y0 + h / 2), xytext=(x + w + 0.2, y0 + h / 2),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.25,head_width=0.15", color=MUTED, lw=0.9, shrinkA=0, shrinkB=0))
    # which random stream / where it runs
    ax.text(xs[1] + w / 2, y0 + h + 1.6, "param stream", ha="center", fontsize=6.2, color=MUTED)
    ax.text(xs[3] + w / 2, y0 + h + 1.6, "pattern stream", ha="center", fontsize=6.2, color=MUTED)
    ax.plot([xs[0], xs[2] + w], [4.6, 4.6], color=AXIS, lw=0.8)
    ax.text((xs[0] + xs[2] + w) / 2, 1.2, "plan job (seconds)", ha="center", fontsize=6.2, color=INK2)
    ax.plot([xs[3], xs[4] + w], [4.6, 4.6], color=AXIS, lw=0.8)
    ax.text((xs[3] + xs[4] + w) / 2, 3.6, "SLURM arrays, 1 CPU per shard\n(Strauss via Rscript)",
            ha="center", va="top", fontsize=6.2, color=INK2)
    ax.plot([xs[5], xs[5] + w], [4.6, 4.6], color=AXIS, lw=0.8)
    ax.text(xs[5] + w / 2, 1.2, "validation job", ha="center", fontsize=6.2, color=INK2)
    save(fig, "gen_pipeline", preview)


# ----------------------------------------------------------------------------- F2 delta explained
def fig_delta(preview):
    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXTW, 2.45), gridspec_kw={"wspace": 0.28})
    z = np.load(OUT / "data_departure.npz")
    n, S = z["thomas__test__n"], z["thomas__test__L_S_max"]
    a.scatter(n, S, s=2.5, color=BLUE, alpha=0.28, lw=0, rasterized=True)
    a.axhline(1, color=MUTED, lw=0.8)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim(45, 1400)
    a.set_ylim(0.4, 120)
    a.set_xlabel("points in the cloud, n")
    a.set_ylabel("observed departure S_L")
    a.text(50, 0.8, "S = 1: not rejected at 5%", fontsize=6.3, color=INK2, va="top")
    a.text(50, 90, "Spearman ρ(n, S_L) = 0.62\n0 of 7,000 clouds reach S ≤ 1", fontsize=6.3, color=INK2, va="top")
    a.set_title("(a) DV1 Thomas: departure is tied to n")

    mu, s = 2.0, 0.5
    for nb, col in [(125.0, SEQ[0]), (250.0, SEQ[2]), (500.0, SEQ[4])]:
        s0, c95, mask = null_at(nb)
        K = dep.K_thomas(R, nb / mu, s / np.sqrt(nb))
        zc = np.where(mask, np.abs(dep.l_minus_r(K, R)) / np.where(mask, s0, 1), np.nan)
        d = dep.departure(dep.l_minus_r(K, R)[None, :], np.array([nb]), "L", TABS, center=False)["S_max"][0]
        b.plot(R, zc, color=col, label=f"n̄ = {int(nb)}:  peak / c₀.₉₅ = δ̃ = {d:.1f}")
        k = int(np.nanargmax(zc))
        b.plot(R[k], zc[k], "o", ms=4.2, color=col, mec="white", mew=1.0)
    b.axhline(3.2, color=MUTED, lw=0.8, label="c₀.₉₅ ≈ 3.2, the null 95% point of the peak")
    b.set_xlim(0, 0.25)
    b.set_ylim(0, 16.5)
    b.set_xlabel("r")
    b.set_ylabel("|L(r) − r| / s₀(r; n̄)")
    b.legend(loc="upper right", handlelength=1.4, fontsize=6.2)
    b.set_title("(b) One Thomas shape (μ = 2, s = 0.5), three n̄")
    save(fig, "gen_delta", preview)


# ----------------------------------------------------------------------------- F3 ladders
def fig_ladders(preview):
    zp = np.load(OUT / "design_preview.npz")
    deltas = zp["ladder_delta"]
    s0, c95, mask = null_at(250.0)
    band = np.where(mask, c95 * s0, np.nan)
    panels = [("thomas", "Thomas: μ ↓ at s = 1", "μ"), ("nested", "Nested: μ₂ ↓ at μ₁ = 3, ρ = 6, s₂ = 0.2", "μ₂"),
              ("lgcp", "LGCP: σ² ↓ at s√n̄ = 1", "σ²"), ("matern2", "Matérn II: τ ↓ (one parameter)", "τ")]
    fig, axes = plt.subplots(2, 2, figsize=(TEXTW, 4.2), gridspec_kw={"hspace": 0.55, "wspace": 0.34})
    for axx, (fam, title, sym) in zip(axes.ravel(), panels):
        curves, amps = zp[f"ladder_{fam}"], zp[f"ladder_{fam}_amp"]
        axx.fill_between(R, -band, band, color=WASH, lw=0, zorder=0)
        axx.axhline(0, color=AXIS, lw=0.6)
        for i, (d, cur) in enumerate(zip(deltas, curves)):
            axx.plot(R, cur, color=SEQ[i], label=f"δ̃ = {d:g}")
        axx.set_xlim(0, 0.25)
        lo, hi = np.nanmin(curves), np.nanmax(curves)
        pad = 0.08 * (hi - lo)
        axx.set_ylim(min(lo, -np.nanmax(band[R < 0.25])) - pad, max(hi, np.nanmax(band[R < 0.25])) + pad)
        axx.set_title(title)
        axx.set_xlabel("r")
        axx.set_ylabel("L(r) − r")
        fmt = (lambda v: f"{v:.3f}") if fam == "matern2" else (lambda v: f"{v:.2g}")
        axx.text(0.99, 0.04 if fam != "matern2" else 0.96, f"{sym} = " + ", ".join(fmt(v) for v in amps),
                 transform=axx.transAxes, ha="right", va="bottom" if fam != "matern2" else "top",
                 fontsize=6.0, color=INK2)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.02), handlelength=1.6)
    save(fig, "gen_ladders", preview)


# ----------------------------------------------------------------------------- F4 S vs delta
def fig_s_vs_delta(preview):
    z = np.load(OUT / "data_departure.npz")
    nz = np.load(OUT / "null.npz")
    i = int(np.argmin(np.abs(nz["n_grid"] - 400)))
    arr = nz["lmr"][i].astype(float)
    h = arr.shape[0] // 2
    t = TABS["L"]
    zz = dep._z(arr[h:], t["m0"][i], t["s0"][i], t["mask"][i])
    Snull = np.abs(zz).max(1) / t["c95_max"][i]

    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXTW, 2.55), sharey=True,
                               gridspec_kw={"width_ratios": [1, 5.2], "wspace": 0.06})
    rng = np.random.default_rng(0)
    a.scatter(rng.uniform(-0.3, 0.3, len(Snull)), Snull, s=4, color=MUTED, alpha=0.6, lw=0)
    a.plot([-0.42, 0.42], [np.median(Snull)] * 2, color=INK, lw=1.2)
    a.set_xlim(-0.6, 0.6)
    a.set_xticks([0])
    a.set_xticklabels(["δ = 0"])
    a.set_yscale("log")
    a.set_ylim(0.3, 120)
    a.set_ylabel("observed departure S_L")
    a.axhline(1, color=MUTED, lw=0.8)
    a.set_title("CSR")
    a.grid(False, axis="x")

    for fam, col, lab in [("thomas", BLUE, "Thomas"), ("nested_thomas", ORANGE, "nested Thomas")]:
        b.scatter(z[f"{fam}__test__Lpop_S_max"], z[f"{fam}__test__L_S_max"], s=2.5, color=col, alpha=0.3,
                  lw=0, rasterized=True, label=lab)
    b.axvspan(0.1, 1.9, color=WASH, lw=0, zorder=0)
    b.text(0.13, 45, "DV1: 0.4% of Thomas and\n0% of nested clouds\nhave δ < 2; DV3 fills this", fontsize=6.3, color=INK2, va="top")
    xx = np.array([0.1, 120])
    b.plot(xx, xx, color=INK2, lw=0.8)
    b.text(60, 36, "S = δ", fontsize=6.3, color=INK2, rotation=0)
    b.axhline(1, color=MUTED, lw=0.8)
    b.set_xscale("log")
    b.set_xlim(0.1, 120)
    b.set_xlabel("population departure δ (closed-form L at the cloud's n)")
    leg = b.legend(loc="lower right", markerscale=3, handletextpad=0.3)
    for hnd in leg.legend_handles:
        hnd.set_alpha(1)
    b.set_title("δ > 0: S tracks δ, with noise that is largest near CSR")
    save(fig, "gen_s_vs_delta", preview)


# ----------------------------------------------------------------------------- F5 test sets
def fig_testsets(preview):
    zp = np.load(OUT / "design_preview.npz")
    js = json.loads((OUT / "design_preview.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 2.3), sharex=True, sharey=True, gridspec_kw={"wspace": 0.08})
    for axx in axes:
        axx.axvspan(0.02, 1.0, color=WASH, lw=0, zorder=0)
        axx.set_xscale("log")
        axx.set_yscale("log")
        axx.set_xlim(0.025, 60)
        axx.set_ylim(90, 900)
        axx.set_yticks([125, 250, 500, 800])
        axx.set_yticklabels(["125", "250", "500", "800"])
        axx.minorticks_off()
        axx.set_xlabel("departure δ")
    axes[0].set_ylabel("expected points n̄")
    axes[0].text(0.03, 820, "near CSR", fontsize=6.3, color=INK2, va="top")

    a, b, c = axes
    a.scatter(zp["draws_thomas_delta"], zp["draws_thomas_nbar"], s=2.5, color=BLUE, alpha=0.35, lw=0, rasterized=True)
    a.set_title("A: i.i.d. from the prior")

    marks = {0.2: ("s", 0.88), 0.45: ("D", 1.0), 1.0: ("^", 1.13)}
    for sv, (mk, off) in marks.items():
        cells = [cc for cc in js["B"]["thomas"]["cells"] if cc["feasible"] and abs(cc["s"] - sv) < 1e-9]
        b.scatter([cc["delta"] for cc in cells], [cc["nbar"] * off for cc in cells], marker=mk, s=16,
                  color=ORANGE, edgecolor="white", linewidth=0.6, label=f"s = {sv:g}")
    b.legend(loc="upper right", handletextpad=0.2, borderaxespad=0.2)
    b.set_title("B: fixed cells × 400 reps")

    for nb in (125, 500):
        dlo, dhi = js["C"]["thomas"][str(nb)]["delta_range_used"]
        lv = np.exp(np.linspace(np.log(dlo), np.log(dhi), 16))
        c.plot(lv, [nb] * 16, color=AQUA, lw=1.0)
        c.scatter(lv, [nb] * 16, s=9, color=AQUA, edgecolor="white", linewidth=0.5, zorder=3)
    for nb in (125, 250, 500):
        c.scatter([0.033], [nb], marker="*", s=40, color=INK, zorder=4)
    c.text(0.04, 175, "★ exact Poisson\n(δ = 0, 2,000 reps)", fontsize=6.0, color=INK2)
    c.set_title("C: ladders to exact CSR")
    save(fig, "gen_testsets", preview)


# ----------------------------------------------------------------------------- F6 compute budget
SETS = {  # set: (clouds, n-bar levels or None for log-uniform, k values computed)
    "training": (60_000, None, 3),
    "A": (30_000, None, 3),
    "B": (60_400, (125.0, 250.0, 500.0), 1),
    "C": (54_000, (125.0, 500.0), 1),
}
DISPERSION = 1.2   # E[(n / nbar)^3.1] over families: ~1.09 Neyman-Scott, ~1.5 LGCP, ~1.0 Strauss


def budget():
    fits = json.loads((OUT / "inspect_generation.json").read_text())["diagram_cost_model t = c*(n/400)^alpha"]
    ks = ["dtm_k5", "dtm_k10", "dtm_k15"]

    def per_cloud(nbar, n_k):
        return DISPERSION * sum(fits[k]["sec_per_diagram_at_n400"] * (nbar / 400.0) ** fits[k]["alpha"] for k in ks[:n_k])

    out = {}
    u = np.exp(np.linspace(np.log(100), np.log(800), 4001))   # log-uniform grid
    for name, (n, levels, n_k) in SETS.items():
        sec = float(np.mean(per_cloud(u, n_k))) if levels is None else float(np.mean([per_cloud(v, n_k) for v in levels]))
        out[name] = {"clouds": n, "sec_per_cloud": sec, "cpu_h": n * sec / 3600.0, "k": n_k}
    return fits, out


def fig_cost(preview):
    zt = np.load(OUT / "diagram_tasks.npz")
    fits, bud = budget()
    (OUT / "compute_budget.json").write_text(json.dumps(bud, indent=1))
    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXTW, 2.35), gridspec_kw={"width_ratios": [1.05, 1], "wspace": 0.45})
    m = np.char.startswith(zt["filt"].astype(str), "dtm")
    a.scatter(zt["n_eff"][m], zt["sec_per_diagram"][m], s=4, color=BLUE, alpha=0.35, lw=0, rasterized=True,
              label="DV1 tasks (DTM k = 5, 10, 15)")
    f = fits["dtm_k5"]
    nn = np.geomspace(100, 1100, 100)
    a.plot(nn, f["sec_per_diagram_at_n400"] * (nn / 400) ** f["alpha"], color=INK, lw=1.2,
           label=f"fit, k = 5: {f['sec_per_diagram_at_n400']:.1f} s × (n/400)^{f['alpha']:.2f}")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim(100, 1100)
    a.set_xticks([125, 250, 500, 800])
    a.set_xticklabels(["125", "250", "500", "800"])
    a.minorticks_off()
    a.set_ylim(0.2, 900)
    a.set_xlabel("points n")
    a.set_ylabel("seconds per diagram")
    leg = a.legend(loc="upper left", markerscale=2.5, handletextpad=0.3)
    for hnd in leg.legend_handles:
        hnd.set_alpha(1)
    a.set_title("(a) Diagram cost grows like n³")

    names = list(bud)[::-1]
    vals = [bud[k]["cpu_h"] for k in names]
    labels = {"training": "training\n60k, 3 k", "A": "A\n30k, 3 k", "B": "B\n60k, k = 5", "C": "C\n54k, k = 5"}
    b.barh(range(len(names)), vals, height=0.45, color=ORANGE, lw=0)
    for i, v in enumerate(vals):
        b.text(v + 25, i, f"{v:,.0f} CPU-h", va="center", fontsize=6.3, color=INK2)
    b.set_yticks(range(len(names)))
    b.set_yticklabels([labels[k] for k in names])
    b.set_xlim(0, max(vals) * 1.35)
    b.set_xlabel("persistence-diagram CPU-hours")
    b.grid(False, axis="y")
    b.spines["left"].set_visible(False)
    b.tick_params(axis="y", length=0)
    b.set_title(f"(b) Budget by set: {sum(vals):,.0f} CPU-h in total")
    save(fig, "gen_cost", preview)


def fig_gantt(preview):
    d0 = date(2026, 9, 15)

    def dd(s):
        mm, day = s.split("-")
        return (date(2026, int(mm), int(day)) - d0).days

    wps = [  # (label, start, end, kind)
        ("WP0  setup, freeze DV1/DV2", "9-15", "9-16", "code"),
        ("WP1  generation library", "9-15", "9-19", "code"),
        ("WP2  pilots", "9-18", "9-21", "cpu"),
        ("WP3  validation batches", "9-21", "9-23", "cpu"),
        ("WP4  bulk generation", "9-23", "9-25", "cpu"),
        ("WP5  diagrams, L / F / G", "9-25", "9-29", "cpu"),
        ("WP6  training-side code", "9-19", "9-26", "code"),
        ("WP7  training, evaluation", "9-28", "10-5", "gpu"),
        ("WP8  analysis, writing", "10-2", "10-12", "code"),
    ]
    col = {"code": BLUE, "cpu": ORANGE, "gpu": AQUA}
    fig, b = plt.subplots(figsize=(TEXTW, 2.45))
    ys = list(range(len(wps)))[::-1]
    for y, (lab, s, e, k) in zip(ys, wps):
        b.barh(y, dd(e) - dd(s), left=dd(s), height=0.42, color=col[k], lw=0)
    b.set_yticks(ys)
    b.set_yticklabels([w[0] for w in wps])
    b.tick_params(axis="y", length=0)
    gates = [("D1  prior, cells, F frozen", "9-21", 6), ("D2  samplers pass", "9-23", 5),
             ("D3  data pass", "9-25", 4), ("D4  A-level sanity", "9-30", 1)]
    for g, day, yrow in gates:
        b.plot(dd(day), yrow + 0.42, marker="D", ms=4.6, color=INK, mec="white", mew=0.8, zorder=4)
        b.text(dd(day) + 0.35, yrow + 0.5, g, fontsize=6.0, color=INK, va="center")
    b.axvline(dd("10-12"), color=MUTED, lw=0.8)
    b.text(dd("10-12") - 0.25, -0.55, "deadline (approx.)", ha="right", fontsize=6.0, color=INK2)
    ticks = ["9-15", "9-22", "9-29", "10-6", "10-12"]
    b.set_xticks([dd(t) for t in ticks])
    b.set_xticklabels(["Tue 15 Sep", "22 Sep", "29 Sep", "6 Oct", "12 Oct"])
    b.set_xlim(-0.3, dd("10-12") + 0.6)
    b.set_ylim(-0.8, len(wps) - 0.1)
    b.grid(False, axis="y")
    b.spines["left"].set_visible(False)
    for k, lab in [("code", "code / people"), ("cpu", "CPU jobs (SLURM)"), ("gpu", "GPU jobs")]:
        b.barh([-9], [0], color=col[k], label=lab)
    b.plot([], [], "D", ms=4.6, color=INK, mec="white", label="review gate")
    b.legend(loc="upper right", ncol=1, handlelength=1.1, bbox_to_anchor=(1.0, 1.0))
    save(fig, "gen_gantt", preview)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", default=None, help="also write PNG previews to this directory")
    args = ap.parse_args()
    if args.preview:
        Path(args.preview).mkdir(parents=True, exist_ok=True)
    for f in (fig_pipeline, fig_delta, fig_ladders, fig_s_vs_delta, fig_testsets, fig_cost, fig_gantt):
        f(args.preview)
        print("done", f.__name__)
