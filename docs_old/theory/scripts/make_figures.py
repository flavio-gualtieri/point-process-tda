#!/usr/bin/env python
"""Figures + numbers for docs/theory/notes.tex from docs/theory/scripts/out/*.npz
(written by theory_sims.py on SLURM). Plotting and light post-processing only:
vectorizing two saved diagrams and 1-D quadratures for the Thomas F/G/J formulas.

    python docs/theory/scripts/make_figures.py      # writes docs/theory/figs/*.pdf
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))
import departure as dep  # noqa: E402

OUT, FIGS = HERE / "out", HERE.parent / "figs"
FIGS.mkdir(exist_ok=True)

# Reference categorical palette (dataviz skill, light mode), fixed slot order.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#ffffff"] + BLUE_RAMP)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
W = 6.5  # \textwidth in inches (A4, 2.2 cm margins)
LAM = 400.0


PREVIEW = os.environ.get("THEORY_FIG_PREVIEW")  # optional dir for PNG previews


def save(fig, name):
    fig.savefig(FIGS / f"{name}.pdf")
    if PREVIEW:
        fig.savefig(Path(PREVIEW) / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"wrote figs/{name}.pdf")


def kfun_params():
    import theory_sims as ts  # parameter definitions only; nothing is simulated
    return ts.KFUN, ts.STRAUSS, ts.EXAMPLES


# ------------------------------------------------------------------ theory
def pcf_minus_1(name, kw):
    if name.startswith("thomas"):
        k, s = kw["parent_intensity"], kw["cluster_scale"]
        return lambda r: np.exp(-r**2 / (4 * s**2)) / (4 * np.pi * s**2 * k)
    if name == "matern_cluster":
        k, R = kw["parent_intensity"], kw["cluster_radius"]

        def f(r):
            d = np.clip(r / (2 * R), 0, 1)
            lens = 2 * R**2 * (np.arccos(d) - d * np.sqrt(1 - d**2))
            return lens / (np.pi * R**2) ** 2 / k
        return f
    if name.startswith("nested"):
        k, m1 = kw["meta_parent_intensity"], kw["meta_offspring"]
        s1, s2 = kw["meta_cluster_scale"], kw["cluster_scale"]
        v = s1**2 + s2**2
        return lambda r: (np.exp(-r**2 / (4 * s2**2)) / (4 * np.pi * s2**2 * k * m1)
                          + np.exp(-r**2 / (4 * v)) / (4 * np.pi * v * k))
    if name == "lgcp":
        return lambda r: np.exp(kw["sigma2"] * np.exp(-r / kw["s"])) - 1.0
    if name == "matern_ii":
        return lambda r: dep.matern2_pcf(r, kw["parent_intensity"], kw["hardcore_radius"]) - 1.0
    raise KeyError(name)


def K_closed(name, kw, r):
    if name.startswith("thomas"):
        return dep.K_thomas(r, kw["parent_intensity"], kw["cluster_scale"])
    if name == "matern_cluster":
        return dep.K_matern_cluster(r, kw["parent_intensity"], kw["cluster_radius"])
    if name.startswith("nested"):
        return dep.K_nested(r, kw["meta_parent_intensity"], kw["meta_offspring"],
                            kw["meta_cluster_scale"], kw["cluster_scale"])
    if name == "lgcp":
        return dep.K_lgcp_exp(r, kw["sigma2"], kw["s"])
    if name == "matern_ii":
        return dep.K_from_pcf(r, lambda t: dep.matern2_pcf(t, kw["parent_intensity"], kw["hardcore_radius"]))
    raise KeyError(name)


def anchoring_gamma(name, kw):
    """Gamma = int gamma_W(h) (g(h)-1) dh / |W|^2 on the unit square, using the
    isotropised set covariance 1 - 4r/pi + r^2/pi (r <= 1)."""
    r = np.linspace(0, 1, 200001)[1:]
    integrand = pcf_minus_1(name, kw)(r) * (1 - 4 * r / np.pi + r**2 / np.pi) * r
    return float(2 * np.pi * np.trapezoid(integrand, r))


def thomas_fgj(r_grid, kappa, mu, sigma, n_rho=4000):
    from scipy.stats import ncx2
    F, J = np.zeros_like(r_grid), np.ones_like(r_grid)
    for i, r in enumerate(r_grid):
        if r <= 0:
            continue
        rho = np.linspace(0.0, r + 10 * sigma, n_rho)
        p = ncx2.cdf((r / sigma) ** 2, df=2, nc=np.maximum((rho / sigma) ** 2, 1e-12))
        F[i] = 1 - np.exp(-kappa * np.trapezoid(2 * np.pi * rho * (1 - np.exp(-mu * p)), rho))
        ray = rho / sigma**2 * np.exp(-rho**2 / (2 * sigma**2))
        J[i] = np.trapezoid(ray * np.exp(-mu * p), rho)
    return F, 1 - J * (1 - F), J


# ------------------------------------------------------------------ figures
def fig_examples():
    z = np.load(OUT / "examples.npz")
    names = ["Poisson", "Thomas", "Matern cluster", "Anisotropic Thomas",
             "Nested Thomas", "LGCP", "Matern II", "Strauss"]
    labels = {"Matern cluster": "Matérn cluster", "Matern II": "Matérn II",
              "Anisotropic Thomas": "aniso. Thomas", "Nested Thomas": "nested Thomas"}
    fig, axes = plt.subplots(2, 4, figsize=(W, 3.55))
    for ax, nm in zip(axes.ravel(), names):
        p = z[f"{nm}__pts"]
        ax.scatter(p[:, 0], p[:, 1], s=1.4, c=INK, linewidths=0)
        ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[], aspect="equal")
        ax.grid(False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(True)
        ax.set_title(f"{labels.get(nm, nm)} ($n$={len(p)})", pad=2.5, fontsize=7)
    fig.tight_layout(w_pad=1.0, h_pad=1.4)
    save(fig, "fig_examples")


def fig_kfun(tabs):
    KFUN, _, _ = kfun_params()
    z = np.load(OUT / "kfun.npz")
    r = z["r_grid"]
    titles = {
        "thomas_A": r"Thomas A: $\kappa$=15, $c$=0.3",
        "thomas_B": r"Thomas B: $\kappa$=60, $c$=0.3",
        "thomas_C": r"Thomas C: $\kappa$=60, $c$=0.9",
        "matern_cluster": r"Matérn cl.: $\kappa$=30, $c$=0.4",
        "nested_A": r"nested A: $\kappa$=20, $\mu_1$=4, $\sigma_1/\sigma_2$=10",
        "nested_B": r"nested B: $\kappa$=40, $\mu_1$=3, $\sigma_1/\sigma_2$=5",
        "lgcp": r"LGCP: $\sigma^2$=1, $s$=0.05",
        "matern_ii": r"Matérn II: $R$=0.025",
    }
    fig, axes = plt.subplots(2, 4, figsize=(W, 3.6), sharex=True)
    rows, sel = [], (r >= 0.01)
    for ax, nm in zip(axes.ravel(), KFUN):
        kw = KFUN[nm][1]
        n = z[f"{nm}__n"].astype(float)
        lmr = z[f"{nm}__lmr"].astype(float)
        K_repo = np.pi * (lmr + r) ** 2
        K_lam = K_repo * (n * (n - 1) / LAM**2)[:, None]
        K_th = K_closed(nm, kw, r)
        gam = anchoring_gamma(nm, kw)
        L_th = dep.l_minus_r(K_th, r)
        L_lam = dep.l_minus_r(K_lam.mean(0), r)
        L_rep = dep.l_minus_r(K_repo.mean(0), r)
        L_pred = dep.l_minus_r(K_th / (1 + gam), r)
        se_K = 2 * K_lam.std(0, ddof=1) / math.sqrt(len(n))
        lo, hi = dep.l_minus_r(K_lam.mean(0) - se_K, r), dep.l_minus_r(K_lam.mean(0) + se_K, r)
        se = hi - L_lam
        ax.fill_between(r, lo, hi, color=C[0], alpha=0.18, linewidth=0, label=r"$\pm$2 MC s.e.")
        ax.plot(r, L_th, color=INK, lw=1.3, label="closed form")
        ax.plot(r, L_lam, color=C[0], lw=1.2, ls=(0, (4, 2)), label=r"mean $\hat K$, true $\lambda$")
        ax.plot(r, L_rep, color=C[1], lw=1.2, label=r"mean $\hat K$, repo ($\hat\lambda^2=n(n-1)$)")
        ax.plot(r, L_pred, color=C[1], lw=0.9, ls=":", label=r"first-order: $K/(1+\Gamma)$")
        if nm == "matern_cluster":
            twin = dep.K_thomas(r, kw["parent_intensity"], kw["cluster_radius"] / 2)
            tl, = ax.plot(r, dep.l_minus_r(twin, r), color=C[2], lw=1.0, ls=(0, (1.5, 1)),
                          label="Thomas twin ($\\sigma=R/2$)")
            ax.legend(handles=[tl], loc="lower center", handlelength=2.2, fontsize=6)
        ax.axhline(0, color=INK2, lw=0.5)
        ax.set_title(titles[nm], pad=3, fontsize=6.8)
        rows.append((nm, gam, float(np.abs(L_lam - L_th)[sel].max()), float(se[sel].max()),
                     float(np.abs(L_rep - L_th)[sel].max()), float(np.abs(L_rep - L_pred)[sel].max()),
                     float(np.abs(L_th[sel]).max())))
    for ax in axes[1]:
        ax.set_xlabel("$r$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$L(r)-r$")
    h, l = axes[0, 0].get_legend_handles_labels()
    order = [1, 2, 0, 3, 4]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.045), handlelength=2.4, columnspacing=1.2)
    fig.tight_layout(w_pad=0.5, h_pad=0.7)
    save(fig, "fig_kfun")
    print("\nK/L check (r in [0.01, 0.25]); all in units of L(r)-r")
    print(f"{'setting':15s} {'Gamma':>7s} {'|lam-th|':>9s} {'2SE':>7s} {'|repo-th|':>9s} {'|repo-pred|':>11s} {'max|L-r|':>8s}")
    for row in rows:
        print(f"{row[0]:15s} {row[1]:7.4f} {row[2]:9.5f} {row[3]:7.5f} {row[4]:9.5f} {row[5]:11.5f} {row[6]:8.4f}")
    # within-parameter spread of the departure statistic
    print("\nWithin-parameter spread of S (400 replicates per setting): q05 / q50 / q95, frac(S<=1)")
    spread = {}
    for nm in KFUN:
        n = z[f"{nm}__n"].astype(float)
        for fn, key in (("L", "lmr"), ("G", "g"), ("F", "f")):
            s = dep.departure(z[f"{nm}__{key}"].astype(float), n, fn, tabs)["S_max"]
            spread[(nm, fn)] = s
        print(f"  {nm:15s} " + "  ".join(
            f"{fn}: {np.quantile(spread[(nm, fn)], .05):6.2f} {np.median(spread[(nm, fn)]):6.2f} "
            f"{np.quantile(spread[(nm, fn)], .95):6.2f} ({np.mean(spread[(nm, fn)] <= 1):.2f})"
            for fn in "LGF"))
    return spread


def fig_checks():
    KFUN, STRAUSS, _ = kfun_params()
    from cloudforger.baselines import summstats
    z = np.load(OUT / "kfun.npz")
    rf = z["fg_grid"]
    fig, axes = plt.subplots(1, 4, figsize=(W, 1.85))
    lam = LAM
    pois = 1 - np.exp(-lam * np.pi * rf**2)
    print("\nThomas F/G/J: max |repo mean - formula| over the plotted range")
    for i, nm in enumerate(("thomas_A", "thomas_B", "thomas_C")):
        kw = KFUN[nm][1]
        f, g = z[f"{nm}__f"].astype(float), z[f"{nm}__g"].astype(float)
        j = np.stack([summstats.j_function(a, b) for a, b in zip(f, g)])
        rr = rf[::2]
        Fth, Gth, Jth = thomas_fgj(rr, kw["parent_intensity"], kw["mean_offspring"], kw["cluster_scale"])
        lab = nm.replace("thomas_", "Thomas ")
        for ax, est, th, rmax in ((axes[0], f, Fth, 0.25), (axes[1], g, Gth, 0.08), (axes[2], j, Jth, 0.06)):
            m = rr <= rmax
            ax.plot(rr[m], th[m], color=C[i], lw=1.2, label=lab)
            ax.plot(rf[rf <= rmax], est.mean(0)[rf <= rmax], color=C[i], lw=0, marker="o", ms=1.6,
                    markevery=6)
        mf, mg = rf[::2] <= 0.25, rf[::2] <= 0.08
        mj = rf[::2] <= 0.06
        print(f"  {nm}: F {np.abs(f.mean(0)[::2][mf] - Fth[mf]).max():.4f}  "
              f"G {np.abs(g.mean(0)[::2][mg] - Gth[mg]).max():.4f}  "
              f"J(r<=0.06) {np.abs(j.mean(0)[::2][mj] - Jth[mj]).max():.4f}")
    for ax, ttl, rmax in ((axes[0], "$F(r)$ empty space", 0.25), (axes[1], "$G(r)$ nearest neighbour", 0.08),
                          (axes[2], "$J(r)=(1-G)/(1-F)$", 0.06)):
        m = rf <= rmax
        ref = np.ones(m.sum()) if "J" in ttl else pois[m]
        ax.plot(rf[m], ref, color=INK2, lw=0.8, ls="--", label="Poisson")
        ax.set_title(ttl, pad=3)
        ax.set_xlabel("$r$")
    axes[0].legend(loc="lower right", handlelength=1.8)
    axes[2].set_yscale("log")
    # Strauss: repo MH budget (x1) vs 10x
    s = np.load(OUT / "strauss.npz")
    r = s["r_grid"]
    for i, nm in enumerate(STRAUSS):
        for mult, ls in ((1, "-"), (10, (0, (3, 1.5)))):
            lmr = s[f"{nm}__x{mult}__lmr"].astype(float)
            axes[3].plot(r, lmr.mean(0), color=C[i], lw=1.0 if mult == 1 else 1.2, ls=ls,
                         label=nm.replace("_", " ") if mult == 1 else None)
    axes[3].axhline(0, color=INK2, lw=0.5)
    axes[3].set(title="Strauss $L(r)-r$: MH 1$\\times$ vs 10$\\times$", xlabel="$r$", xlim=(0, 0.12))
    axes[3].legend(loc="center right", handlelength=1.5, fontsize=6)
    for ax, tag in zip(axes, "abcd"):
        ax.set_title(f"({tag}) " + ax.get_title(), pad=3)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_checks")
    print("\nStrauss: mean n (se) at x1 / x10, PS approx, max |mean L x1 - mean L x10| (r<=0.12) vs 2*SE")
    for nm in STRAUSS:
        a, b = s[f"{nm}__x1__n"], s[f"{nm}__x10__n"]
        la, lb = s[f"{nm}__x1__lmr"].astype(float), s[f"{nm}__x10__lmr"].astype(float)
        m = r <= 0.12
        d = np.abs(la.mean(0) - lb.mean(0))[m].max()
        se = 2 * np.sqrt(la.var(0, ddof=1) / len(la) + lb.var(0, ddof=1) / len(lb))[m].max()
        print(f"  {nm:9s} {a.mean():6.1f} ({a.std(ddof=1) / np.sqrt(len(a)):.1f}) / {b.mean():6.1f} "
              f"({b.std(ddof=1) / np.sqrt(len(b)):.1f})  PS {float(s[nm + '__lam_ps']):6.1f}  "
              f"dL {d:.4f} vs 2SE {se:.4f}")


def fig_ph():
    from cloudforger.core.diagram import PersistenceDiagram
    from cloudforger.calibration.diagram_calibration import axis_bounds, axis_bounds_1d
    from cloudforger.vectorization.persistence_images.persistence_image import PersistenceImager
    from cloudforger.vectorization.scalar_features.betti_curve import BettiCurve
    from cloudforger.vectorization.landscapes.landscape_silhouette import (
        LandscapeTransformer, SilhouetteTransformer)

    z = np.load(OUT / "ph.npz")

    def pdg(tag, filt):
        return PersistenceDiagram(diagrams={0: z[f"{tag}__{filt}_h0"], 1: z[f"{tag}__{filt}_h1"]},
                                  generator_name=tag, generator_params={})

    th_r, th_d, cs_d = pdg("thomas", "rips"), pdg("thomas", "dtm"), pdg("csr", "dtm")
    fig, axes = plt.subplots(2, 4, figsize=(W, 3.45))
    # (a) cloud coloured by 2*DTM_5 (darker = denser)
    ax = axes[0, 0]
    p, v = z["thomas__pts"], 2 * z["thomas__dtm_vals"]
    ax.scatter(p[:, 0], p[:, 1], s=2.2, c=-v, cmap=SEQ, vmin=-np.quantile(v, 0.98), linewidths=0)
    ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[], aspect="equal",
           title=f"(a) Thomas, n={len(p)}; shade = DTM$_5$")
    ax.grid(False)

    def dgm_panel(ax, dg, title, ref=None):
        top = 0
        if ref is not None:
            for d in (0, 1):
                q = ref.finite_pairs(d)
                ax.scatter(q[:, 0], q[:, 1], s=3, facecolors="none", edgecolors="#b8b7b2", linewidths=0.4,
                           label="CSR (same n)" if d == 0 else None)
                top = max(top, q[:, 1].max())
        for d, col, mk in ((0, C[0], "o"), (1, C[1], "^")):
            q = dg.finite_pairs(d)
            ax.scatter(q[:, 0], q[:, 1], s=4, color=col, marker=mk, linewidths=0, label=f"$H_{d}$")
            top = max(top, q[:, 1].max())
        ax.plot([0, top * 1.05], [0, top * 1.05], color=INK2, lw=0.5)
        ax.set(xlim=(-0.01 * top, top * 1.05), ylim=(0, top * 1.05), xlabel="birth", ylabel="death", title=title)
        ax.legend(loc="lower right", handletextpad=0.2, markerscale=1.8, fontsize=5.8,
                  bbox_to_anchor=(1.04, -0.02), borderaxespad=0.1)

    dgm_panel(axes[0, 1], th_r, "(b) Rips diagram")
    dgm_panel(axes[0, 2], th_d, "(c) DTM-5 diagram", ref=cs_d)
    # (d) Betti curves, DTM
    hi = 1.1 * max(d.finite_pairs(k)[:, 1].max() for d in (th_d, cs_d) for k in (0, 1))
    bc = BettiCurve(homology_dims=(0, 1), grid_size=128, grid_range=(0.0, hi))
    ax = axes[0, 3]
    for dg, ls, tag in ((th_d, "-", "Thomas"), (cs_d, (0, (3, 1.5)), "CSR")):
        cur = bc.compute(dg).curves
        ax.plot(bc.grid, cur[0], color=C[0], ls=ls, label=rf"$\beta_0$ {tag}")
        ax.plot(bc.grid, cur[1], color=C[1], ls=ls, label=rf"$\beta_1$ {tag}")
    ax.set(xlabel="filtration value $t$", title="(d) Betti curves (DTM-5)")
    ax.legend(loc="upper right", fontsize=5.8, handlelength=1.8)
    # (e, f) persistence images, repo settings: 64x64, sigma = 0.5 px, box from both diagrams
    for ax, d in ((axes[1, 0], 0), (axes[1, 1], 1)):
        (b_lo, b_hi), (_, p_hi) = axis_bounds([th_d, cs_d], d, coverage=1.0, pad=1.05)
        img = PersistenceImager((b_lo, b_hi), (0.0, p_hi), resolution=64, sigma_pixels=0.5).transform(th_d, d)
        ax.imshow(img, cmap=SEQ, extent=(b_lo, b_hi, 0, p_hi), aspect="auto", interpolation="nearest")
        ax.set(xlabel="birth", ylabel="persistence", title=f"(e) PI $H_{d}$, 64$^2$, $\\sigma$=0.5 px"
               if d == 0 else f"(f) PI $H_{d}$, 64$^2$, $\\sigma$=0.5 px")
        ax.grid(False)
    # (g) landscape of DTM H1
    t_min, T = axis_bounds_1d([th_d, cs_d], 1, q=1.0, pad_factor=1.05)
    lt = LandscapeTransformer(t_min, T, 128, K=5)
    lam = lt.transform(th_d, 1)
    ax = axes[1, 2]
    for k in range(5):
        ax.plot(lt.grid, lam[k], color=BLUE_RAMP[-1 - k], lw=1.0, label=rf"$\lambda_{k + 1}$")
    ax.set(xlabel="$t$", title="(g) landscape $H_1$, $K$=5")
    ax.legend(loc="upper right", fontsize=5.8, handlelength=1.2, ncol=1)
    # (h) silhouette p=1 of DTM H1
    st = SilhouetteTransformer(t_min, T, 128, p=1.0)
    ax = axes[1, 3]
    ax.plot(st.grid, st.transform(th_d, 1)["silhouette"], color=C[0], label="Thomas")
    ax.plot(st.grid, st.transform(cs_d, 1)["silhouette"], color=INK2, ls=(0, (3, 1.5)), label="CSR")
    ax.set(xlabel="$t$", title="(h) silhouette $H_1$, $p$=1")
    ax.legend(loc="upper right", fontsize=6)
    fig.tight_layout(w_pad=0.4, h_pad=0.6)
    save(fig, "fig_ph")
    for tag in ("thomas", "csr"):
        for filt in ("rips", "dtm"):
            h1 = z[f"{tag}__{filt}_h1"]
            pers = h1[:, 1] - h1[:, 0]
            print(f"  ph {tag:6s} {filt:4s}: #H1={len(h1):3d} max pers={pers.max():.4f} "
                  f"max d/b={np.max(h1[:, 1] / h1[:, 0]):.2f}")
    print(f"  CSR MST-edge NN share {z['csr_mst_nn_frac'].mean():.3f}; "
          f"E[DTM5^2]={z['csr_dtm2'].mean():.4e} vs {4 / (2 * np.pi * LAM):.4e}")


def fig_departure(spread):
    KFUN, _, _ = kfun_params()
    z = np.load(OUT / "data_departure.npz")
    procs = [("thomas", "Thomas"), ("matern_cluster", "Matérn cl."), ("nested_thomas", "nested"),
             ("aniso_thomas", "aniso"), ("strauss", "Strauss")]
    styles = ["-", (0, (4, 1.5)), "-", (0, (1.5, 1)), "-"]
    fig, axes = plt.subplots(1, 4, figsize=(W, 2.05), gridspec_kw={"width_ratios": [1, 1, 1, 1.3]})
    for ax, fn, ttl, tag in zip(axes[:3], "LGF", ("$S_L$", "$S_G$", "$S_F$"), "abc"):
        ax.axvspan(1e-2, 1.0, color="#f0efec", zorder=0)
        for i, (p, lab) in enumerate(procs):
            s = np.sort(z[f"{p}__test__{fn}_S_max"])
            ax.plot(s, np.arange(1, len(s) + 1) / len(s), color=C[i], ls=styles[i], lw=1.1, label=lab)
        ax.axvline(1.0, color=INK2, lw=0.6)
        ax.set(xscale="log", xlim=(0.2, 400), ylim=(0, 1), xlabel=ttl, title=f"({tag}) ECDF of {ttl}")
    axes[0].set_ylabel("fraction of clouds")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.4, 1.07), handlelength=2.4,
               columnspacing=1.4)
    # within-parameter spread (synthetic settings)
    ax = axes[3]
    order = ["thomas_A", "thomas_B", "thomas_C", "matern_cluster", "nested_A", "nested_B", "lgcp", "matern_ii"]
    lab = {"thomas_A": "Thomas A", "thomas_B": "Thomas B", "thomas_C": "Thomas C",
           "matern_cluster": "Matérn cl.", "nested_A": "nested A", "nested_B": "nested B",
           "lgcp": "LGCP", "matern_ii": "Matérn II"}
    s_str = np.load(OUT / "strauss.npz")
    tabs = dep.null_tables(OUT / "null.npz")
    extra = []
    for nm in ("weak", "moderate", "strong", "small_R"):
        n = s_str[f"{nm}__x10__n"].astype(float)
        extra.append((f"Strauss {nm.replace('_', ' ')}",
                      dep.departure(s_str[f"{nm}__x10__lmr"].astype(float), n, "L", tabs)["S_max"]))
    rows = [(lab[nm], spread[(nm, "L")]) for nm in order] + extra
    ax.axvspan(1e-2, 1.0, color="#f0efec", zorder=0)
    for y, (name, s) in enumerate(rows):
        q = np.quantile(s, [0.05, 0.25, 0.5, 0.75, 0.95])
        ax.plot([q[0], q[4]], [y, y], color=C[0], lw=0.8)
        ax.plot([q[1], q[3]], [y, y], color=C[0], lw=3.0, solid_capstyle="butt")
        ax.plot(q[2], y, "o", color="white", mec=C[0], ms=3, mew=0.8)
    ax.axvline(1.0, color=INK2, lw=0.6)
    ax.set(xscale="log", xlim=(0.2, 400), yticks=range(len(rows)), ylim=(len(rows) - 0.5, -0.5),
           xlabel="$S_L$", title="(d) $S_L$ at fixed parameters")
    ax.set_yticklabels([r[0] for r in rows], fontsize=5.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout(w_pad=0.5)
    save(fig, "fig_departure")
    print("\nCurrent data: fraction with S<=1 and S<=2 (test pool) and Spearman(S_L, S_L*)")
    from scipy.stats import spearmanr
    for p, labp in procs:
        msg = "  ".join(f"{fn}: {np.mean(z[f'{p}__test__{fn}_S_max'] <= 1):.3f}/"
                        f"{np.mean(z[f'{p}__test__{fn}_S_max'] <= 2):.3f}" for fn in "LGF")
        key = f"{p}__test__Lpop_S_max"
        if key in z.files:
            rho = spearmanr(z[f"{p}__test__L_S_max"], z[key]).statistic
            msg += f"  spearman(S_L,S_L*)={rho:.3f}"
        print(f"  {labp:11s} {msg}")
    for name, s in extra:
        print(f"  {name:18s} S_L q05/q50/q95 = {np.quantile(s, .05):.2f} {np.median(s):.2f} "
              f"{np.quantile(s, .95):.2f}  frac<=1 {np.mean(s <= 1):.2f}")


def strauss_heuristic():
    """Spearman between the data's S_L and simple dimensionless signal indices."""
    from scipy.stats import spearmanr
    z = np.load(OUT / "data_departure.npz")
    n = z["thomas__test__n"]
    P = z["thomas__test__params"]  # parent_intensity, mean_offspring, cluster_scale
    s = P[:, 2] * np.sqrt(n)
    idx = P[:, 1] * np.sqrt(n) / s
    print(f"\nThomas: spearman(S_L, mu*sqrt(n)/s) = {spearmanr(z['thomas__test__L_S_max'], idx).statistic:.3f}")
    n = z["strauss__test__n"]
    P = z["strauss__test__params"]  # beta, gamma, radius
    tau = P[:, 2] * np.sqrt(n)
    idx = np.sqrt(n) * tau * (1 - P[:, 1])
    for fn in "LG":
        print(f"Strauss: spearman(S_{fn}, sqrt(n) tau (1-gamma)) = "
              f"{spearmanr(z[f'strauss__test__{fn}_S_max'], idx).statistic:.3f}")
    print(f"Strauss: fraction with pi tau^2 (1-gamma) < 0.05: {np.mean(np.pi * tau**2 * (1 - P[:, 1]) < 0.05):.3f}; "
          f"tau range {tau.min():.3f}-{tau.max():.3f}")


def main():
    tabs = dep.null_tables(OUT / "null.npz")
    fig_examples()
    spread = fig_kfun(tabs)
    fig_checks()
    fig_ph()
    fig_departure(spread)
    strauss_heuristic()
    m = np.load(OUT / "matern2.npz")
    print(f"\nMatern II: as-is mean n {float(m['asis__mean_n']):.1f}, edge ratio {float(m['asis__edge_ratio']):.3f}; "
          f"dilated {float(m['dilated__mean_n']):.1f}, {float(m['dilated__edge_ratio']):.3f}; "
          f"theory {float(m['lambda_theory']):.1f}")


if __name__ == "__main__":
    main()
