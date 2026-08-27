#!/usr/bin/env python3
# scripts/make_variance_identity_figure.py
"""Build the "canonical identity vs exact Var[N(W)]" figure requested by the
\\fig{} TODO in writeup/writeup.tex Section 5.3 (Validation): the textbook
canonical identity Var[N(W)] = kappa*mu*(1+mu) plotted against the exact
finite-window result of \\eqref{eq:varN}, as a function of the dimensionless
overlap index nu = sigma*sqrt(kappa), with the three empirical validation
points (Table sim-validation / audit/validate_generator.py's TRIPLES)
overlaid. This is what makes the "175% gap at nu=1" claim in the text visible
in one glance.

Both variance formulas and the three (kappa, mu, sigma) triples are NOT
reimplemented here: they are imported directly from audit/validate_generator.py
(var_N_exact, var_N_approx, TRIPLES), and the empirical mean/SE at each triple
is read from audit/results.json, the cached output of that script's own
Monte Carlo validation (3000 reps/triple) -- so this figure and the audit
numbers it visualizes can never silently drift apart. If that cache is
missing, we fall back to calling validate_generator.moment_test() directly,
which re-runs the same real generator code rather than approximating it here.

Usage:
    python scripts/make_variance_identity_figure.py

Output:
    figs/variance_identity.png
    figs/variance_identity.pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "audit"))

import validate_generator as vg  # noqa: E402  (var_N_exact, var_N_approx, TRIPLES)

FIG_DIR = ROOT / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = ROOT / "audit" / "results.json"

# Same palette/roles as audit/validate_generator.py: blue=theory, orange=empirical.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#8a8a86"
MARKERS = {"A_tight_cluster": "o", "B_moderate": "s", "C_near_poisson": "^"}
LINESTYLES = {"A_tight_cluster": "-", "B_moderate": "--", "C_near_poisson": ":"}
NICE_NAME = {
    "A_tight_cluster": "tight cluster",
    "B_moderate": "moderate",
    "C_near_poisson": "near-Poisson",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#c9c8c3",
    "axes.grid": True,
    "grid.color": "#e7e6e1",
    "grid.linewidth": 0.8,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_empirical():
    """Cached Monte Carlo moments per triple (mean_empirical/se not needed
    here, only var_empirical/var_se), falling back to a fresh run of
    validate_generator.moment_test() if the cache is absent."""
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            cached = json.load(f)
        if "moments" in cached:
            return cached["moments"]
    return vg.moment_test()


def main():
    empirical = load_empirical()

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    nu_grid = np.linspace(0.02, 1.3, 400)
    ax.axhline(1.0, color=GRAY, lw=1.5, ls="--", zorder=1,
               label=r"canonical identity $\kappa\mu(1+\mu)$")

    for label, params in vg.TRIPLES.items():
        kappa, mu = params["kappa"], params["mu"]
        v_naive = vg.var_N_approx(kappa, mu, params["sigma"])

        sigma_grid = nu_grid / np.sqrt(kappa)
        v_exact = np.array([vg.var_N_exact(kappa, mu, s) for s in sigma_grid])
        ratio = v_exact / v_naive
        ax.plot(nu_grid, ratio, color=BLUE, lw=2, ls=LINESTYLES[label], zorder=2,
                 label=rf"exact, {NICE_NAME[label]} ($\kappa$={kappa:g}, $\mu$={mu:g})")

        rec = empirical[label]
        nu_emp = rec["sigma_sqrt_kappa"]
        ratio_emp = rec["var_empirical"] / rec["var_approx_theory"]
        err_emp = rec["var_se"] / rec["var_approx_theory"]
        ax.errorbar(nu_emp, ratio_emp, yerr=err_emp, color=ORANGE,
                     marker=MARKERS[label], markersize=8, mec="black", mew=0.5,
                     capsize=3, lw=0, elinewidth=1.3, zorder=3)

    # Annotate the 175% gap called out in the text, at the near-Poisson design's nu=1.
    c = vg.TRIPLES["C_near_poisson"]
    nu_c = c["sigma"] * np.sqrt(c["kappa"])
    ratio_c = vg.var_N_exact(c["kappa"], c["mu"], c["sigma"]) / vg.var_N_approx(c["kappa"], c["mu"], c["sigma"])
    gap_pct = 100 * (1.0 - ratio_c) / ratio_c
    ax.annotate(
        "", xy=(nu_c, ratio_c), xytext=(nu_c, 1.0),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1),
    )
    ax.text(nu_c + 0.03, (1.0 + ratio_c) / 2, f"+{gap_pct:.0f}%",
             fontsize=9, va="center", ha="left")

    ax.set_xlabel(r"overlap index $\nu = \sigma\sqrt{\kappa}$")
    ax.set_ylabel(r"$\mathrm{Var}[N(W)]\;/\;\kappa\mu(1+\mu)$")
    ax.set_xlim(0, 1.3)
    ax.set_ylim(0, 1.08)
    ax.set_title("Canonical identity vs. the exact finite-window variance")
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    fig.tight_layout()
    png_path = FIG_DIR / "variance_identity.png"
    pdf_path = FIG_DIR / "variance_identity.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
