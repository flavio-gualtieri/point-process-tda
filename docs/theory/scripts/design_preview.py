"""Closed-form preview of the PROVISIONAL DV3 design, for docs/theory/generation.tex.

Evaluates the provisional prior and the B/C test-set designs using closed-form K
functions (departure.py) and the saved CSR null tables (out/null.npz, SLURM job
26430628). No point pattern is simulated. Strauss has no closed-form K, so it is
previewed only from the 8,000 existing DV1 Strauss clouds (parameters and counts).

delta_tilde below is the notes' S applied to the closed-form L(r) - r at n = nbar
(departure(..., center=False)): the design-time approximation of delta defined in
generation.tex, Sec. 2.3. It ignores the estimator's own bias (the lambda-hat^2
normalization), which the replicate-based delta-hat of B and C includes.

    python docs/theory/scripts/design_preview.py      # seconds; writes out/design_preview.{json,npz}
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

import departure as dep

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "out"
TABS = dep.null_tables(OUT / "null.npz")
R = TABS["r_grid"]
RNG = np.random.default_rng(20260914)          # preview draws only; not a DV3 stream

NBAR_RANGE = (100.0, 800.0)
NB_B = (125.0, 250.0, 500.0)
NB_B_NESTED = (250.0, 500.0)
NB_C = (125.0, 500.0)
NB_C_NESTED = (250.0, 500.0)
DELTA_B = (0.5, 1.0, 2.0, 4.0, 8.0)
DELTA_FIG = (0.25, 0.5, 1.0, 2.0, 4.0)

PRIOR = {  # provisional ranges (generation.tex Table 3); all log-uniform
    "thomas": {"mu": (0.1, 30.0), "s": (0.1, 1.5)},
    "nested": {"mu1": (1.5, 6.0), "mu2": (0.03, 20.0), "s2": (0.05, 0.5), "rho": (3.0, 12.0)},
    "matern2": {"tau": (0.05, 0.55)},
    "lgcp": {"sigma2": (0.05, 3.0), "sp": (0.5, 2.0)},
}
B_SHAPES = {"thomas": [{"s": 0.2}, {"s": 0.45}, {"s": 1.0}],
            "nested": [{"s2": 0.1, "mu1": 3.0, "rho": 6.0}, {"s2": 0.25, "mu1": 3.0, "rho": 6.0}],
            "lgcp": [{"sp": 0.6}, {"sp": 0.9}, {"sp": 1.4}],
            "matern2": [{}]}
C_SHAPE = {"thomas": {"s": 1.0}, "nested": {"s2": 0.2, "mu1": 3.0, "rho": 6.0},
           "lgcp": {"sp": 1.0}, "matern2": {}}
AMP = {"thomas": "mu", "nested": "mu2", "lgcp": "sigma2", "matern2": "tau"}


def lu(lo, hi, size):
    return np.exp(RNG.uniform(np.log(lo), np.log(hi), size))


def interior(lo, hi, frac=0.1):
    """Central 80% of a log-uniform marginal."""
    return lo * (hi / lo) ** frac, lo * (hi / lo) ** (1 - frac)


# ------------------------------------------------------------------ model maps
def constraints_ok(fam, p, nbar):
    if fam == "thomas":
        return (p["s"] / np.sqrt(nbar) <= 0.1) & (nbar / p["mu"] >= 5.0)
    if fam == "nested":
        s2 = p["s2"] / np.sqrt(nbar)
        s1 = p["rho"] * s2
        return (nbar / (p["mu1"] * p["mu2"]) >= 15.0) & (2 * np.sqrt(s1 ** 2 + s2 ** 2) <= 0.2)
    if fam == "lgcp":
        return p["sp"] / np.sqrt(nbar) <= 0.15
    return np.ones_like(np.asarray(nbar, float), bool)


def K_curves(fam, p, nbar):
    """Closed-form K on R for arrays of design coordinates (all shape (m,))."""
    nbar = np.asarray(nbar, float)
    r = R[None, :]
    if fam == "thomas":
        return dep.K_thomas(r, (nbar / p["mu"])[:, None], (p["s"] / np.sqrt(nbar))[:, None])
    if fam == "nested":
        s2 = p["s2"] / np.sqrt(nbar)
        return dep.K_nested(r, (nbar / (p["mu1"] * p["mu2"]))[:, None], p["mu1"][:, None],
                            (p["rho"] * s2)[:, None], s2[:, None])
    if fam == "lgcp":
        return dep.K_lgcp_exp(r, np.asarray(p["sigma2"], float)[:, None], (p["sp"] / np.sqrt(nbar))[:, None])
    if fam == "matern2":
        out = np.empty((len(nbar), len(R)))
        tau = np.asarray(p["tau"], float)
        for i in range(len(nbar)):
            Rh = tau[i] / np.sqrt(nbar[i])
            lam_p = -np.log(1 - np.pi * tau[i] ** 2) / (np.pi * Rh ** 2)
            out[i] = dep.K_from_pcf(R, lambda t: dep.matern2_pcf(t, lam_p, Rh))
        return out
    raise ValueError(fam)


def delta_tilde(fam, p, nbar):
    nbar = np.atleast_1d(np.asarray(nbar, float))
    p = {k: np.broadcast_to(np.asarray(v, float), nbar.shape).copy() for k, v in p.items()}
    curves = dep.l_minus_r(K_curves(fam, p, nbar), R[None, :])
    return dep.departure(curves, nbar, "L", TABS, center=False)["S_max"]


def draw_shape(fam, nbar, m):
    """m shape draws from the provisional prior, truncated by the constraints GIVEN nbar."""
    acc, tries = {k: [] for k in PRIOR[fam]}, 0
    while len(next(iter(acc.values()))) < m:
        k = 4 * m
        cand = {name: lu(*rng, k) for name, rng in PRIOR[fam].items()}
        ok = constraints_ok(fam, cand, np.full(k, nbar))
        tries += k
        for name in acc:
            acc[name].extend(cand[name][ok].tolist())
    n_acc = sum(1 for _ in acc[next(iter(acc))])
    return {k: np.array(v[:m]) for k, v in acc.items()}, n_acc / tries


# ------------------------------------------------------------------ coverage
report: dict = {"prior": PRIOR, "coverage": {}, "acceptance": {}}
fig_draws = {}
for fam in PRIOR:
    report["coverage"][fam], report["acceptance"][fam] = {}, {}
    for nb in NB_B:
        shp, rate = draw_shape(fam, nb, 1500)
        d = delta_tilde(fam, shp, np.full(1500, nb))
        report["coverage"][fam][str(int(nb))] = {
            "frac_delta_le_1": float((d <= 1).mean()), "frac_delta_ge_10": float((d >= 10).mean()),
            "q05": float(np.quantile(d, .05)), "q50": float(np.median(d)), "q95": float(np.quantile(d, .95))}
        report["acceptance"][fam][str(int(nb))] = float(rate)
    # full prior (nbar log-uniform) for the test-set figure
    nb_all = lu(*NBAR_RANGE, 1500)
    shp = {k: np.empty(1500) for k in PRIOR[fam]}
    for i, nb in enumerate(nb_all):
        one, _ = draw_shape(fam, nb, 1)
        for k in shp:
            shp[k][i] = one[k][0]
    fig_draws[fam] = (nb_all, delta_tilde(fam, shp, nb_all))
    print(f"coverage {fam:8s}", {k: (round(v['frac_delta_le_1'], 3), round(v['frac_delta_ge_10'], 3))
                                  for k, v in report["coverage"][fam].items()},
          "accept", {k: round(v, 3) for k, v in report["acceptance"][fam].items()})


# ------------------------------------------------------------------ B cells and C ladders
def solve_amp(fam, shape, nbar, target, lo=1e-3, hi=1e3):
    amp = AMP[fam]
    if fam == "matern2":
        hi = 0.5635

    def f(a):
        return delta_tilde(fam, {**shape, amp: a}, nbar)[0] - target
    a, b = np.log(lo), np.log(hi)
    fa, fb = f(np.exp(a)), f(np.exp(b))
    if fa > 0 or fb < 0:
        return None
    for _ in range(45):
        mid = 0.5 * (a + b)
        if f(np.exp(mid)) < 0:
            a = mid
        else:
            b = mid
    return float(np.exp(0.5 * (a + b)))


def cell_ok(fam, shape, amp_val, nbar):
    p = {**shape, AMP[fam]: amp_val}
    full = {k: np.array([v]) for k, v in p.items()}
    if not constraints_ok(fam, full, np.array([nbar]))[0]:
        return False, "constraint"
    for name, rng in PRIOR[fam].items():
        lo, hi = interior(*rng)
        if not (lo <= p[name] <= hi):
            return False, f"{name} outside interior [{lo:.3g},{hi:.3g}]"
    return True, ""


report["B"], report["C"] = {}, {}
for fam in B_SHAPES:
    cells = []
    for nb in (NB_B_NESTED if fam == "nested" else NB_B):
        for shape in B_SHAPES[fam]:
            for dt in DELTA_B:
                a = solve_amp(fam, shape, np.array([nb]), dt)
                ok, why = (False, "unreachable") if a is None else cell_ok(fam, shape, a, nb)
                cells.append({"nbar": nb, **shape, "delta": dt, AMP[fam]: a, "feasible": bool(ok), "why": why})
    report["B"][fam] = {"cells": cells, "n_feasible": int(sum(c["feasible"] for c in cells)), "n_total": len(cells)}
    print(f"B {fam:8s} feasible {report['B'][fam]['n_feasible']}/{len(cells)}")

    ladders = {}
    for nb in (NB_C_NESTED if fam == "nested" else NB_C):
        shape = C_SHAPE[fam]
        lo, hi = interior(*PRIOR[fam][AMP[fam]])
        # respect the family's constraints at this nbar
        if fam == "thomas":
            hi = min(hi, nb / 5.0)
        if fam == "nested":
            hi = min(hi, nb / (15.0 * shape["mu1"]))
        dlo = float(delta_tilde(fam, {**shape, AMP[fam]: lo}, np.array([nb]))[0])
        dhi = float(delta_tilde(fam, {**shape, AMP[fam]: hi}, np.array([nb]))[0])
        ladders[str(int(nb))] = {"amp_interior": [lo, hi], "delta_at_interior": [dlo, dhi],
                                 "delta_range_used": [max(0.1, dlo), min(10.0, dhi)]}
    report["C"][fam] = ladders
    print(f"C {fam:8s}", {k: [round(x, 3) for x in v["delta_at_interior"]] for k, v in ladders.items()})

# ladder curves for the figure: delta-tilde targets at nbar = 250
fig_ladders = {}
for fam in C_SHAPE:
    rows = []
    for dt in DELTA_FIG:
        a = solve_amp(fam, C_SHAPE[fam], np.array([250.0]), dt)
        if a is None:
            rows.append((dt, np.nan, np.full(len(R), np.nan)))
            continue
        p = {k: np.array([v]) for k, v in {**C_SHAPE[fam], AMP[fam]: a}.items()}
        rows.append((dt, a, dep.l_minus_r(K_curves(fam, p, np.array([250.0])), R[None, :])[0]))
    fig_ladders[fam] = rows
report["fig_ladder_amplitudes_nbar250"] = {f: [(dt, a) for dt, a, _ in rows] for f, rows in fig_ladders.items()}


# ------------------------------------------------------------------ worked example + tau conversions
ex = {"nbar": 312.0, "mu": 4.0, "s": 0.4}
kappa, sigma = ex["nbar"] / ex["mu"], ex["s"] / np.sqrt(ex["nbar"])
b = sigma * 3.719016485
report["worked_thomas"] = {**ex, "kappa": kappa, "sigma": sigma, "buffer": b,
                           "expanded_area": (1 + 2 * b) ** 2, "mean_parents": kappa * (1 + 2 * b) ** 2,
                           "tau": 2 * np.sqrt(np.log(2)) * ex["s"],
                           "delta_tilde": float(delta_tilde("thomas", {"mu": 4.0, "s": 0.4}, np.array([312.0]))[0])}


def r_half(r, excess, limit):
    return float(r[np.argmax(np.abs(excess) >= 0.5 * abs(limit))])


conv = {"lgcp_rhalf_over_s": {}, "matern2_rhalf_over_R": {}}
s = 0.02
rr = np.linspace(0, 40 * s, 40001)
for s2v in (0.1, 1.0, 3.0):
    exc = dep.K_lgcp_exp(rr, s2v, s) - np.pi * rr ** 2
    k = np.arange(1, 120)
    from math import lgamma
    lim = 2 * np.pi * s ** 2 * np.sum(np.exp(k * np.log(s2v) - np.array([lgamma(i + 1) for i in k])) / k ** 2)
    conv["lgcp_rhalf_over_s"][str(s2v)] = r_half(rr, exc, lim) / s
for tau in (0.1, 0.3, 0.5):
    Rh = tau / np.sqrt(250.0)
    lam_p = -np.log(1 - np.pi * tau ** 2) / (np.pi * Rh ** 2)
    rr = np.linspace(0, 12 * Rh, 24001)
    exc = dep.K_from_pcf(rr, lambda t: dep.matern2_pcf(t, lam_p, Rh), n_fine=48001) - np.pi * rr ** 2
    conv["matern2_rhalf_over_R"][str(tau)] = r_half(rr, exc, exc[-1]) / Rh
report["tau_conversions"] = conv
print("tau conversions", conv)


# ------------------------------------------------------------------ Strauss from existing DV1 clouds
recs = pickle.load(open(ROOT / "data/strauss/clouds.pkl", "rb")) + pickle.load(open(ROOT / "data/strauss/adversarial_clouds.pkl", "rb"))
beta = np.array([r["params"]["beta"] for r in recs])
gam = np.array([r["params"]["gamma"] for r in recs])
rad = np.array([r["params"]["radius"] for r in recs])
n = np.array([r["n_points"] for r in recs], float)
x1, x2 = np.log(n * rad ** 2), gam                      # log tau^2, gamma
X = np.column_stack([x1 ** i * x2 ** j for i in range(4) for j in range(4) if i + j <= 3])
yb = np.log(beta * rad ** 2)
coef, *_ = np.linalg.lstsq(X, yb, rcond=None)
res = yb - X @ coef


def beta_R2(tau, g):
    t = np.log(tau ** 2)
    return float(np.exp(np.array([t ** i * g ** j for i in range(4) for j in range(4) if i + j <= 3]) @ coef))


grid = {}
for tau in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
    grid[str(tau)] = {str(g): round(np.pi * beta_R2(tau, g), 3) for g in (0.05, 0.2, 0.4, 0.6, 0.8, 0.95)}
covered = {"tau2": [float(np.quantile(n * rad ** 2, q)) for q in (0.01, 0.99)], "gamma": [0.05, 0.95]}
tau_ex, g_ex, nb_ex = 0.5, 0.3, 250.0
Rex = tau_ex / np.sqrt(nb_ex)
ps_bR2 = tau_ex ** 2 * np.exp(tau_ex ** 2 * (1 - g_ex) * np.pi)
report["strauss_dv1_lookup"] = {
    "inverse_fit_resid_sd_log_betaR2": float(res.std()),
    "beta_pi_R2_by_tau_gamma": grid, "dv1_support": covered,
    "worked": {"nbar": nb_ex, "tau": tau_ex, "gamma": g_ex, "R": Rex,
               "betaR2_PS": float(ps_bR2), "beta_PS": float(ps_bR2 / Rex ** 2),
               "betaR2_lookup": beta_R2(tau_ex, g_ex), "beta_lookup": beta_R2(tau_ex, g_ex) / Rex ** 2}}
print("strauss worked", report["strauss_dv1_lookup"]["worked"], "resid sd", res.std())
print("strauss beta*pi*R^2 grid", json.dumps(grid))

(OUT / "design_preview.json").write_text(json.dumps(report, indent=1, default=float))
np.savez(OUT / "design_preview.npz",
         **{f"draws_{f}_nbar": v[0] for f, v in fig_draws.items()},
         **{f"draws_{f}_delta": v[1] for f, v in fig_draws.items()},
         **{f"ladder_{f}": np.array([row[2] for row in rows]) for f, rows in fig_ladders.items()},
         **{f"ladder_{f}_amp": np.array([row[1] for row in rows]) for f, rows in fig_ladders.items()},
         ladder_delta=np.array(DELTA_FIG), r_grid=R)
print("wrote", OUT / "design_preview.json")


# ------------------------------------------------------------------ Strauss CFTP timing-probe grid (WP2b)
# beta from the DV1 lookup above (extrapolated at gamma = 0); timing only, so a rough beta suffices.
rows = ["cell,tau,gamma,nbar,R,beta"]
cell = 0
for nb in (400.0, 800.0):
    for tau in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        for g in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9):
            Rr = tau / np.sqrt(nb)
            rows.append(f"{cell},{tau},{g},{nb:.0f},{Rr:.8f},{beta_R2(tau, g) / Rr ** 2:.4f}")
            cell += 1
(OUT / "strauss_probe_grid.csv").write_text("\n".join(rows) + "\n")
print("wrote", OUT / "strauss_probe_grid.csv", cell, "cells")
