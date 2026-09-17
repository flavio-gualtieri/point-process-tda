"""Read-only inspection of the EXISTING generated data (2026-09-02 regen), for the
generation audit in docs/theory/generation.tex. Simulates nothing, trains nothing.

Reads data/<proc>/{,adversarial_}clouds.pkl (parameters + point counts only),
docs/theory/scripts/out/data_departure.npz (S_L from departure.py), and SLURM
accounting + logs of the two diagram jobs, then prints a report and writes
docs/theory/scripts/out/inspect_generation.json.

    python docs/theory/scripts/inspect_generation.py      # cloud-env python, seconds
"""

from __future__ import annotations

import json
import pickle
import re
import subprocess
from pathlib import Path

import numpy as np
from scipy.special import lambertw
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "out"
DIAG_JOBS = {25052116: "diagrams_compute", 25082983: "diagrams_compute_sa"}
report: dict = {}


def load(proc: str) -> tuple[list[dict], list[dict]]:
    tt = pickle.load(open(ROOT / f"data/{proc}/clouds.pkl", "rb"))
    adv = pickle.load(open(ROOT / f"data/{proc}/adversarial_clouds.pkl", "rb"))
    return tt, adv


def arr(recs, key):
    return np.array([r["params"][key] for r in recs], dtype=float)


def q(x, ps=(0, 1, 5, 25, 50, 75, 95, 99, 100)):
    return {f"q{p}": float(np.percentile(x, p)) for p in ps}


def rho(a, b):
    return float(spearmanr(a, b).correlation)


def ratio_stats(n, nbar):
    r = n / nbar
    return {"mean": float(r.mean()), "se": float(r.std(ddof=1) / np.sqrt(len(r)))}


def binned_ratio(n, nbar, x, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() > 10:
            out.append({"bin": [float(lo), float(hi)], "N": int(m.sum()), **ratio_stats(n[m], nbar[m])})
    return out


# ---------------------------------------------------------------- per process
recs = {p: load(p) for p in ["thomas", "matern_cluster", "nested_thomas", "strauss", "lgcp", "lgcp_strauss"]}
n_all = {p: np.array([r["n_points"] for r in tt + adv], float) for p, (tt, adv) in recs.items()}
R_all = {p: tt + adv for p, (tt, adv) in recs.items()}

# Thomas / Matern cluster: nbar = kappa * mu (|W| = 1)
for p, scale_key in [("thomas", "cluster_scale"), ("matern_cluster", "cluster_radius")]:
    R, n = R_all[p], n_all[p]
    kap, mu, sc = arr(R, "parent_intensity"), arr(R, "mean_offspring"), arr(R, scale_key)
    nbar = kap * mu
    sig = sc if p == "thomas" else sc / 2.0          # twin-matched RMS per-axis displacement
    s_dimless = sig * np.sqrt(nbar)                  # sigma / ell
    c = 2 * sig * np.sqrt(kap)                       # overlap index
    report[p] = {
        "n": q(n), "nbar": q(nbar), "n_over_nbar": ratio_stats(n, nbar),
        "cv_n_given_nbar": float(np.std(n / nbar)),
        "mu": q(mu), "s=sigma*sqrt(lambda)": q(s_dimless), "c": q(c),
        "g0_minus_1=1/(pi c^2)": q(1 / (np.pi * c ** 2)),
        "spearman(nbar, mu)": rho(nbar, mu), "spearman(nbar, s)": rho(nbar, s_dimless),
        "spearman(nbar, c)": rho(nbar, c), "spearman(n, mu)": rho(n, mu),
    }

# Nested Thomas: nbar = kappa * mu1 * mu2
R, n = R_all["nested_thomas"], n_all["nested_thomas"]
kap, mu1, mu2 = arr(R, "parent_intensity"), arr(R, "meta_offspring"), arr(R, "mean_offspring")
s1, s2 = arr(R, "meta_cluster_scale"), arr(R, "cluster_scale")
nbar = kap * mu1 * mu2
report["nested_thomas"] = {
    "n": q(n), "nbar": q(nbar), "n_over_nbar": ratio_stats(n, nbar),
    "kappa(meta-clusters in W)": q(kap), "mu1": q(mu1), "mu2": q(mu2),
    "sigma1/sigma2": q(s1 / s2),
    "inner_mass=mu2": q(mu2), "outer_mass=mu1*mu2": q(mu1 * mu2),
    "outer_scale 2*sqrt(s1^2+s2^2)": q(2 * np.sqrt(s1 ** 2 + s2 ** 2)),
    "inner_scale 2*s2": q(2 * s2),
    "s2*sqrt(lambda)": q(s2 * np.sqrt(nbar)),
    "spearman(nbar, mu1*mu2)": rho(nbar, mu1 * mu2),
}

# Strauss: lambda unknown in closed form; scale invariance says lambda R^2 = h(beta R^2, gamma)
R, n = R_all["strauss"], n_all["strauss"]
beta, gam, rad = arr(R, "beta"), arr(R, "gamma"), arr(R, "radius")
G = (1 - gam) * np.pi * rad ** 2
lam_ps = np.real(lambertw(beta * G)) / G
tau = rad * np.sqrt(n)
x1, x2 = np.log(beta * rad ** 2), gam
y = np.log(n * rad ** 2)                             # log(lambda_hat R^2)
# cubic polynomial in (log beta R^2, gamma): a 2-D lookup, the scale-invariant inversion
Xd = np.column_stack([x1 ** i * x2 ** j for i in range(4) for j in range(4) if i + j <= 3])
coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
resid = y - Xd @ coef
# 3-D fit adds log R as a free covariate: if it barely helps, scale invariance holds (no edge effect)
Xd3 = np.column_stack([Xd, np.log(rad), np.log(rad) ** 2, np.log(rad) * x1])
coef3, *_ = np.linalg.lstsq(Xd3, y, rcond=None)
resid3 = y - Xd3 @ coef3
report["strauss"] = {
    "n": q(n), "beta": q(beta), "lambda_hat/beta": q(n / beta),
    "log(n/lambda_PS)": q(np.log(n / lam_ps)), "mean log(n/lambda_PS)": float(np.log(n / lam_ps).mean()),
    "tau=R*sqrt(n)": q(tau), "frac tau<0.1": float((tau < 0.1).mean()), "frac tau<0.2": float((tau < 0.2).mean()),
    "packing hard-core eta=pi tau^2/4": q(np.pi * tau ** 2 / 4),
    "beta*pi*R^2 (dominating pts per interaction disc)": q(beta * np.pi * rad ** 2),
    "2D lookup: resid sd of log n": float(resid.std()),
    "3D (+log R) fit: resid sd of log n": float(resid3.std()),
    "poisson-scale sd of log n, median 1/sqrt(n)": float(np.median(1 / np.sqrt(n))),
    "spearman(n, gamma)": rho(n, gam), "spearman(n, R)": rho(n, rad), "spearman(n, beta)": rho(n, beta),
}

# LGCP: nbar = exp(mu + sigma2/2); check realized mean ratio against it, by sigma2 (thinning cap)
R, n = R_all["lgcp"], n_all["lgcp"]
mu, s2v, sL = arr(R, "mu"), arr(R, "sigma2"), arr(R, "s")
nbar = np.exp(mu + s2v / 2)
report["lgcp"] = {
    "n": q(n), "nbar": q(nbar), "n_over_nbar": ratio_stats(n, nbar),
    "n_over_nbar by sigma2": binned_ratio(n, nbar, s2v, [0, 1, 2, 3, 4.01]),
    "s*sqrt(nbar)": q(sL * np.sqrt(nbar)), "frac s*sqrt(nbar)<0.5": float((sL * np.sqrt(nbar) < 0.5).mean()),
    "spearman(nbar, sigma2)": rho(nbar, s2v),
}
R, n = R_all["lgcp_strauss"], n_all["lgcp_strauss"]
report["lgcp_strauss"] = {"n": q(n), "activity exp(mu+s2/2)": q(np.exp(arr(R, "mu") + arr(R, "sigma2") / 2)),
                          "n/activity": q(n / np.exp(arr(R, "mu") + arr(R, "sigma2") / 2))}

# Cross-family coupling: same seed -> same design draw
cpl = {}
for a, b, keys in [("lgcp", "lgcp_strauss", ["sigma2", "s"]), ("thomas", "matern_cluster", ["parent_intensity", "mean_offspring"])]:
    same = np.all([np.isclose(arr(R_all[a], k), arr(R_all[b], k)) for k in keys], axis=0)
    cpl[f"{a}~{b} identical {keys}"] = float(same.mean())
report["cross_family_coupling"] = cpl

# ---------------------------------------------------- confound with departure S_L
dep = np.load(OUT / "data_departure.npz")
report["departure"] = {}
for p in ["thomas", "matern_cluster", "nested_thomas", "strauss"]:
    nn, SL = dep[f"{p}__test__n"], dep[f"{p}__test__L_S_max"]
    report["departure"][p] = {"spearman(n, S_L)": rho(nn, SL), "S_L": q(SL, (1, 5, 50, 95))}

# ------------------------------------------------ diagram cost vs n (SLURM data)
def sacct_elapsed(job: int) -> dict[int, float]:
    out = subprocess.run(["sacct", "-j", str(job), "-X", "-P", "-n", "--format=JobID,Elapsed,State"],
                         capture_output=True, text=True).stdout
    el = {}
    for line in out.splitlines():
        jid, e, st = line.split("|")
        if st != "COMPLETED" or "_" not in jid:
            continue
        h, m, s = (e.split("-")[-1]).split(":")
        el[int(jid.split("_")[1])] = int(h) * 3600 + int(m) * 60 + float(s)
    return el


rows = []
for job, stem in DIAG_JOBS.items():
    el = sacct_elapsed(job)
    for task, sec in el.items():
        f = ROOT / f"logs/{stem}_{job}_{task}.out"
        if not f.exists():
            continue
        txt = f.read_text()
        m = re.search(r"process=(\w+)\s+filtration=(\w+)", txt)
        tt = re.search(r"train/test: clouds \[(\d+):(\d+)\)", txt)
        ad = re.search(r"adversarial: clouds \[(\d+):(\d+)\)", txt)
        if not (m and tt):
            continue
        proc, filt = m.group(1), m.group(2)
        if proc not in recs:
            continue
        ntt = np.array([r["n_points"] for r in recs[proc][0][int(tt.group(1)):int(tt.group(2))]], float)
        nad = np.array([r["n_points"] for r in recs[proc][1][int(ad.group(1)):int(ad.group(2))]], float) if ad else np.empty(0)
        rows.append((filt, sec, np.concatenate([ntt, nad])))

fits = {}
for filt in ["dtm_k5", "dtm_k10", "dtm_k15", "rips"]:
    sub = [(s, ns) for f, s, ns in rows if f == filt]
    if len(sub) < 20:
        continue
    t = np.array([s for s, _ in sub])
    best = None
    for a in np.arange(1.0, 4.01, 0.05):
        S = np.array([((ns / 400.0) ** a).sum() for _, ns in sub])  # per-task sum of (n/400)^alpha
        c = (S @ t) / (S @ S)
        sse = ((t - c * S) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, a, c, 1 - sse / ((t - t.mean()) ** 2).sum())
    fits[filt] = {"tasks": len(sub), "alpha": float(best[1]), "sec_per_diagram_at_n400": float(best[2]),
                  "R2": float(best[3]), "total_cpu_h": float(t.sum() / 3600)}
report["diagram_cost_model t = c*(n/400)^alpha"] = fits
# per-task points for the compute figure: seconds per diagram vs the task's power-mean n
np.savez(OUT / "diagram_tasks.npz",
         filt=np.array([f for f, _, _ in rows]),
         sec_per_diagram=np.array([s / len(ns) for _, s, ns in rows]),
         n_eff=np.array([np.mean(ns ** fits.get(f, {"alpha": 3.1})["alpha"]) ** (1 / fits.get(f, {"alpha": 3.1})["alpha"])
                         for f, _, ns in rows]))

OUT.mkdir(exist_ok=True)
(OUT / "inspect_generation.json").write_text(json.dumps(report, indent=1))


def show(d, ind=0):
    for k, v in d.items():
        if isinstance(v, dict):
            print(" " * ind + f"{k}:")
            show(v, ind + 2)
        elif isinstance(v, list):
            print(" " * ind + f"{k}:")
            for e in v:
                print(" " * (ind + 2) + json.dumps(e))
        else:
            print(" " * ind + f"{k}: {v:.4g}" if isinstance(v, float) else " " * ind + f"{k}: {v}")


show(report)
