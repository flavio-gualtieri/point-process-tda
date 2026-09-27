#!/usr/bin/env python3
"""Classical baseline: minimum contrast on Ripley's K, per family, and a K-based model selection.

    python oneshot/mincontrast.py clouds                     # cloud set + K-hat + ring table
    python oneshot/mincontrast.py fit --task I               # one (model, chunk) unit of the fit array
    python oneshot/mincontrast.py assemble                   # tune on val, write units, summary
    bash oneshot/mincontrast_submit.sh                       # all three, then the end-to-end score

Estimator. For family f with free parameters x (log scale, intensity fixed at n on the unit window):

    x_hat = argmin_x  mean_{r <= r_max} ( K_hat(r)^c - K_f(r; x)^c )^2

K_hat is cloudforger.classical.lfunction.k_function (isotropic edge correction, the estimator the
classical features use); K_f = pi r^2 + Family.excess, the closed forms the bank's generator is built
on, so the baseline fits the exact model that simulated the data. The search box is the family's
train-prior range on the FREE parameters, starts are the best `starts` of `init` train-prior draws,
then bounded Nelder-Mead. Learned estimators get the same information: they train on those draws and
are clipped to their range. The dependent parameters (mu = n / kappa, ...) are NOT clipped: clipping
them independently breaks the fit's intensity = n (a nested mu2 clipped up to 1 once implied 2800
points for a cloud of a few hundred) and leaves a theta the sampler cannot realise. Cell has K = pi r^2, so K
carries nothing about k: its estimate is nbar = n and k = the train median of k (error ~ its s.d.).

Selection (the classifier). Every cloud is fitted under every family's model; the call is
    argmin_f  log D_f + lambda * p_f          (poisson: D of pi r^2, p = 0)
over poisson and the six fittable families; cell is never called, K cannot tell it from poisson.

Hyperparameters are tuned on the val split, as every learned model's early stopping and `best` are:
per family (c, r_max) by val error; for selection one shared (c, r_max) and lambda by val accuracy.
The textbook setting (config mincontrast.default: c = 1/4, r_max = quarter side) is reported too.

Cloud set: every val and test cloud of every family -- the unit contract of core.py, so the units
compare with the learned ones on the full test set and endtoend.py scores them unchanged.

Output
    data/oneshot/mincontrast/<run>/{clouds.csv, khat.npy, ring_table.npz, prior_<model>.npz}
    oneshot/results/<run>/mincontrast/fits/<model>/chunk_<i>.npz
    oneshot/results/<run>/{classify/mincontrast, classify/oracle, estimate/<f>/mincontrast}/  (unit contract)
    oneshot/results/<run>/mincontrast/{summary.md, report.json, tuning.csv, selection_tuning.csv}
"""

from __future__ import annotations

import argparse
import itertools
import math
import multiprocessing as mp
import os
import time

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize
from scipy.special import lambertw

from core import (BANK, DATA, TARGETS, config_arg, load_classifier, load_config, load_estimator,
                  rows as load_rows, run_dir, save_config, save_predictions, unit_dir, write_json)

from cloudforger.classical.lfunction import RADII, k_function         # noqa: E402
from cloudforger.simulation.families import FAMILIES, Rules, ring_step  # noqa: E402

OUT = DATA / "mincontrast"                  # / <run name>, set in main()
MODELS = ("thomas", "nested", "lgcp", "matern2", "ring", "matern1")    # families with a K to fit
N_FREE = {"thomas": 2, "nested": 4, "lgcp": 2, "matern2": 1, "ring": 3, "matern1": 1}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------------------- models

# Free parameters on the log scale, and how they (with n) become the family's parameters. Intensity
# is n on the unit window, the classical estimate; everything K cannot see is fixed by it.
def _matern2(core, n):
    R, x = core / math.sqrt(n), math.pi * core**2
    return {"R": R, "lam_p": n * -math.log1p(-x) / x}


def _matern1(core, n):
    R = core / math.sqrt(n)
    a = math.pi * R**2
    return {"R": R, "lam_p": float(-lambertw(-a * n, 0).real) / a}


FREE = {
    "thomas": (lambda r: [np.log(r.kappa), np.log(r.sigma)],
               lambda e, n: {"kappa": e[0], "sigma": e[1], "mu": n / e[0]}),
    "nested": (lambda r: [np.log(r.kappa), np.log(r.mu1), np.log(r.sigma1), np.log(r.sigma1 / r.sigma2)],
               lambda e, n: {"kappa": e[0], "mu1": e[1], "sigma1": e[2], "sigma2": e[2] / e[3],
                             "mu2": n / (e[0] * e[1])}),
    "lgcp": (lambda r: [np.log(r.sigma2), np.log(r.s)],
             lambda e, n: {"sigma2": e[0], "s": e[1], "nbar": n}),
    "matern2": (lambda r: [np.log(r.R * np.sqrt(r.nbar))], lambda e, n: _matern2(e[0], n)),
    "ring": (lambda r: [np.log(r.kappa), np.log(r.rho), np.log(r.sigma / r.rho)],
             lambda e, n: {"kappa": e[0], "rho": e[1], "sigma": e[2] * e[1], "mu": n / e[0]}),
    "matern1": (lambda r: [np.log(r.R * np.sqrt(r.nbar))], lambda e, n: _matern1(e[0], n)),
}


def ring_table(path=None) -> None:
    """ring_step is scale-free, G(r / rho, sigma / rho); tabulate it once (it costs ~50 ms a call)."""
    t = np.exp(np.linspace(np.log(0.03), np.log(1.5), 120))
    u = np.linspace(0, 2 + 8 * math.sqrt(2) * t[-1], 2400)
    G = np.stack([ring_step(u, 1.0, ti) for ti in t])
    np.savez(path or OUT / "ring_table.npz", log_t=np.log(t), u=u, G=G)


class Excess:
    """K(r) - pi r^2 of each fittable family; ring by interpolation in its table."""

    def __init__(self):
        rules = Rules.load()
        self.fam = {f: FAMILIES[f](rules) for f in MODELS}
        z = np.load(OUT / "ring_table.npz")
        self.ring = RegularGridInterpolator((z["log_t"], z["u"]), z["G"], bounds_error=False, fill_value=None)
        self.u_max = z["u"][-1]

    def __call__(self, f: str, r: np.ndarray, p: dict) -> np.ndarray:
        if f != "ring":
            return self.fam[f].excess(r, p)
        u = r / p["rho"]
        g = self.ring(np.column_stack([np.full(len(r), math.log(p["sigma"] / p["rho"])), np.minimum(u, self.u_max)]))
        return np.where(u >= self.u_max, 1.0, np.clip(g, 0.0, 1.0)) / p["kappa"]


# ------------------------------------------------------------------------------------------ fit

def settings(mc: dict) -> list[tuple[float, float]]:
    return list(itertools.product(mc["c"], mc["r_max"]))


def radii_mask(mc: dict, r_max: float) -> np.ndarray:
    m = np.zeros(len(RADII), bool)
    m[mc["radii_step"] - 1::mc["radii_step"]] = True
    return m & (RADII <= r_max + 1e-12)


_W = {}


def _init_worker(model, mc, box, prior_x):
    _W.update(model=model, mc=mc, box=box, prior=prior_x, excess=Excess(), setts=settings(mc),
              masks=[radii_mask(mc, rm) for _, rm in settings(mc)])


def fit_one(job):
    """All settings for one cloud under _W['model']: (theta_hat per setting, D, at_bound)."""
    khat, n = job
    f, mc, (lo, hi), X0, ex = _W["model"], _W["mc"], _W["box"], _W["prior"], _W["excess"]
    to_params = FREE[f][1]
    T = TARGETS[f]
    theta, D, bound = np.full((len(_W["setts"]), len(T)), np.nan), np.full(len(_W["setts"]), np.nan), np.zeros(len(_W["setts"]), bool)
    for s, ((c, _), m) in enumerate(zip(_W["setts"], _W["masks"])):
        r, target = RADII[m], np.maximum(khat[m], 0.0) ** c

        def contrast(x):
            try:
                k = np.pi * r**2 + ex(f, r, to_params(np.exp(x), n))
            except (ValueError, ZeroDivisionError, OverflowError):
                return 1e6
            v = float(np.mean((target - np.maximum(k, 0.0) ** c) ** 2))
            return v if np.isfinite(v) else 1e6

        order = np.argsort([contrast(x0) for x0 in X0])[:mc["starts"]]
        best = None
        for x0 in X0[order]:
            res = minimize(contrast, x0, method="Nelder-Mead", bounds=list(zip(lo, hi)),
                           options={"xatol": 1e-4, "fatol": 1e-14, "maxiter": 300 * len(lo)})
            if best is None or res.fun < best.fun:
                best = res
        p = to_params(np.exp(best.x), n)
        theta[s] = [p[t] for t in T]
        D[s] = best.fun
        bound[s] = bool(np.any(np.minimum(best.x - lo, hi - best.x) < 1e-3))
    return theta, D, bound


def cmd_fit(cfg: dict, task: int, workers: int) -> None:
    mc = cfg["mincontrast"]
    model, chunk = MODELS[task // mc["chunks"]], task % mc["chunks"]
    out = run_dir(cfg, "mincontrast", "fits", model) / f"chunk_{chunk}.npz"
    if out.exists():
        log(f"{out} exists")
        return
    clouds = pd.read_csv(OUT / "clouds.csv")
    khat = np.load(OUT / "khat.npy", mmap_mode="r")
    idx = np.arange(chunk, len(clouds), mc["chunks"])
    z = np.load(OUT / f"prior_{model}.npz")
    box, prior_x = (z["lo"], z["hi"]), z["starts"]
    log(f"{model} chunk {chunk}: {len(idx)} clouds x {len(settings(mc))} settings, {workers} workers")
    t0 = time.time()
    jobs = [(np.asarray(khat[i]), float(clouds.n[i])) for i in idx]
    with mp.get_context("fork").Pool(workers, _init_worker, (model, mc, box, prior_x)) as pool:
        res = pool.map(fit_one, jobs, chunksize=4)
    save_predictions(out, case_id=clouds.case_id.to_numpy()[idx], theta=np.stack([r[0] for r in res]),
                     D=np.stack([r[1] for r in res]), at_bound=np.stack([r[2] for r in res]),
                     targets=np.array(TARGETS[model]), settings=np.array(settings(mc)))
    log(f"-> {out} ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------------------------------- clouds

def _khat(args):
    pts, = args
    return k_function(pts).astype(np.float32)


def cmd_clouds(cfg: dict, workers: int) -> None:
    mc = cfg["mincontrast"]
    OUT.mkdir(parents=True, exist_ok=True)
    r = load_rows(cfg)
    clouds = r.loc[r.split.isin(["val", "test"]), ["family", "split", "theta", "rep", "n"]].reset_index()
    log(f"{len(clouds)} clouds: {clouds.groupby(['family', 'split']).size().unstack().to_dict()}")
    khat = np.zeros((len(clouds), len(RADII)), np.float32)
    for f, g in clouds.groupby("family"):
        man = pd.read_csv(BANK / f / "manifest.csv", usecols=["case_id"])
        pos = pd.Index(man.case_id).get_indexer(g.case_id)
        z = np.load(BANK / f / "points.npz")
        pts, off = z["points"], z["offsets"]
        with mp.get_context("fork").Pool(workers) as pool:
            khat[g.index] = np.stack(pool.map(_khat, [(pts[off[i]:off[i + 1]],) for i in pos], chunksize=16))
        log(f"  K-hat {f}: {len(g)}")
    np.save(OUT / "khat.npy", khat)
    clouds.to_csv(OUT / "clouds.csv", index=False)
    for f in MODELS:            # search box and starts: the family's train prior, on the free log scale
        X = np.column_stack(FREE[f][0](r[(r.split == "train") & (r.family == f)]))
        starts = X[np.random.default_rng(mc["seed"]).choice(len(X), mc["init"], replace=False)]
        np.savez(OUT / f"prior_{f}.npz", lo=X.min(0), hi=X.max(0), starts=starts)
    ring_table()
    log(f"-> {OUT}")


# -------------------------------------------------------------------------------------- assemble

def err(pred_log, true_log, sd):
    """compare.py's normalisation: RMSE of log(target) / s.d. over the family's test clouds, mean over targets."""
    return float(np.mean(np.sqrt(((pred_log - true_log) ** 2).mean(0)) / sd))


def boot_diff(a, b, true, sd, thetas, B=200, seed=0):
    """err(a) - err(b), 95% interval over test thetas."""
    rng = np.random.default_rng(seed)
    ut = np.unique(thetas)
    groups = {t: np.flatnonzero(thetas == t) for t in ut}
    d = []
    for _ in range(B):
        i = np.concatenate([groups[t] for t in rng.choice(ut, len(ut))])
        d.append(err(a[i], true[i], sd) - err(b[i], true[i], sd))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def cmd_assemble(cfg: dict) -> None:
    from compare import in_regime
    import json
    mc = cfg["mincontrast"]
    out = run_dir(cfg, "mincontrast")
    fams = cfg["families"]
    setts = settings(mc)
    dflt = setts.index((mc["default"]["c"], mc["default"]["r_max"]))
    clouds = pd.read_csv(OUT / "clouds.csv").set_index("case_id")
    r = load_rows(cfg)
    cut_path = run_dir(cfg, "compare") / "report.json"
    cuts = json.loads(cut_path.read_text())["cutoffs"] if cut_path.exists() else None
    info = r.loc[clouds.index]

    # fits[model] -> theta (N, S, T), D (N, S), bound (N, S), aligned to clouds
    fits = {}
    for f in MODELS:
        parts = [np.load(p) for p in sorted((out / "fits" / f).glob("chunk_*.npz"))]
        if len(parts) != mc["chunks"]:
            raise SystemExit(f"{f}: {len(parts)} of {mc['chunks']} chunks")
        cid = np.concatenate([z["case_id"] for z in parts])
        at = pd.Index(cid).get_indexer(clouds.index)
        fits[f] = {k: np.concatenate([z[k] for z in parts])[at] for k in ("theta", "D", "at_bound")}

    split, family, n = clouds.split.to_numpy(), clouds.family.to_numpy(), clouds.n.to_numpy(float)
    val, test = split == "val", split == "test"
    k_med = float(np.exp(np.median(np.log(r.k[(r.split == "train") & (r.family == "cell")]))))
    sd_all = {f: np.log(r[(r.split == "test") & (r.family == f)][TARGETS[f]].to_numpy(float)).std(0) for f in fams}
    vsd = {f: np.log(r[(r.split == "val") & (r.family == f)][TARGETS[f]].to_numpy(float)).std(0) for f in fams}
    Y = {f: np.log(info[TARGETS[f]].to_numpy(float)) for f in fams if f != "poisson"}
    def est(f, s):
        return np.log(fits[f]["theta"][:, s])           # free parameters box-bounded; see the module doc

    # ---- estimator tuning: per family, the setting with the lowest val error on its own clouds
    tune, chosen = [], {}
    for f in MODELS:
        own = family == f
        for s, (c, rm) in enumerate(setts):
            lg = est(f, s)
            tune.append({"family": f, "c": c, "r_max": rm, "val": err(lg[own & val], Y[f][own & val], vsd[f]),
                         "test": err(lg[own & test], Y[f][own & test], sd_all[f]),
                         "at_bound_test": float(fits[f]["at_bound"][own & test, s].mean())})
        rows_f = [t for t in tune if t["family"] == f]
        chosen[f] = int(np.argmin([t["val"] for t in rows_f]))
    tune = pd.DataFrame(tune)
    tune.to_csv(out / "tuning.csv", index=False)

    # ---- selection: log D_f + lambda p_f, poisson's D = the contrast of pi r^2 under the same setting
    khat = np.load(OUT / "khat.npy", mmap_mode="r")

    def d_poisson(s):
        c, rm = setts[s]
        m = radii_mask(mc, rm)
        return np.mean((np.maximum(np.asarray(khat[:, m], float), 0) ** c - (np.pi * RADII[m] ** 2) ** c) ** 2, axis=1)

    fam_w = {f: 1 / len(fams) for f in fams}

    def balanced_acc(call, mask):
        return float(sum(fam_w[f] * np.mean(call[mask & (family == f)] == f) for f in fams))

    def select(s, lam):
        crit = np.column_stack([np.log(d_poisson(s))] + [np.log(fits[f]["D"][:, s]) + lam * N_FREE[f] for f in MODELS])
        return np.array(["poisson", *MODELS])[np.argmin(crit, axis=1)]

    sel = [{"c": setts[s][0], "r_max": setts[s][1], "lambda": lam, "val": balanced_acc(select(s, lam), val),
            "test": balanced_acc(select(s, lam), test)} for s in range(len(setts)) for lam in mc["lambdas"]]
    sel = pd.DataFrame(sel)
    sel.to_csv(out / "selection_tuning.csv", index=False)
    b = sel.val.idxmax()
    s_sel, lam = setts.index((sel.c[b], sel.r_max[b])), float(sel["lambda"][b])
    call = select(s_sel, lam)
    call_default = select(dflt, mc["default"]["lambda"])

    # ---- units, in the oneshot contract, so endtoend.py scores them like any trained model
    onehot = lambda labels: (np.array(labels)[:, None] == np.array(fams)[None, :]).astype(np.float32)
    for name, labels in (("mincontrast", call), ("oracle", family)):
        d = unit_dir(cfg, "classify", name)
        save_predictions(d / "predictions.npz", case_id=clouds.index.to_numpy(), split=split,
                         posterior=onehot(labels), classes=np.array(fams))
        write_json(d / "report.json", {"source": "oneshot/mincontrast.py",
                                       "note": "K-contrast selection" if name == "mincontrast" else "the true family"})
    for f in [*MODELS, "cell"]:
        theta = (np.column_stack([n, np.full(len(n), k_med)]) if f == "cell" else np.exp(est(f, chosen[f])))
        d = unit_dir(cfg, "estimate", "mincontrast", f)
        save_predictions(d / "predictions.npz", case_id=clouds.index.to_numpy(), split=split, theta_hat=theta,
                         targets=np.array(TARGETS[f]), family=f)
        write_json(d / "report.json", {"source": "oneshot/mincontrast.py",
                                       "setting": None if f == "cell" else dict(zip(("c", "r_max"), setts[chosen[f]]))})

    # ---- comparison with the learned units on the same test clouds
    learned = [m for m in cfg["estimate"] if m in ("hgb_classical", "hgb_classical_ph", "nn_curves")]
    report = {"settings": setts, "chosen": {f: setts[s] for f, s in chosen.items()},
              "selection": {"c": setts[s_sel][0], "r_max": setts[s_sel][1], "lambda": lam}, "estimators": {}, "classifiers": {}}
    L = [f"# Minimum-contrast baseline: {cfg['name']}", "",
         f"Test clouds: {int(test.sum())}, every test cloud of every family, as in compare/summary.md; learned and "
         "classical columns are paired. Error = RMSE(log θ)/s.d., "
         "mean over targets (compare.py's normalisation); intervals are 95% over test θ.", "",
         "## Estimators (true family known)", "",
         f"`mc default` = c {setts[dflt][0]}, r_max {setts[dflt][1]}; `mc tuned` = the setting with the lowest val error.", "",
         "| family | mc default | mc tuned (c, r_max) | at bound | " + " | ".join(learned) + " | tuned − hgb_classical_ph |",
         "|---|---|---|---|" + "---|" * len(learned) + "---|"]
    for f in [*MODELS, "cell"]:
        own = test & (family == f)
        tid = clouds.index[own]
        true = Y[f][own]
        mc_t = (np.log(np.column_stack([n[own], np.full(own.sum(), k_med)])) if f == "cell" else est(f, chosen[f])[own])
        mc_d = mc_t if f == "cell" else est(f, dflt)[own]
        cells, ref = {}, None
        for m in learned:
            p = load_estimator(cfg, f, m)
            if p is None:
                continue
            lg = np.log(p.reindex(tid)[TARGETS[f]].to_numpy())
            cells[m] = err(lg, true, sd_all[f])
            if m == "hgb_classical_ph":
                ref = lg
        lo, hi = boot_diff(mc_t, ref, true, sd_all[f], info.theta.to_numpy()[own]) if ref is not None else (np.nan, np.nan)
        eb = "—" if f == "cell" else f"{fits[f]['at_bound'][own, chosen[f]].mean():.2f}"
        st = "n, median k" if f == "cell" else f"{setts[chosen[f]][0]}, {setts[chosen[f]][1]}"
        e_t, e_d = err(mc_t, true, sd_all[f]), err(mc_d, true, sd_all[f])
        L.append(f"| {f} | {e_d:.3f} | {e_t:.3f} ({st}) | {eb} | " + " | ".join(f"{cells.get(m, np.nan):.3f}" for m in learned)
                 + f" | {e_t - cells.get('hgb_classical_ph', np.nan):+.3f} [{lo:+.3f}, {hi:+.3f}] |")
        report["estimators"][f] = {"default": e_d, "tuned": e_t, **cells, "diff_ci": [lo, hi]}

    L += ["", "By regime (tuned mc vs hgb_classical_ph; in regime = compare's frozen cutoffs, τ = 0.5 / 0.9):", "",
          "| family | subset | n | mc tuned | mc at bound | hgb_classical_ph |", "|---|---|---|---|---|---|"]
    for f in MODELS:
        own = test & (family == f)
        sub = info[own]
        p = load_estimator(cfg, f, "hgb_classical_ph")
        if p is None or cuts is None:
            continue
        ref = np.log(p.reindex(sub.index)[TARGETS[f]].to_numpy())
        for name, m in [("all", np.ones(own.sum(), bool)), ("out of regime 0.5", ~in_regime(cuts, sub, 0.5)),
                        ("in regime 0.5", in_regime(cuts, sub, 0.5)), ("in regime 0.9", in_regime(cuts, sub, 0.9))]:
            if m.sum() >= 20:
                L.append(f"| {f} | {name} | {m.sum()} | {err(est(f, chosen[f])[own][m], Y[f][own][m], sd_all[f]):.3f} | "
                         f"{fits[f]['at_bound'][own, chosen[f]][m].mean():.2f} | {err(ref[m], Y[f][own][m], sd_all[f]):.3f} |")

    L += ["", "## Selection (the classical pipeline's classifier)", "",
          f"Tuned on val: c {setts[s_sel][0]}, r_max {setts[s_sel][1]}, λ {lam}; default: c {setts[dflt][0]}, "
          f"r_max {setts[dflt][1]}, λ {mc['default']['lambda']}. Balanced accuracy (equal prior), same test clouds.", "",
          "| classifier | accuracy | " + " | ".join(fams) + " |", "|---|---|" + "---|" * len(fams)]
    cands = {"mincontrast (tuned)": call, "mincontrast (default)": call_default}
    for m in ("hgb_classical", "hgb_classical_ph"):
        post = load_classifier(cfg, m)
        if post is not None:
            pr = post.reindex(clouds.index[test])[fams].to_numpy()
            full = np.full(len(clouds), "", object)
            full[test] = np.array(fams)[pr.argmax(1)]
            cands[m] = full
    for name, cl in cands.items():
        acc = balanced_acc(cl, test)
        rec = [np.mean(cl[test & (family == f)] == f) for f in fams]
        L.append(f"| {name} | {acc:.3f} | " + " | ".join(f"{x:.2f}" for x in rec) + " |")
        report["classifiers"][name] = {"accuracy": acc, "recall": dict(zip(fams, rec))}
    L += ["", "Cell is never called: its K is exactly πr², so any K-based selection sends it to poisson.", ""]
    write_json(out / "report.json", report)
    (out / "summary.md").write_text("\n".join(L) + "\n")
    log(f"-> {out / 'summary.md'}")


# ----------------------------------------------------------------------------------------- main

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    config_arg(p)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("clouds")
    f = sub.add_parser("fit")
    f.add_argument("--task", type=int, required=True, help="model * chunks + chunk")
    sub.add_parser("assemble")
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    global OUT
    OUT = OUT / cfg["name"]
    if args.cmd == "clouds":
        cmd_clouds(cfg, args.workers)
    elif args.cmd == "fit":
        cmd_fit(cfg, args.task, args.workers)
    else:
        save_config(args.config, run_dir(cfg, "mincontrast"))
        cmd_assemble(cfg)


if __name__ == "__main__":
    main()
