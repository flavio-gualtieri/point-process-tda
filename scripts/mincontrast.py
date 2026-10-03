#!/usr/bin/env python3
"""Classical baseline: minimum contrast on Ripley's K per family, and a K-based model selection
(the estimator itself is cloudforger.baselines.mincontrast).

    python scripts/mincontrast.py clouds                     # cloud set, K-hat, prior boxes, ring table
    python scripts/mincontrast.py fit --task I               # one (model, chunk) unit of the fit array
    python scripts/mincontrast.py tasks                      # how many fit units there are
    python scripts/mincontrast.py assemble                   # tune on val, write units, summary

Hyperparameters are tuned on the val split, as every learned model's early stopping and `best` are:
per family (c, r_max) by val error; for selection one shared (c, r_max) and lambda by val accuracy.
The textbook setting (config mincontrast.default: c = 1/4, r_max = quarter side) is reported too.
Learned estimators get the same information: they train on the train-prior draws and are clipped to
their range. Cell's estimate is nbar = n and k = the train median of k (K carries nothing about k).

Cloud set: every val and test cloud of every family -- the unit contract of pipeline.core, so the
units compare with the learned ones on the full test set and endtoend.py scores them unchanged.

Output
    <data>/mincontrast/<run>/{clouds.csv, khat.npy, ring_table.npz, prior_<model>.npz}
    <results>/<run>/mincontrast/fits/<model>/chunk_<i>.npz
    <results>/<run>/{classify/mincontrast, classify/oracle, estimate/<f>/mincontrast}/  (unit contract)
    <results>/<run>/mincontrast/{summary.md, report.json, tuning.csv, selection_tuning.csv}
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd

from cloudforger.baselines.mincontrast import (FALLBACK, FREE, N_FREE, fit_one, init_worker, models, radii_mask,
                                               ring_table, settings)
from cloudforger.classical.lfunction import RADII, k_function
from cloudforger.paths import BANK, MINCONTRAST
from cloudforger.pipeline.core import (TARGETS, config_arg, load_classifier, load_config, load_estimator, log,
                                       rows as load_rows, run_dir, save_config, save_predictions, unit_dir, write_json)
from cloudforger.pipeline.regime import in_regime

OUT = MINCONTRAST                   # / <run name>, set in main()


def cmd_fit(cfg: dict, task: int, workers: int) -> None:
    mc = cfg["mincontrast"]
    model, chunk = models(cfg["families"])[task // mc["chunks"]], task % mc["chunks"]
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
    with mp.get_context("fork").Pool(workers, init_worker, (model, mc, box, prior_x, OUT / "ring_table.npz")) as pool:
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
    for f in models(cfg["families"]):   # search box and starts: the family's train prior, on the free log scale
        X = np.column_stack(FREE[f][0](r[(r.split == "train") & (r.family == f)]))
        starts = X[np.random.default_rng(mc["seed"]).choice(len(X), mc["init"], replace=False)]
        np.savez(OUT / f"prior_{f}.npz", lo=X.min(0), hi=X.max(0), starts=starts)
    ring_table(OUT / "ring_table.npz")
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
    mc = cfg["mincontrast"]
    out = run_dir(cfg, "mincontrast")
    fams = cfg["families"]
    fitted, fallback = models(fams), [f for f in FALLBACK if f in fams]
    setts = settings(mc)
    dflt = setts.index((mc["default"]["c"], mc["default"]["r_max"]))
    clouds = pd.read_csv(OUT / "clouds.csv").set_index("case_id")
    r = load_rows(cfg)
    cut_path = run_dir(cfg, "compare") / "report.json"
    cuts = json.loads(cut_path.read_text())["cutoffs"] if cut_path.exists() else None
    info = r.loc[clouds.index]

    # fits[model] -> theta (N, S, T), D (N, S), bound (N, S), aligned to clouds
    fits = {}
    for f in fitted:
        parts = [np.load(p) for p in sorted((out / "fits" / f).glob("chunk_*.npz"))]
        if len(parts) != mc["chunks"]:
            raise SystemExit(f"{f}: {len(parts)} of {mc['chunks']} chunks")
        cid = np.concatenate([z["case_id"] for z in parts])
        at = pd.Index(cid).get_indexer(clouds.index)
        fits[f] = {k: np.concatenate([z[k] for z in parts])[at] for k in ("theta", "D", "at_bound")}

    split, family, n = clouds.split.to_numpy(), clouds.family.to_numpy(), clouds.n.to_numpy(float)
    val, test = split == "val", split == "test"
    prior = {f: FALLBACK[f](r[(r.split == "train") & (r.family == f)], n) for f in fallback}   # aligned to clouds
    sd_all = {f: np.log(r[(r.split == "test") & (r.family == f)][TARGETS[f]].to_numpy(float)).std(0) for f in fams}
    vsd = {f: np.log(r[(r.split == "val") & (r.family == f)][TARGETS[f]].to_numpy(float)).std(0) for f in fams}
    Y = {f: np.log(info[TARGETS[f]].to_numpy(float)) for f in fams if f != "poisson"}
    def est(f, s):
        return np.log(fits[f]["theta"][:, s])           # free parameters box-bounded; see the module doc

    # ---- estimator tuning: per family, the setting with the lowest val error on its own clouds
    tune, chosen = [], {}
    for f in fitted:
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
        crit = np.column_stack([np.log(d_poisson(s))] + [np.log(fits[f]["D"][:, s]) + lam * N_FREE[f] for f in fitted])
        return np.array(["poisson", *fitted])[np.argmin(crit, axis=1)]

    sel = [{"c": setts[s][0], "r_max": setts[s][1], "lambda": lam, "val": balanced_acc(select(s, lam), val),
            "test": balanced_acc(select(s, lam), test)} for s in range(len(setts)) for lam in mc["lambdas"]]
    sel = pd.DataFrame(sel)
    sel.to_csv(out / "selection_tuning.csv", index=False)
    b = sel.val.idxmax()
    s_sel, lam = setts.index((sel.c[b], sel.r_max[b])), float(sel["lambda"][b])
    call = select(s_sel, lam)
    call_default = select(dflt, mc["default"]["lambda"])

    # ---- units, in the pipeline's contract, so endtoend.py scores them like any trained model
    onehot = lambda labels: (np.array(labels)[:, None] == np.array(fams)[None, :]).astype(np.float32)
    for name, labels in (("mincontrast", call), ("oracle", family)):
        d = unit_dir(cfg, "classify", name)
        save_predictions(d / "predictions.npz", case_id=clouds.index.to_numpy(), split=split,
                         posterior=onehot(labels), classes=np.array(fams))
        write_json(d / "report.json", {"source": "scripts/mincontrast.py",
                                       "note": "K-contrast selection" if name == "mincontrast" else "the true family"})
    for f in [*fitted, *fallback]:
        theta = prior[f] if f in FALLBACK else np.exp(est(f, chosen[f]))
        d = unit_dir(cfg, "estimate", "mincontrast", f)
        save_predictions(d / "predictions.npz", case_id=clouds.index.to_numpy(), split=split, theta_hat=theta,
                         targets=np.array(TARGETS[f]), family=f)
        write_json(d / "report.json", {"source": "scripts/mincontrast.py",
                                       "setting": None if f in FALLBACK else dict(zip(("c", "r_max"), setts[chosen[f]]))})

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
    for f in [*fitted, *fallback]:
        own = test & (family == f)
        tid = clouds.index[own]
        true = Y[f][own]
        mc_t = np.log(prior[f][own]) if f in FALLBACK else est(f, chosen[f])[own]
        mc_d = mc_t if f in FALLBACK else est(f, dflt)[own]
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
        eb = "—" if f in FALLBACK else f"{fits[f]['at_bound'][own, chosen[f]].mean():.2f}"
        st = "n, prior median" if f in FALLBACK else f"{setts[chosen[f]][0]}, {setts[chosen[f]][1]}"
        e_t, e_d = err(mc_t, true, sd_all[f]), err(mc_d, true, sd_all[f])
        L.append(f"| {f} | {e_d:.3f} | {e_t:.3f} ({st}) | {eb} | " + " | ".join(f"{cells.get(m, np.nan):.3f}" for m in learned)
                 + f" | {e_t - cells.get('hgb_classical_ph', np.nan):+.3f} [{lo:+.3f}, {hi:+.3f}] |")
        report["estimators"][f] = {"default": e_d, "tuned": e_t, **cells, "diff_ci": [lo, hi]}

    L += ["", "By regime (tuned mc vs hgb_classical_ph; in regime = compare's frozen cutoffs, τ = 0.5 / 0.9):", "",
          "| family | subset | n | mc tuned | mc at bound | hgb_classical_ph |", "|---|---|---|---|---|---|"]
    for f in fitted:
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
    L += ["", f"Never called: {', '.join(fallback)} -- no K to fit (cell's is exactly πr², so K-based selection "
          "sends it to poisson); estimated as n and the train-prior median.", ""]
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
    sub.add_parser("tasks")
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    global OUT
    OUT = OUT / cfg["name"]
    if args.cmd == "tasks":
        print(len(models(cfg["families"])) * cfg["mincontrast"]["chunks"])
    elif args.cmd == "clouds":
        cmd_clouds(cfg, args.workers)
    elif args.cmd == "fit":
        cmd_fit(cfg, args.task, args.workers)
    else:
        save_config(cfg, run_dir(cfg, "mincontrast"))
        cmd_assemble(cfg)


if __name__ == "__main__":
    main()
