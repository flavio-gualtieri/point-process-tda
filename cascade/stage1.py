#!/usr/bin/env python3
"""Stage 1 of the cascade: is a cloud poisson, clustered or repulsive?

    python cascade/stage1.py train [--config cascade/configs/default.yaml]
    python cascade/stage1.py benchmark        # existing 5-way PH runs, collapsed to 3
    python cascade/stage1.py compare --runs nn_curves logreg default     # paired, on test clouds

Labels are the family GROUP, never delta-tilde:

    poisson -> poisson     thomas, nested, lgcp -> clustered     matern2 -> repulsive

A near-CSR Thomas pattern is still labelled `clustered`. The classifier nevertheless calls it
poisson, because the poisson class piles all its mass on CSR while the clustered class spreads its
mass over the whole parameter prior, so at CSR the poisson likelihood wins. Where that switch
happens is the detection boundary, and it is MEASURED here (against delta-tilde) rather than
imposed through the labels -- which also keeps the classical delta-tilde out of training.

Split is cloudforger's theta split (train < 7000 <= val < 8000 <= test < 20000), so the test
patterns are exactly the ones every results/classify run was scored on.

Output  cascade/results/<run>/stage1/predictions.npz   case_id, split, posterior (P, 3), classes
            train rows are OUT-OF-FOLD (config stage1.crossfit_folds); val/test from the full fit
        cascade/results/<run>/stage1/report.json        test scores, see `evaluate`
        cascade/results/benchmark_stage1.json           (benchmark; independent of any run)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import (CLASSES, GROUPS, REGIME, RESULTS, ROOT, Data, config_arg, crossfit, fit,
                    load_config, make_classifier, manifest, proba, run_dir, sample_weights,
                    save_predictions, write_json)

DELTA_EDGES = np.array([-np.inf, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0, np.inf])


def label(family: pd.Series) -> np.ndarray:
    return family.map(lambda f: CLASSES.index(REGIME[f])).to_numpy()


def evaluate(case_id: np.ndarray, posterior: np.ndarray, rows: pd.DataFrame) -> dict:
    """Scores under the prior, the call rate of every family, and the detection curve: the
    fraction of each non-poisson family called poisson, per delta-tilde bin."""
    r = rows.loc[case_id]
    y, pred = label(r.family), posterior.argmax(1)
    w = sample_weights(r.family)
    p_true = np.clip(posterior[np.arange(len(y)), y], 1e-12, None)
    out = {"n": int(len(y)),
           "balanced_accuracy": float(np.average(pred == y, weights=w)),
           "nll": float(np.average(-np.log(p_true), weights=w)),
           "calls": {}, "called_poisson_by_delta": {}}
    for family, g in r.assign(pred=pred).groupby("family"):
        out["calls"][family] = {c: float((g.pred == i).mean()) for i, c in enumerate(CLASSES)}
        if family == "poisson":
            continue
        bins = pd.cut(g.delta_tilde, DELTA_EDGES, right=False)
        out["called_poisson_by_delta"][family] = {
            str(b): {"n": int(len(s)), "rate": float((s.pred == 0).mean())}
            for b, s in g.groupby(bins, observed=True)}
    return out


def show(name: str, rep: dict) -> None:
    print(f"\n== {name}   balanced acc {rep['balanced_accuracy']:.4f}   nll {rep['nll']:.4f}   "
          f"(n = {rep['n']})")
    print(f"   {'family':8s} " + " ".join(f"{c:>10s}" for c in CLASSES))
    for family, calls in rep["calls"].items():
        print(f"   {family:8s} " + " ".join(f"{calls[c]:10.3f}" for c in CLASSES))
    print("   called poisson, by delta-tilde bin:")
    for family, bins in rep["called_poisson_by_delta"].items():
        print(f"   {family:8s} " + "  ".join(f"{b}:{v['rate']:.2f}" for b, v in bins.items()))


def train(config: str) -> None:
    cfg = load_config(config)
    c1 = cfg["stage1"]
    if c1["model"] == "nn":
        raise SystemExit("stage1.model = nn trains on a GPU: CONFIG=... sbatch cascade/stage1_nn.sh")
    out = run_dir(cfg, "stage1", config)
    d = Data.load()
    y, w = label(d.family), sample_weights(d.family)
    tr = d.split == "train"

    t0 = time.time()
    model = fit(make_classifier(c1["model"]), d.X[tr], y[tr], w[tr])
    posterior = proba(model, d.X, len(CLASSES))
    print(f"fit {c1['model']} on {tr.sum()} patterns, {len(d.columns)} features, {time.time() - t0:.0f}s")
    posterior[tr] = crossfit(c1["model"], d.X[tr], y[tr], w[tr], d.rows.theta.to_numpy()[tr],
                             c1["crossfit_folds"])
    print(f"cross-fit {c1['crossfit_folds']} folds, {time.time() - t0:.0f}s total")

    te, va = d.split == "test", d.split == "val"
    rep = evaluate(d.case_id[te], posterior[te], d.rows)
    rep |= {"val_balanced_accuracy": evaluate(d.case_id[va], posterior[va], d.rows)["balanced_accuracy"],
            "train_oof_balanced_accuracy": evaluate(d.case_id[tr], posterior[tr], d.rows)["balanced_accuracy"],
            "model": c1["model"], "features": d.columns, "n_train": int(tr.sum())}
    show(f"{cfg['name']}/stage1 ({c1['model']})", rep)
    save_predictions(out / "predictions.npz", d.case_id, d.split, posterior, CLASSES)
    write_json(out / "report.json", rep)


def collapse(posterior5: np.ndarray, families5: list[str]) -> np.ndarray:
    """5-way family posterior (trained on equal family counts) -> 3-way group posterior under the
    balanced-groups prior: reweight each family by 1/|its group| before summing, renormalize."""
    out = np.zeros((len(posterior5), len(CLASSES)))
    for j, f in enumerate(families5):
        g = CLASSES.index(REGIME[f])
        out[:, g] += posterior5[:, j] / len(GROUPS[CLASSES[g]])
    return out / out.sum(1, keepdims=True)


def benchmark(pattern: str) -> None:
    rows = manifest()
    runs: dict[str, list[Path]] = {}
    for p in sorted((ROOT / "results").glob(pattern)):
        if (p / "predictions.npz").exists():
            runs.setdefault(str(p.parent.relative_to(ROOT / "results")), []).append(p)
    summary = {}
    for run, seeds in runs.items():
        accs, posts, case_id = [], [], None
        for seed in seeds:
            z = np.load(seed / "predictions.npz")
            if case_id is not None and not np.array_equal(case_id, z["case_id"]):
                raise SystemExit(f"{seed}: test patterns differ from the other seeds of this run")
            case_id = z["case_id"]
            families5 = json.loads((seed / "run.json").read_text())["targets"]["classes"]
            post = collapse(z["posterior"], families5)
            accs.append(evaluate(case_id, post, rows)["balanced_accuracy"])
            posts.append(post)
        rep = evaluate(case_id, np.mean(posts, 0), rows)   # seed-averaged posterior
        rep["seed_balanced_accuracy"] = {"mean": float(np.mean(accs)), "sd": float(np.std(accs)),
                                         "n_seeds": len(accs)}
        summary[run] = rep
        print(f"{run:40s} {np.mean(accs):.4f} ± {np.std(accs):.4f}  (n={len(accs)})", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / "benchmark_stage1.json", summary)


def compare(runs: list[str], n_boot: int = 1000, seed: int = 0) -> None:
    """Test balanced accuracy of several runs' stage 1 on the SAME test clouds, and each run's
    paired difference against the first, with a 95% bootstrap interval over test THETAS (the two
    replicates of a theta share parameters, so they are resampled together), as scripts/regimes.py."""
    rows = manifest()
    preds = {}
    for run in runs:
        z = np.load(RESULTS / run / "stage1" / "predictions.npz")
        te = z["split"] == "test"
        preds[run] = pd.Series(list(z["posterior"][te]), index=z["case_id"][te])
    common = sorted(set.intersection(*(set(p.index) for p in preds.values())))
    r = rows.loc[common]
    y, w = label(r.family), sample_weights(r.family)
    correct = {run: np.stack(preds[run].loc[common].to_numpy()).argmax(1) == y for run in runs}
    theta_key = (r.family + "-" + r.theta.astype(str)).to_numpy()
    uniq, inv = np.unique(theta_key, return_inverse=True)
    rng = np.random.default_rng(seed)
    boots = {run: [] for run in runs}
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[inv]
        for run in runs:
            boots[run].append(np.average(correct[run], weights=w * counts))
    ref = runs[0]
    print(f"{len(common)} common test clouds, {len(uniq)} thetas, balanced-groups prior")
    for run in runs:
        acc = np.average(correct[run], weights=w)
        line = f"  {run:12s} {acc:.4f}"
        if run != ref:
            diff = np.array(boots[run]) - np.array(boots[ref])
            lo, hi = np.percentile(diff, [2.5, 97.5])
            line += f"   vs {ref}: {acc - np.average(correct[ref], weights=w):+.4f}  [{lo:+.4f}, {hi:+.4f}]"
        print(line)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    config_arg(sub.add_parser("train"))
    b = sub.add_parser("benchmark")
    b.add_argument("--glob", default="classify/all/*/*/seed_*", help="seed dirs under results/")
    c = sub.add_parser("compare")
    c.add_argument("--runs", nargs="+", required=True, help="run names under cascade/results/; first is the reference")
    args = p.parse_args(argv)
    if args.cmd == "train":
        train(args.config)
    elif args.cmd == "compare":
        compare(args.runs)
    else:
        benchmark(args.glob)


if __name__ == "__main__":
    main()
