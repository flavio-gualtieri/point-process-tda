#!/usr/bin/env python3
"""Is the regime a property of the data or of our classifier? The checks behind pipeline.regime.

    python scripts/regime.py [--config ...] [--boot 200] [--workers 16]     # after compare (reads cutoffs.json)
    python scripts/regime.py --size 0.05                                     # classifiers as tests of size 0.05

Detectors. Every classifier of the config (event: not called poisson) and the classical CSR test: the
exact-size extremum test on L(r) - r (departure.tables, default reduction; event: rejects at 5%), which
is model-free -- it knows nothing of the families or of our training. All on the TEST clouds.

  1  one coordinate  On the reference classifier's fitting clouds (out-of-fold train): AUC of the frozen
                     u = log x + a log nbar against boosted trees on every log parameter and log nbar
                     (out of fold over theta). Close AUCs = detection is (nearly) a level set of u.
  2  every detector  On the FROZEN axis u (the reference's a): AUC, and the boundary u*(tau) of an
                     isotonic P(event | u). Same axis for every detector, so the boundaries compare
                     directly; the CSR test's is the model-free detectability boundary. Each detector's
                     OWN fitted exponent a too: does the nbar direction agree?
  3  uncertainty     Bootstrap over test thetas (both replicates together): 95% intervals for every
                     u*(tau) and a.

Poisson's own clouds give each detector's false-detection rate (the CSR test's is its size, ~0.05).
An argmax call is not a test: an equal-prior classifier calls a large share of true Poisson clouds
something else, so its P(detected | u) has a floor far above 0.05 and a boundary at tau sits at a
different power for each detector. --size alpha makes every classifier a test of size alpha: the
event is P(poisson | x) below its alpha quantile over the VAL Poisson clouds, so detectors compare at
equal size and u*(tau) is where each reaches power tau.

Output  <results>/<run>/compare/regime.{md,json}; regime.json also holds P(event | u) in u bins per
        family and detector, for figures.
"""

from __future__ import annotations

import argparse
import json
import os
from multiprocessing import get_context

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from cloudforger.paths import CURVES
from cloudforger.pipeline.core import TARGETS, config_arg, load_classifier, load_config, log, rows as load_rows, write_json
from cloudforger.pipeline.regime import coordinate, fit_cutoff
from cloudforger.pipeline.units import run_dir

CSR_TEST = "CSR test"
BINS = 16


# ------------------------------------------------------------------------------------- detectors

def classifier_events(cfg: dict, model: str, index: pd.Index, rows: pd.DataFrame,
                      size: float | None = None) -> np.ndarray | None:
    """Not called poisson (argmax); with `size`, P(poisson | x) below its `size` quantile on val Poisson clouds."""
    pred = load_classifier(cfg, model)
    if pred is None:
        return None
    classes = [c for c in pred.columns if c != "split"]
    post = pred.reindex(index)[classes].to_numpy()
    if np.isnan(post).any():
        return None
    if size is None:
        return np.array(classes)[post.argmax(1)] != "poisson"
    val = pred[(pred.split == "val") & (rows.reindex(pred.index).family == "poisson").to_numpy()]
    threshold = np.quantile(val["poisson"].to_numpy(), size)
    return post[:, classes.index("poisson")] < threshold


def csr_test_events(r: pd.DataFrame) -> np.ndarray:
    """Rejects CSR at 5%: the default reduction of the exact-size L test, on the stored L(r) - r."""
    from cloudforger.departure.tables import Tables
    tables, out = Tables(), np.zeros(len(r), bool)
    for f in r.family.unique():
        z = np.load(CURVES / f / "fixed" / "curves.npz")
        at = pd.Index(z["case_id"].astype(str)).get_indexer(r.index[r.family == f])
        m = (r.family == f).to_numpy()
        out[m] = tables.statistic(z["L"][at].astype(np.float64), r.n.to_numpy(float)[m]) > 1
    return out


# ------------------------------------------------------------------------------------- boundaries

def frozen_boundary(u: np.ndarray, y: np.ndarray, increasing: bool, taus) -> dict:
    """u*(tau) of an isotonic P(event | u) in the frozen direction; None where tau is never reached."""
    iso = IsotonicRegression(increasing=increasing, out_of_bounds="clip").fit(u, y)
    grid = np.sort(np.unique(u))
    p = iso.predict(grid)
    out = {}
    for tau in taus:
        hit = np.flatnonzero(p >= tau)
        out[str(tau)] = None if not len(hit) else float(grid[hit[0]] if increasing else grid[hit[-1]])
    return out


def one(u, x, nbar, y, increasing, taus) -> dict:
    own = fit_cutoff(x, nbar, y, taus) if 0 < y.mean() < 1 else {"nbar_exponent": np.nan}
    return {"u_star": frozen_boundary(u, y, increasing, taus), "a_own": own["nbar_exponent"]}


def _task(args):
    """Point estimate and bootstrap intervals for one (family, detector)."""
    fam, det, u, x, nbar, y, theta, increasing, taus, B, seed = args
    est = one(u, x, nbar, y, increasing, taus)
    groups = pd.Series(np.arange(len(theta))).groupby(theta).indices
    keys = np.array(list(groups))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(B):
        i = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])
        boots.append(one(u[i], x[i], nbar[i], y[i], increasing, taus))

    def ci(get):
        v = np.array([get(b) for b in boots], float)
        v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) > B // 2 else None

    nan = lambda v: np.nan if v is None else v
    edges = np.quantile(u, np.linspace(0, 1, BINS + 1))
    b = np.clip(np.searchsorted(edges, u, side="right") - 1, 0, BINS - 1)
    return fam, det, {
        "detected": float(y.mean()),
        "auc_u": float(roc_auc_score(y, u if increasing else -u)) if 0 < y.mean() < 1 else None,
        "u_star": est["u_star"], "u_star_ci": {t: ci(lambda bb: nan(bb["u_star"][t])) for t in est["u_star"]},
        "a_own": est["a_own"], "a_own_ci": ci(lambda bb: bb["a_own"]),
        "curve": {"u": [float(u[b == k].mean()) for k in range(BINS)], "p": [float(y[b == k].mean()) for k in range(BINS)]},
    }


def all_parameter_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int = 5) -> float:
    p = np.zeros(len(y))
    for tr, te in GroupKFold(folds).split(X, y, groups):
        m = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, p))


# ------------------------------------------------------------------------------------------- main

def fmt_ci(v, c) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:+.2f}" + (f" [{c[0]:+.2f}, {c[1]:+.2f}]" if c else "")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    config_arg(p)
    p.add_argument("--boot", type=int, default=200)
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    p.add_argument("--size", type=float, help="classifiers as tests of this size (default: argmax calls)")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    taus = cfg["regime"]["taus"]
    cuts = json.loads(run_dir(cfg, "compare", "cutoffs.json").read_text())
    cuts = {f: c for f, c in cuts.items() if "constant" not in c}
    ref = cfg["regime"]["classifier"]
    rows = load_rows(cfg)
    out = {"reference": ref, "taus": taus, "size": args.size, "one_coordinate": {}, "detectors": {}, "poisson": {}}

    # ---- 1. one coordinate, on the reference's fitting clouds
    pred = load_classifier(cfg, ref)
    fit_split = "train" if (pred.split == "train").any() else "val"
    pred = pred[pred.split == fit_split]
    classes = [c for c in pred.columns if c != "split"]
    threshold = next(iter(cuts.values()), {}).get("poisson_threshold")   # the cutoffs' own event
    det_fit = (pred["poisson"].to_numpy() < threshold if threshold is not None
               else np.array(classes)[pred[classes].to_numpy().argmax(1)] != "poisson")
    r_fit = rows.loc[pred.index]
    for f, c in cuts.items():
        m = (r_fit.family == f).to_numpy()
        rf, y = r_fit[m], det_fit[m]
        u = np.log(coordinate(rf, f, c["coordinate"])) + c["nbar_exponent"] * np.log(rf.nbar.to_numpy(float))
        X = np.log(rf[sorted({*TARGETS[f], "nbar"})].to_numpy(float))
        out["one_coordinate"][f] = {"auc_u": float(roc_auc_score(y, u if c["direction"] == "increasing" else -u)),
                                    "auc_all": all_parameter_auc(X, y, rf.theta.to_numpy())}
        log(f"one coordinate {f}: {out['one_coordinate'][f]}")

    # ---- 2, 3. every detector on the test clouds, on the frozen axis, with bootstrap intervals
    test = rows[rows.split == "test"]
    events = {m: classifier_events(cfg, m, test.index, rows, args.size) for m in cfg["classify"]}
    events = {m: e for m, e in events.items() if e is not None}
    events[CSR_TEST] = csr_test_events(test)
    log(f"detectors: {', '.join(events)}")
    pois = (test.family == "poisson").to_numpy()
    out["poisson"] = {d: float(e[pois].mean()) for d, e in events.items()}

    tasks = []
    for f, c in cuts.items():
        m = (test.family == f).to_numpy()
        tf = test[m]
        x, nbar = coordinate(tf, f, c["coordinate"]), tf.nbar.to_numpy(float)
        u = np.log(x) + c["nbar_exponent"] * np.log(nbar)
        for d, e in events.items():
            tasks.append((f, d, u, x, nbar, e[m], tf.theta.to_numpy(), c["direction"] == "increasing", taus,
                          args.boot, 0))
    with get_context("fork").Pool(args.workers) as pool:
        for f, d, res in pool.imap_unordered(_task, tasks):
            out["detectors"].setdefault(f, {})[d] = res
            log(f"{f} / {d}: u*(0.5) {res['u_star'].get('0.5')}")

    # ---- report
    event = "not called poisson" if args.size is None else f"P(poisson | x) below its {args.size:g} quantile on val Poisson clouds (a test of size {args.size:g})"
    L = [f"# Is the regime a property of the data? {cfg['name']}", "",
         f"Frozen axis: u = log x + a log nbar from `{ref}`'s cutoffs (compare/cutoffs.json). Detectors: every "
         f"classifier ({event}) and the {CSR_TEST} (exact-size L test, rejects at 5%), on the test clouds. "
         f"Intervals: 95%, bootstrap over test thetas ({args.boot} resamples).", "",
         "## 1. One coordinate", "",
         f"`{ref}`, {fit_split} clouds{' (out of fold)' if fit_split == 'train' else ''}: AUC of u alone vs boosted "
         "trees on every log parameter and log nbar (out of fold over theta).", "",
         "| family | x | a | AUC u | AUC all parameters | gap |", "|---|---|---|---|---|---|"]
    for f, v in out["one_coordinate"].items():
        L.append(f"| {f} | {cuts[f]['coordinate']} | {cuts[f]['nbar_exponent']:+.2f} | {v['auc_u']:.3f} | "
                 f"{v['auc_all']:.3f} | {v['auc_all'] - v['auc_u']:.3f} |")
    L += ["", "## 2–3. Every detector on the frozen axis", "",
          "Boundaries u*(τ) on the SAME axis per family, so they compare across detectors; `a own` = the exponent "
          "the detector's own logistic fit gives (frozen: the reference's).", "",
          "| family | detector | detected | AUC u | u*(0.5) | u*(0.9) | a own |", "|---|---|---|---|---|---|---|"]
    order = [*[m for m in cfg["classify"] if m in events], CSR_TEST]
    for f in cuts:
        for d in order:
            v = out["detectors"][f][d]
            L.append(f"| {f} | {d} | {v['detected']:.2f} | {v['auc_u']:.3f} | "
                     f"{fmt_ci(v['u_star'].get('0.5'), v['u_star_ci'].get('0.5'))} | "
                     f"{fmt_ci(v['u_star'].get('0.9'), v['u_star_ci'].get('0.9'))} | {fmt_ci(v['a_own'], v['a_own_ci'])} |")
        L.append(f"| | *frozen ({ref}, train)* | | | {cuts[f]['u_boundary'].get('0.5', float('nan')):+.2f} | "
                 f"{cuts[f]['u_boundary'].get('0.9') or float('nan'):+.2f} | {cuts[f]['nbar_exponent']:+.2f} |")
    L += ["", "False detections on Poisson's own test clouds: "
          + ", ".join(f"{d} {v:.3f}" for d, v in out["poisson"].items()) + "."]
    if "cell" in cfg["families"]:
        m = (test.family == "cell").to_numpy()
        k = test.k.to_numpy()[m]
        edges = [2, 3, 5, 10, 20, 31]
        L += ["", "Cell (no monotone coordinate), detection by k:", "",
              "| detector | " + " | ".join(f"k {a}-{b - 1}" for a, b in zip(edges[:-1], edges[1:])) + " |",
              "|---|" + "---|" * (len(edges) - 1)]
        out["cell"] = {}
        for d in order:
            e = events[d][m]
            rates = [float(e[(k >= a) & (k < b)].mean()) for a, b in zip(edges[:-1], edges[1:])]
            out["cell"][d] = dict(zip([f"{a}-{b - 1}" for a, b in zip(edges[:-1], edges[1:])], rates))
            L.append(f"| {d} | " + " | ".join(f"{v:.2f}" for v in rates) + " |")
    stem = "regime" if args.size is None else f"regime_size{args.size:g}"
    write_json(run_dir(cfg, "compare", f"{stem}.json"), out)
    path = run_dir(cfg, "compare", f"{stem}.md")
    path.write_text("\n".join(L) + "\n")
    log(f"-> {path}")


if __name__ == "__main__":
    main()
