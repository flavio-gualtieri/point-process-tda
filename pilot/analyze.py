#!/usr/bin/env python3
"""The pilot's experiments, all gradient-boosted trees (cascade.common's hgb), one per feature set.

    python pilot/analyze.py            # after pilot/featurize.py assemble

1. stage 1     poisson | clustered | repulsive on the 7 grouped families (balanced-groups prior).
               Where it sends `cell` (K = pi r^2, held out) is reported by k. Out-of-fold routing on
               train (stage1_features) fits the regime cutoffs, as cascade/regime_analysis.py does.
2. stage 2     clustered: thomas | nested | lgcp | ring, repulsive: matern2 | matern1, trained on all
               of the branch's train clouds (tau 0, no reject), scored on the branch's test clouds
               (oracle routing) -- all, and in regime. Ring vs Thomas by jitter, Matern I vs II by core.
3. inference   per family and target, log-target regression; RMSE(log)/s.d. averaged over targets,
               on all test clouds and in regime. The question: does PH beat classical anywhere?
4. end to end  family accuracy over the 7 families: cascades (stage-1 features -> stage-2 features)
               against one-shot 7-way classifiers, under the balanced-groups prior.
5. 8-way       one-shot with `cell` as a class (uniform prior): can anything recognise it?

Every interval is a 95% bootstrap over test thetas, resampled within family, and every comparison
is paired (same resamples). Output: pilot/results/{report.json, summary.md, predictions.npz}.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from shared import DATA, RESULTS, TARGETS, coordinate, equal_weights, families, group_weights, load_config, rows

from common import crossfit, fit, make_classifier, make_regressor, proba     # cascade/common.py
from regime import fit_cutoff                                                # cascade/regime.py

CLASSES = ["poisson", "clustered", "repulsive"]
BRANCHES = {"clustered": ["thomas", "nested", "lgcp", "ring"], "repulsive": ["matern2", "matern1"]}
CASCADES = [("classical", "classical"), ("classical", "ph"), ("classical", "classical+ph"),
            ("classical+ph", "classical+ph"), ("ph", "ph")]
ONESHOT = ["classical", "ph", "classical+ph"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------ bootstrap

class Boot:
    """Resamples of the test thetas, drawn within each family. Statistics are ratios of per-theta
    sums, so a resample is one matrix product: counts (B, thetas) @ sums (thetas,)."""

    def __init__(self, test: pd.DataFrame, B: int, seed: int):
        self.codes, uniq = pd.factorize(test.family + ":" + test.theta.astype(str))
        fam = pd.Index(uniq).str.split(":").str[0].to_numpy()
        rng = np.random.default_rng(seed)
        self.counts = np.zeros((B + 1, len(uniq)), np.float64)
        self.counts[0] = 1.0                                   # row 0 = the point estimate
        for f in np.unique(fam):
            g = np.flatnonzero(fam == f)
            draws = rng.integers(0, len(g), (B, len(g)))
            for b in range(B):
                self.counts[b + 1, g] = np.bincount(draws[b], minlength=len(g))

    def total(self, v: np.ndarray) -> np.ndarray:
        """Sum of per-test-row values v under every resample: (B + 1,)."""
        return self.counts @ np.bincount(self.codes, weights=np.asarray(v, float), minlength=self.counts.shape[1])


def summary(x: np.ndarray) -> dict:
    """Row 0 is the estimate, the rest the resamples."""
    lo, hi = np.nanpercentile(x[1:], [2.5, 97.5])
    return {"est": float(x[0]), "lo": float(lo), "hi": float(hi)}


def accuracy(boot: Boot, family: np.ndarray, correct: np.ndarray, prior: dict, mask=None) -> np.ndarray:
    """sum_f prior_f * recall_f over the families in `prior` (renormalised), per resample."""
    mask = np.ones(len(family), bool) if mask is None else mask
    total = sum(prior.values())
    out = 0.0
    for f, p in prior.items():
        m = (family == f) & mask
        out = out + p / total * boot.total(correct & m) / np.maximum(boot.total(m), 1e-12)
    return out


def rmse_sd(boot: Boot, err2: np.ndarray, m: np.ndarray, sd: float) -> np.ndarray:
    return np.sqrt(boot.total(np.where(m, err2, 0.0)) / np.maximum(boot.total(m), 1e-12)) / sd


def balanced_groups(groups: dict) -> dict:
    """Family -> prior weight: 1/3 per group, split evenly within the group."""
    size = pd.Series(groups).value_counts()
    return {f: 1 / (len(size) * size[g]) for f, g in groups.items()}


# ------------------------------------------------------------------------------------------ data

def load_features(cfg: dict, table: pd.DataFrame) -> tuple[dict, dict]:
    blocks = {}
    for comp in {c for comps in cfg["feature_sets"].values() for c in comps}:
        parts, names = [], None
        for f in families(cfg):
            z = np.load(DATA / "features" / comp / f"{f}.npz")
            parts.append(pd.DataFrame(z["X"], index=z["case_id"]))
            names = list(z["names"])
        blocks[comp] = (pd.concat(parts).loc[table.index].to_numpy(np.float32), names)
    X = {k: np.hstack([blocks[c][0] for c in comps]) for k, comps in cfg["feature_sets"].items()}
    names = {k: sum((blocks[c][1] for c in comps), []) for k, comps in cfg["feature_sets"].items()}
    return X, names


def in_regime(cut: dict, sub: pd.DataFrame, family: str) -> np.ndarray:
    ub = cut["u_boundary"][next(iter(cut["u_boundary"]))]
    if ub is None:
        return np.zeros(len(sub), bool)
    u = np.log(coordinate(sub, family, cut["coordinate"])) + cut["nbar_exponent"] * np.log(sub.nbar.to_numpy(float))
    return u >= ub if cut["direction"] == "increasing" else u <= ub


def bins(x: np.ndarray, edges) -> np.ndarray:
    labels = [f"{a:g}-{b:g}" for a, b in zip(edges[:-1], edges[1:])]
    return pd.cut(x, edges, labels=labels, include_lowest=True).astype(str)


# ------------------------------------------------------------------------------------------ main

def main() -> None:
    cfg = load_config()
    threads = os.environ.get("SLURM_CPUS_PER_TASK")
    if threads:
        os.environ["OMP_NUM_THREADS"] = threads
    RESULTS.mkdir(parents=True, exist_ok=True)
    table = rows(cfg)
    X, names = load_features(cfg, table)
    log(f"{len(table)} rows, feature sets " + ", ".join(f"{k} ({v.shape[1]})" for k, v in X.items()))

    fam = table.family.to_numpy()
    split = table.split.to_numpy()
    tr, te = np.flatnonzero(split == "train"), np.flatnonzero(split == "test")
    test = table.iloc[te]
    tfam = test.family.to_numpy()
    boot = Boot(test, cfg["bootstrap"], cfg["seed"])
    grouped = np.isin(fam, list(cfg["groups"]))
    y_group = np.array([CLASSES.index(cfg["groups"][f]) if f in cfg["groups"] else -1 for f in fam])
    prior7 = balanced_groups(cfg["groups"])
    report = {"config": cfg, "rows": table.groupby(["family", "split"]).size().unstack().to_dict()}
    preds = {"case_id": test.index.to_numpy(str)}

    # ---------------------------------------------------------------- 1. stage 1 + regime cutoffs
    s1 = {}
    idx = tr[grouped[tr]]
    w1 = group_weights(table.family.iloc[idx], cfg["groups"])
    for fs in ONESHOT:
        s1[fs] = proba(fit(make_classifier("hgb"), X[fs][idx], y_group[idx], w1), X[fs][te], 3)
        preds[f"stage1_{fs}"] = s1[fs]
        log(f"stage 1 {fs}")
    key = pd.factorize(table.family.iloc[idx] + ":" + table.theta.iloc[idx].astype(str))[0]
    oof = crossfit("hgb", X[cfg["stage1_features"]][idx], y_group[idx], w1, key, cfg["crossfit_folds"], cfg["seed"])
    log("stage 1 out-of-fold")

    tg = y_group[te]
    rep1 = {}
    k_edges = [2, 2.5, 4.5, 9.5, 30]
    for fs in ONESHOT:
        call = s1[fs].argmax(1)
        acc = accuracy(boot, tfam, call == tg, prior7)
        calls = {f: {c: float((call[tfam == f] == i).mean()) for i, c in enumerate(CLASSES)} for f in np.unique(tfam)}
        cell = test.family.to_numpy() == "cell"
        kb = bins(test.k.to_numpy()[cell], k_edges)
        cell_by_k = {b: {c: float((call[cell][kb == b] == i).mean()) for i, c in enumerate(CLASSES)} for b in np.unique(kb)}
        rep1[fs] = {"group_accuracy": summary(acc), "calls": calls, "cell_by_k": cell_by_k}
    report["stage1"] = rep1

    cutoffs = {}
    tr_rows = table.iloc[idx]
    for f, coord in cfg["regime"]["coordinates"].items():
        m = tr_rows.family.to_numpy() == f
        sub = tr_rows[m]
        own = oof[m].argmax(1) == CLASSES.index(cfg["groups"][f])
        c = fit_cutoff(coordinate(sub, f, coord), sub.nbar.to_numpy(float), own, [cfg["regime"]["tau"]])
        c["coordinate"], c["routed_own"] = coord, float(own.mean())
        cutoffs[f] = c
    report["cutoffs"] = cutoffs
    regime = np.ones(len(test), bool)                      # poisson and cell: everything
    for f, c in cutoffs.items():
        m = tfam == f
        regime[m] = in_regime(c, test[m], f)
    report["test_in_regime"] = {f: float(regime[tfam == f].mean()) for f in cutoffs}
    log("regime cutoffs: " + ", ".join(f"{f} {v:.2f}" for f, v in report["test_in_regime"].items()))

    # ---------------------------------------------------------------- 2. stage 2 (oracle routing)
    s2 = {fs: {} for fs in cfg["feature_sets"]}
    rep2 = {}
    for fs in cfg["feature_sets"]:
        rep2[fs] = {}
        for branch, fams in BRANCHES.items():
            i = tr[np.isin(fam[tr], fams)]
            y = np.array([fams.index(f) for f in fam[i]])
            p = proba(fit(make_classifier("hgb"), X[fs][i], y, equal_weights(table.family.iloc[i])), X[fs][te], len(fams))
            s2[fs][branch] = p
            preds[f"stage2_{branch}_{fs}"] = p
            call = np.array(fams)[p.argmax(1)]
            ok = call == tfam
            prior = {f: 1.0 for f in fams}
            out = {"accuracy": summary(accuracy(boot, tfam, ok, prior)),
                   "accuracy_in_regime": summary(accuracy(boot, tfam, ok, prior, regime)),
                   "confusion": {f: {g: float((call[tfam == f] == g).mean()) for g in fams} for f in fams}}
            if branch == "clustered":
                ring = tfam == "ring"
                jb = bins((test.sigma / test.rho).to_numpy()[ring], [0.05, 0.1, 0.2, 0.4, 1.0])
                out["ring_by_jitter"] = {b: {g: float((call[ring][jb == b] == g).mean()) for g in fams}
                                         for b in np.unique(jb)}
            else:
                out["by_core"] = {}
                for f in fams:
                    m = tfam == f
                    cb = bins((test.R * np.sqrt(test.nbar)).to_numpy()[m], [0.05, 0.1, 0.2, 0.34, 0.55])
                    out["by_core"][f] = {b: float((call[m][cb == b] == f).mean()) for b in np.unique(cb)}
            rep2[fs][branch] = out
        log(f"stage 2 {fs}")
    report["stage2"] = rep2

    # ---------------------------------------------------------------- 3. parameter inference
    rep3, err = {}, {}
    for f, targets in TARGETS.items():
        i = tr[fam[tr] == f]
        m = tfam == f
        rep3[f] = {}
        for fs in cfg["feature_sets"]:
            e2 = np.zeros((len(te), len(targets)))
            for j, t in enumerate(targets):
                y = np.log(table[t].to_numpy(float))
                model = fit(make_regressor("hgb"), X[fs][i], y[i], np.ones(len(i)))
                pr = model.predict(X[fs][te][m])
                lo, hi = y[i].min(), y[i].max()
                e2[m, j] = (np.clip(pr, lo, hi) - y[te][m]) ** 2
                preds[f"theta_{f}_{t}_{fs}"] = np.full(len(te), np.nan)
                preds[f"theta_{f}_{t}_{fs}"][m] = np.exp(np.clip(pr, lo, hi))
            sds = [np.log(table[t].to_numpy(float))[te][m].std() for t in targets]
            err[(f, fs)] = {
                "all": np.mean([rmse_sd(boot, e2[:, j], m, sds[j]) for j in range(len(targets))], axis=0),
                "in_regime": np.mean([rmse_sd(boot, e2[:, j], m & regime, sds[j]) for j in range(len(targets))], axis=0),
                "per_target": {t: float(rmse_sd(boot, e2[:, j], m, sds[j])[0]) for j, t in enumerate(targets)}}
            rep3[f][fs] = {"all": summary(err[(f, fs)]["all"]), "in_regime": summary(err[(f, fs)]["in_regime"]),
                           "per_target": err[(f, fs)]["per_target"]}
        for fs in cfg["feature_sets"]:
            if fs != "classical":
                for sub in ("all", "in_regime"):
                    rep3[f][fs][f"{sub}_minus_classical"] = summary(err[(f, fs)][sub] - err[(f, "classical")][sub])
        log(f"inference {f}")
    report["inference"] = rep3

    # ---------------------------------------------------------------- 4. end to end (7 families)
    seven = np.isin(tfam, list(cfg["groups"]))
    e2e, correct = {}, {}
    for a, b in CASCADES:
        g = s1[a].argmax(1)
        hat = np.full(len(te), "poisson", dtype=object)
        for branch, fams in BRANCHES.items():
            m = g == CLASSES.index(branch)
            hat[m] = np.array(fams)[s2[b][branch][m].argmax(1)]
        correct[f"cascade {a} -> {b}"] = hat == tfam
    labels = list(cfg["groups"])
    i = tr[grouped[tr]]
    y7 = np.array([labels.index(f) for f in fam[i]])
    for fs in ONESHOT:
        p = proba(fit(make_classifier("hgb"), X[fs][i], y7, w1), X[fs][te], len(labels))
        preds[f"oneshot7_{fs}"] = p
        correct[f"one-shot {fs}"] = np.array(labels)[p.argmax(1)] == tfam
        log(f"one-shot 7 {fs}")
    acc = {k: accuracy(boot, tfam, c, prior7, seven) for k, c in correct.items()}
    acc_regime = {k: accuracy(boot, tfam, c, prior7, seven & regime) for k, c in correct.items()}
    for k, c in correct.items():
        e2e[k] = {"accuracy": summary(acc[k]), "accuracy_in_regime": summary(acc_regime[k]),
                  "recall": {f: float(c[tfam == f].mean()) for f in labels}}
    pairs = [("one-shot classical+ph", "one-shot classical"),
             ("cascade classical -> classical", "one-shot classical"),
             ("cascade classical -> classical+ph", "one-shot classical+ph"),
             ("cascade classical -> classical+ph", "cascade classical -> classical"),
             ("cascade classical+ph -> classical+ph", "one-shot classical+ph")]
    report["end_to_end"] = {"models": e2e, "differences": {
        f"{a}  minus  {b}": {"all": summary(acc[a] - acc[b]), "in_regime": summary(acc_regime[a] - acc_regime[b])}
        for a, b in pairs}}

    # ---------------------------------------------------------------- 5. 8-way with cell
    labels8 = families(cfg)
    y8 = np.array([labels8.index(f) for f in fam[tr]])
    rep5 = {}
    for fs in ONESHOT:
        p = proba(fit(make_classifier("hgb"), X[fs][tr], y8, equal_weights(table.family.iloc[tr])), X[fs][te], len(labels8))
        call = np.array(labels8)[p.argmax(1)]
        cell = tfam == "cell"
        kb = bins(test.k.to_numpy()[cell], k_edges)
        rep5[fs] = {"macro_accuracy": summary(accuracy(boot, tfam, call == tfam, {f: 1.0 for f in labels8})),
                    "cell_calls": pd.Series(call[cell]).value_counts(normalize=True).round(4).to_dict(),
                    "cell_recall_by_k": {b: float((call[cell][kb == b] == "cell").mean()) for b in np.unique(kb)},
                    "called_cell": {f: float((call[tfam == f] == "cell").mean()) for f in labels8}}
        log(f"8-way {fs}")
    report["eight_way"] = rep5

    (RESULTS / "report.json").write_text(json.dumps(report, indent=1, default=float))
    np.savez(RESULTS / "predictions.npz", **preds)
    (RESULTS / "summary.md").write_text(render(report, cfg))
    log(f"wrote {RESULTS}")


# ------------------------------------------------------------------------------------ summary.md

def ci(s: dict, digits: int = 3) -> str:
    return f"{s['est']:.{digits}f} [{s['lo']:.{digits}f}, {s['hi']:.{digits}f}]"


def render(r: dict, cfg: dict) -> str:
    fsets = list(cfg["feature_sets"])
    L = ["# Family pilot", "",
         "Gradient-boosted trees throughout; feature sets differ only in what the trees see. "
         "Intervals: 95% bootstrap over test thetas (within family), paired across models.", ""]

    L += ["## Parameter inference: RMSE(log θ)/s.d., mean over targets (lower is better)", "",
          "| family | " + " | ".join(fsets) + " | PH − classical | classical+PH − classical |",
          "|---|" + "---|" * (len(fsets) + 2)]
    for f, d in r["inference"].items():
        L.append(f"| {f} | " + " | ".join(f"{d[fs]['all']['est']:.3f}" for fs in fsets)
                 + f" | {ci(d['ph']['all_minus_classical'])} | {ci(d['classical+ph']['all_minus_classical'])} |")
    L += ["", "In regime (τ = %g):" % cfg["regime"]["tau"], "",
          "| family | " + " | ".join(fsets) + " | PH − classical | classical+PH − classical |",
          "|---|" + "---|" * (len(fsets) + 2)]
    for f, d in r["inference"].items():
        if f == "cell":
            continue
        L.append(f"| {f} | " + " | ".join(f"{d[fs]['in_regime']['est']:.3f}" for fs in fsets)
                 + f" | {ci(d['ph']['in_regime_minus_classical'])} | {ci(d['classical+ph']['in_regime_minus_classical'])} |")

    L += ["", "## Stage 2 with oracle routing: balanced accuracy", "",
          "| branch | " + " | ".join(fsets) + " |", "|---|" + "---|" * len(fsets)]
    for b in BRANCHES:
        L.append(f"| {b} | " + " | ".join(ci(r["stage2"][fs][b]["accuracy"]) for fs in fsets) + " |")
        L.append(f"| {b}, in regime | " + " | ".join(ci(r["stage2"][fs][b]["accuracy_in_regime"]) for fs in fsets) + " |")
    L += ["", "Ring called ring / thomas, by jitter σ/ρ:", "", "| jitter | " + " | ".join(fsets) + " |",
          "|---|" + "---|" * len(fsets)]
    for jb in r["stage2"]["classical"]["clustered"]["ring_by_jitter"]:
        L.append(f"| {jb} | " + " | ".join(
            f"{r['stage2'][fs]['clustered']['ring_by_jitter'][jb]['ring']:.2f} / "
            f"{r['stage2'][fs]['clustered']['ring_by_jitter'][jb]['thomas']:.2f}" for fs in fsets) + " |")
    L += ["", "Matérn recall by core size R√n̄ (matern2 / matern1):", "", "| core | " + " | ".join(fsets) + " |",
          "|---|" + "---|" * len(fsets)]
    cores = sorted(set(r["stage2"]["classical"]["repulsive"]["by_core"]["matern2"])
                   | set(r["stage2"]["classical"]["repulsive"]["by_core"]["matern1"]), key=lambda s: float(s.split("-")[0]))
    for cb in cores:
        cell = lambda fs, f: r["stage2"][fs]["repulsive"]["by_core"][f].get(cb)
        L.append(f"| {cb} | " + " | ".join(
            " / ".join("—" if cell(fs, f) is None else f"{cell(fs, f):.2f}" for f in ("matern2", "matern1"))
            for fs in fsets) + " |")

    L += ["", "## Stage 1 (7 families) and where it sends `cell`", "",
          "| features | group accuracy | " + " | ".join(f"cell k {b}: P / C / R" for b in r["stage1"]["classical"]["cell_by_k"]) + " |",
          "|---|---|" + "---|" * len(r["stage1"]["classical"]["cell_by_k"])]
    for fs, d in r["stage1"].items():
        L.append(f"| {fs} | {ci(d['group_accuracy'])} | " + " | ".join(
            " / ".join(f"{v[c]:.2f}" for c in CLASSES) for v in d["cell_by_k"].values()) + " |")
    L += ["", "Regime cutoffs (from stage-1 out-of-fold routing on train):", "",
          "| family | coordinate | n̄ exponent | routed to own group | test in regime |", "|---|---|---|---|---|"]
    for f, c in r["cutoffs"].items():
        L.append(f"| {f} | {c['coordinate']} | {c['nbar_exponent']:.2f} | {c['routed_own']:.3f} | {r['test_in_regime'][f]:.3f} |")

    L += ["", "## End to end: family accuracy over the 7 grouped families (balanced-groups prior)", "",
          "| model | all | in regime | " + " | ".join(cfg["groups"]) + " |", "|---|---|---|" + "---|" * len(cfg["groups"])]
    for k, d in r["end_to_end"]["models"].items():
        L.append(f"| {k} | {ci(d['accuracy'])} | {ci(d['accuracy_in_regime'])} | "
                 + " | ".join(f"{d['recall'][f]:.2f}" for f in cfg["groups"]) + " |")
    L += ["", "| difference | all | in regime |", "|---|---|---|"]
    for k, d in r["end_to_end"]["differences"].items():
        L.append(f"| {k} | {ci(d['all'])} | {ci(d['in_regime'])} |")

    L += ["", "## 8-way one-shot with `cell` as a class (uniform prior)", "",
          "| features | macro accuracy | cell recall by k | other families called cell |", "|---|---|---|---|"]
    for fs, d in r["eight_way"].items():
        L.append(f"| {fs} | {ci(d['macro_accuracy'])} | "
                 + ", ".join(f"{b}: {v:.2f}" for b, v in d["cell_recall_by_k"].items()) + " | "
                 + ", ".join(f"{f} {v:.2f}" for f, v in d["called_cell"].items() if f != "cell" and v >= 0.01) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
