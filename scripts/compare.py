#!/usr/bin/env python3
"""Score every trained classifier, estimator and classifier x estimator pipeline on the same test
clouds. Nothing is retrained: a lookup into each unit's predictions.npz. Units not trained yet are
listed and skipped, so this can run at any point.

    python scripts/compare.py [--config ...]

Regime  cutoffs from the reference classifier (config regime.classifier), see pipeline.regime. The rule
        is on theta only, so every model below is scored on the same in-regime clouds.

Scores, test split; 95% intervals from a bootstrap over test thetas (within family), and every
difference against the baseline model is paired (same resamples):
  classifiers  accuracy under the prior (all; in regime at each tau, where poisson and coordinate-less
               families keep every cloud), NLL, recall per family, detection rate per family
  estimators   per family: RMSE(log target) / s.d., averaged over targets. The s.d. is over ALL the
               family's test clouds in every subset, so in-regime numbers are on the same scale. Also on
               val, which is what `best` selects on.
  pipelines    classifier x estimator assignment: accuracy, and RMSE/s.d. on the clouds each family's
               pipeline identifies correctly (poisson: nbar_hat = n). These subsets differ between
               classifiers -- compare estimators on the estimator table, classifiers on accuracy.
  seeds        a config with `seeds` (the ablation) scores every (model, seed) as `model/seed_<s>`,
               pairs it with the baseline at the same seed, and adds mean and s.d. over seeds per model.

Output  <results>/<run>/compare/{summary.md, report.json, cutoffs.json, classifiers.csv,
        estimators.csv, pipelines.csv[, classifiers_by_model.csv, estimators_by_model.csv]}
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd

from cloudforger.pipeline.core import (TARGETS, config_arg, estimated_families, family_weights, load_classifier,
                                       load_config, load_estimator, log, rows as load_rows, run_dir, save_config,
                                       write_json)
from cloudforger.pipeline.regime import fit_cutoffs, in_regime
from cloudforger.pipeline.units import label, seeds


# ------------------------------------------------------------------------------------ bootstrap

class Boot:
    """Resamples of test thetas, within family. Row 0 of every statistic is the point estimate.
    A statistic is a ratio of per-theta sums, so each resample is one product counts @ sums."""

    def __init__(self, test: pd.DataFrame, B: int, seed: int):
        self.codes, uniq = pd.factorize(test.family + ":" + test.theta.astype(str))
        fam = pd.Index(uniq).str.split(":").str[0].to_numpy()
        rng = np.random.default_rng(seed)
        self.counts = np.zeros((B + 1, len(uniq)), np.float32)
        self.counts[0] = 1.0
        for f in np.unique(fam):
            g = np.flatnonzero(fam == f)
            draws = rng.integers(0, len(g), (B, len(g)))
            for b in range(B):
                self.counts[b + 1, g] = np.bincount(draws[b], minlength=len(g))

    def total(self, v) -> np.ndarray:
        return self.counts @ np.bincount(self.codes, weights=np.asarray(v, float), minlength=self.counts.shape[1]).astype(np.float32)


def ci(x: np.ndarray) -> dict:
    lo, hi = np.nanpercentile(x[1:], [2.5, 97.5])
    return {"est": float(x[0]), "lo": float(lo), "hi": float(hi)}


def fmt(s: dict, d: int = 3) -> str:
    return f"{s['est']:.{d}f} [{s['lo']:.{d}f}, {s['hi']:.{d}f}]"


def accuracy(boot, fam, correct, prior, mask) -> np.ndarray:
    out = 0.0
    for f, p in prior.items():
        m = (fam == f) & mask
        out = out + p * boot.total(correct & m) / np.maximum(boot.total(m), 1e-12)
    return out


def rmse_sd(boot, e2: np.ndarray, m: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Mean over targets of RMSE/s.d. on rows m, per resample. e2: (rows, targets)."""
    per = [np.sqrt(boot.total(np.where(m, e2[:, j], 0.0)) / np.maximum(boot.total(m), 1e-12)) / sd[j]
           for j in range(e2.shape[1])]
    return np.mean(per, axis=0)


# ----------------------------------------------------------------------------------------- main

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    config_arg(p)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    cc, taus, fams = cfg["compare"], cfg["regime"]["taus"], cfg["families"]
    out_dir = run_dir(cfg, "compare")
    save_config(cfg, out_dir)
    prior = family_weights(cfg)

    r = load_rows(cfg)
    test = r[r.split == "test"]
    tid, tfam = test.index, test.family.to_numpy()
    boot = Boot(test, cc["bootstrap"], cc["seed"])
    cuts = fit_cutoffs(cfg, r)
    regime = {t: in_regime(cuts, test, t) for t in taus}
    everything = np.ones(len(test), bool)
    subsets = {"all": everything, **{f"regime {t:g}": regime[t] for t in taus}}
    log(f"{len(test)} test clouds; in regime: " + ", ".join(
        f"tau {t:g} " + " ".join(f"{f} {regime[t][tfam == f].mean():.2f}" for f in cuts) for t in taus))
    missing = []

    # (model, seed) -> label; the baseline a label is paired with is the baseline at its seed
    seed_of, model_of = {}, {}
    for m in {*cfg["classify"], *cfg["estimate"]}:
        for seed in seeds(cfg):
            seed_of[label(m, seed)], model_of[label(m, seed)] = seed, m
    base_of = {lab: label(cc["baseline"], seed) for lab, seed in seed_of.items()}

    # --------------------------------------------------------------------------- classifiers
    clf, post = {}, {}
    for m in cfg["classify"]:
        for seed in seeds(cfg):
            lab = label(m, seed)
            pr = load_classifier(cfg, m, seed)
            if pr is None:
                missing.append(f"classify {lab}")
                continue
            post[lab] = pr.reindex(tid)[fams].to_numpy()
            if np.isnan(post[lab]).any():
                raise SystemExit(f"classifier {lab}: predictions missing for some test clouds")
    calls = {m: np.array(fams)[P.argmax(1)] for m, P in post.items()}
    acc = {m: {s: accuracy(boot, tfam, calls[m] == tfam, prior, mask) for s, mask in subsets.items()} for m in post}
    y = np.array([fams.index(f) for f in tfam])
    for m, P in post.items():
        nll = -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None))
        clf[m] = {**{f"accuracy {s}": ci(v) for s, v in acc[m].items()},
                  "nll": float(sum(prior[f] * nll[tfam == f].mean() for f in fams)),
                  "recall": {f: float((calls[m][tfam == f] == f).mean()) for f in fams},
                  "detected": {f: float((calls[m][tfam == f] != "poisson").mean()) for f in fams}}
        if base_of[m] in acc and m != base_of[m]:
            clf[m]["accuracy minus baseline"] = {s: ci(acc[m][s] - acc[base_of[m]][s]) for s in subsets}
    log(f"classifiers: {len(post)}")

    # ---------------------------------------------------------------------------- estimators
    est, e2s, val_err = {}, {}, {}
    val = r[r.split == "val"]
    for f in estimated_families(cfg):
        targets, mine = TARGETS[f], tfam == f
        ytrue = np.log(test[targets].to_numpy(float))
        sd = ytrue[mine].std(0)
        vmask = (val.family == f).to_numpy()
        vtrue = np.log(val[targets].to_numpy(float))[vmask]
        est[f] = {}
        for m, seed in ((m, seed) for m in cfg["estimate"] for seed in seeds(cfg)):
            pr = load_estimator(cfg, f, m, seed)
            m = label(m, seed)
            if pr is None:
                missing.append(f"estimate {f} {m}")
                continue
            e2 = (np.log(pr.reindex(tid)[targets].to_numpy()) - ytrue) ** 2
            e2[~mine] = 0.0
            e2s[(f, m)] = e2
            vhat = np.log(pr.reindex(val.index[vmask])[targets].to_numpy())
            val_err[(f, m)] = float(np.mean(np.sqrt(((vhat - vtrue) ** 2).mean(0)) / vtrue.std(0)))
            est[f][m] = {"val": val_err[(f, m)],
                         **{s: ci(rmse_sd(boot, e2, mine & mask, sd)) for s, mask in subsets.items()},
                         "per_target": dict(zip(targets, (np.sqrt(e2[mine].mean(0)) / sd).round(4).tolist()))}
        for m in est[f]:
            base = base_of[m]
            if m != base and (f, base) in e2s:
                est[f][m]["minus baseline"] = {
                    s: ci(rmse_sd(boot, e2s[(f, m)], mine & mask, sd) - rmse_sd(boot, e2s[(f, base)], mine & mask, sd))
                    for s, mask in subsets.items()}
    best = {f: min((m for (g, m) in val_err if g == f), key=lambda m: val_err[(f, m)]) for f in est if est[f]}
    log(f"estimators: {len(e2s)}; best on val: " + ", ".join(f"{f} {m}" for f, m in best.items()))

    # ----------------------------------------------------------------------------- pipelines
    def assignment(spec) -> dict | None:
        if spec == "best":
            a = dict(best)
        elif isinstance(spec, dict):
            a = {k: v for k, v in spec.items() if k != "name"}
        else:
            a = {f: spec for f in estimated_families(cfg)}
        return a if all((f, m) in e2s for f, m in a.items()) and len(a) == len(estimated_families(cfg)) else None

    chosen = list(post) if cc["classifiers"] == "all" else [m for m in cc["classifiers"] if m in post]
    pipes = {}
    ln_err2 = (np.log(test.n.to_numpy(float)) - np.log(test.nbar.to_numpy(float))) ** 2   # poisson MLE
    pois_sd = np.log(test.nbar.to_numpy(float))[tfam == "poisson"].std()
    for c, spec in itertools.product(chosen, cc["estimators"]):
        a = assignment(spec)
        est_name = spec if isinstance(spec, str) else spec.get("name") or json.dumps(spec, sort_keys=True)
        if a is None:
            missing.append(f"pipeline {c} x {est_name}: an estimator is not trained")
            continue
        call = calls[c].copy()
        if cc["poisson_threshold"] is not None:
            call[post[c][:, fams.index("poisson")] >= cc["poisson_threshold"]] = "poisson"
        rec = {"classifier": c, "estimators": est_name, "assignment": a,
               **{f"accuracy {s}": ci(accuracy(boot, tfam, call == tfam, prior, mask)) for s, mask in subsets.items()}}
        for f in fams:
            hit = (tfam == f) & (call == f)
            if f == "poisson":
                e = rmse_sd(boot, ln_err2[:, None], hit, np.array([pois_sd]))
            else:
                sd = np.log(test[TARGETS[f]].to_numpy(float))[tfam == f].std(0)
                e = rmse_sd(boot, e2s[(f, a[f])], hit, sd)
            rec[f"{f} identified"] = float(hit.sum() / (tfam == f).sum())
            rec[f"{f} rmse/sd"] = ci(e)
        pipes[f"{c} x {est_name}"] = rec
    log(f"pipelines: {len(pipes)}")

    report = {"cutoffs": cuts, "in_regime": {str(t): {f: float(regime[t][tfam == f].mean()) for f in fams} for t in taus},
              "classifiers": clf, "estimators": est, "best": best, "pipelines": pipes, "missing": missing}
    if cfg.get("seeds"):
        report["by_model"] = over_seeds(clf, est, model_of, list(subsets))
    write_json(out_dir / "report.json", report)
    write_json(out_dir / "cutoffs.json", cuts)
    tables(report, subsets, out_dir)
    (out_dir / "summary.md").write_text(render(report, cfg, list(subsets)))
    log(f"wrote {out_dir}" + (f"; {len(missing)} missing units (see summary)" if missing else ""))


def over_seeds(clf: dict, est: dict, model_of: dict, subsets: list[str]) -> dict:
    """Mean and s.d. over seeds of every point estimate, per model (the ablation's table)."""
    def agg(values: list[float]) -> dict:
        v = np.asarray(values, float)
        return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n_seeds": len(v)}

    out = {"classifiers": {}, "estimators": {}}
    for m in dict.fromkeys(model_of[lab] for lab in clf):
        labs = [lab for lab in clf if model_of[lab] == m]
        out["classifiers"][m] = {f"accuracy {s}": agg([clf[lab][f"accuracy {s}"]["est"] for lab in labs]) for s in subsets}
    for f, d in est.items():
        out["estimators"][f] = {}
        for m in dict.fromkeys(model_of[lab] for lab in d):
            labs = [lab for lab in d if model_of[lab] == m]
            out["estimators"][f][m] = {s: agg([d[lab][s]["est"] for lab in labs]) for s in subsets}
    return out


# -------------------------------------------------------------------------------------- outputs

def tables(rep: dict, subsets, out_dir) -> None:
    pd.DataFrame([{"model": m, **{f"accuracy {s}": v[f"accuracy {s}"]["est"] for s in subsets}, "nll": v["nll"],
                   **{f"recall {f}": x for f, x in v["recall"].items()}, **{f"detected {f}": x for f, x in v["detected"].items()}}
                  for m, v in rep["classifiers"].items()]).to_csv(out_dir / "classifiers.csv", index=False)
    pd.DataFrame([{"family": f, "model": m, "val": v["val"], **{s: v[s]["est"] for s in subsets}, **v["per_target"]}
                  for f, d in rep["estimators"].items() for m, v in d.items()]).to_csv(out_dir / "estimators.csv", index=False)
    pd.DataFrame([{k: (v["est"] if isinstance(v, dict) and "est" in v else v) for k, v in p.items() if k != "assignment"}
                  for p in rep["pipelines"].values()]).to_csv(out_dir / "pipelines.csv", index=False)
    if "by_model" in rep:
        bm = rep["by_model"]
        pd.DataFrame([{"model": m, "n_seeds": v["accuracy all"]["n_seeds"],
                       **{f"{k} {stat}": x[stat] for k, x in v.items() for stat in ("mean", "sd")}}
                      for m, v in bm["classifiers"].items()]).to_csv(out_dir / "classifiers_by_model.csv", index=False)
        pd.DataFrame([{"family": f, "model": m, "n_seeds": v["all"]["n_seeds"],
                       **{f"{k} {stat}": x[stat] for k, x in v.items() for stat in ("mean", "sd")}}
                      for f, d in bm["estimators"].items() for m, v in d.items()]).to_csv(out_dir / "estimators_by_model.csv", index=False)


def render(rep: dict, cfg: dict, subsets: list[str]) -> str:
    fams, base = cfg["families"], cfg["compare"]["baseline"]
    L = [f"# Comparison: {cfg['name']}", "",
         f"Test split, 95% bootstrap intervals over test thetas; differences are paired, against `{base}`. "
         f"Prior: {cfg['prior']}.", ""]
    if rep["missing"]:
        L += ["**Not trained yet:** " + "; ".join(rep["missing"]), ""]

    L += ["## Classifiers", "", "| model | " + " | ".join(f"accuracy {s}" for s in subsets) + " | − baseline (all) | NLL |",
          "|---|" + "---|" * (len(subsets) + 2)]
    for m, v in sorted(rep["classifiers"].items(), key=lambda kv: -kv[1]["accuracy all"]["est"]):
        diff = fmt(v["accuracy minus baseline"]["all"]) if "accuracy minus baseline" in v else "—"
        L.append(f"| {m} | " + " | ".join(fmt(v[f"accuracy {s}"]) for s in subsets) + f" | {diff} | {v['nll']:.3f} |")
    L += ["", "Recall (called own family) / detected (not called poisson):", "",
          "| model | " + " | ".join(fams) + " |", "|---|" + "---|" * len(fams)]
    for m, v in rep["classifiers"].items():
        L.append(f"| {m} | " + " | ".join(f"{v['recall'][f]:.2f} / {v['detected'][f]:.2f}" for f in fams) + " |")

    if rep["cutoffs"]:
        reg = cfg["regime"]
        source = (f"the frozen cutoffs of run `{reg['from']}`" if reg.get("from")
                  else f"reference classifier `{reg['classifier']}`, event: {reg['event']}")
        L += ["", f"## Regime ({source})", "",
              "| family | coordinate | n̄ exponent | event rate (fit) | " + " | ".join(f"test in regime τ={t:g}" for t in cfg["regime"]["taus"]) + " |",
              "|---|---|---|---|" + "---|" * len(cfg["regime"]["taus"])]
        for f, c in rep["cutoffs"].items():
            L.append(f"| {f} | {c['coordinate']} | {c['nbar_exponent']:.2f} | {c['event_rate']:.3f} ({c['fitted_on']}) | "
                     + " | ".join(f"{rep['in_regime'][str(t)][f]:.3f}" for t in cfg["regime"]["taus"]) + " |")

    L += ["", "## Estimators: RMSE(log θ)/s.d., mean over targets (lower is better)", ""]
    models = list(dict.fromkeys(m for d in rep["estimators"].values() for m in d))
    for s in subsets:
        L += [f"**{s}** (bold = best on val)", "", "| family | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
        for f, d in rep["estimators"].items():
            cells = [("**" if rep["best"].get(f) == m else "") + (f"{d[m][s]['est']:.3f}" if m in d else "—")
                     + ("**" if rep["best"].get(f) == m else "") for m in models]
            L.append(f"| {f} | " + " | ".join(cells) + " |")
        L.append("")
    L += [f"Difference against `{base}` (all test clouds of the family):", "",
          "| family | " + " | ".join(m for m in models if m != base) + " |", "|---|" + "---|" * (len(models) - 1)]
    for f, d in rep["estimators"].items():
        L.append(f"| {f} | " + " | ".join(fmt(d[m]["minus baseline"]["all"]) if m in d and "minus baseline" in d[m] else "—"
                                            for m in models if m != base) + " |")

    L += ["", "## Pipelines (classifier × estimators)", "",
          "RMSE/s.d. on the clouds each pipeline identifies correctly; identified share in brackets.", "",
          "| pipeline | accuracy all | " + " | ".join(fams) + " |", "|---|---|" + "---|" * len(fams)]
    for name, p in sorted(rep["pipelines"].items(), key=lambda kv: -kv[1]["accuracy all"]["est"]):
        L.append(f"| {name} | {fmt(p['accuracy all'])} | "
                 + " | ".join(f"{p[f'{f} rmse/sd']['est']:.3f} ({p[f'{f} identified']:.2f})" for f in fams) + " |")
    if "by_model" in rep:
        bm = rep["by_model"]
        ms = lambda x: f"{x['mean']:.3f} ± {x['sd']:.3f}"
        L += ["", "## Over seeds (mean ± s.d. of the point estimates)", "",
              "| classifier | seeds | " + " | ".join(f"accuracy {s}" for s in subsets) + " |",
              "|---|---|" + "---|" * len(subsets)]
        for m, v in sorted(bm["classifiers"].items(), key=lambda kv: -kv[1]["accuracy all"]["mean"]):
            L.append(f"| {m} | {v['accuracy all']['n_seeds']} | " + " | ".join(ms(v[f"accuracy {s}"]) for s in subsets) + " |")
        models = list(dict.fromkeys(m for d in bm["estimators"].values() for m in d))
        L += ["", "Estimators, RMSE(log θ)/s.d. over all test clouds of the family:", "",
              "| family | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
        for f, d in bm["estimators"].items():
            L.append(f"| {f} | " + " | ".join(ms(d[m]["all"]) if m in d else "—" for m in models) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
