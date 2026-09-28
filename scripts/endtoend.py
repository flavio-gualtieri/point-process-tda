#!/usr/bin/env python3
"""End-to-end score of whole pipelines: fit on one replicate, score on the other.

    python scripts/endtoend.py --set main [--config ...] [--workers N] [--limit N]

A pipeline = a classifier + an estimator assignment, as in compare.py (`best` = compare's
validation choice per family, read from compare/report.json). Each named set in the config's
`evaluation.sets` holds its pipelines and may override any `evaluation.clouds` key; kernel, DSS and
seed settings are shared. Scoring is cloudforger.scores.endtoend: kernel score on local
configurations (primary), Dawid-Sebastiani on non-PH statistics (secondary); regret, gain, skill.

Fits come from replicate 0 of a test theta and are scored against replicate 1 of the same theta --
an independent pattern, so a fit cannot be flattered by having seen what it is scored on.

Clouds: test split, replicate 0, stratified by the regime (P(detected | u) from compare's frozen
cutoffs: below 0.5, 0.5-0.9, above 0.9) for families with a coordinate, by k for cell, plus
per_bin x 3 poisson clouds as the noise floor. `clouds.strata` keeps only the named strata. Every
pipeline of a set is scored on the same clouds with the same oracle and CSR simulations.

Output  <results>/<run>/evaluation/<set>/{clouds.csv, report.json, summary.md, config.yaml}
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd

from cloudforger.pipeline.core import (TARGETS, config_arg, estimated_families, load_classifier, load_config,
                                       load_estimator, log, rows as load_rows, run_dir, save_config, write_json)
from cloudforger.pipeline.regime import in_regime
from cloudforger.scores.endtoend import score_cloud, summarize
from cloudforger.scores.simulate import observed, true_model

STRATA = ("below 0.5", "0.5-0.9", "above 0.9")


# ------------------------------------------------------------------------------------- clouds

def pick(clouds: dict, families: list[str], r: pd.DataFrame, cuts: dict) -> pd.DataFrame:
    """Test clouds of replicate 0, with a `stratum` column; see the module doc."""
    test = r[(r.split == "test") & (r.rep == 0)].copy()
    test["stratum"] = "all"
    for f in cuts:
        m = (test.family == f).to_numpy()
        hi, mid = in_regime(cuts, test[m], 0.9), in_regime(cuts, test[m], 0.5)
        test.loc[m, "stratum"] = np.select([hi, mid], [STRATA[2], STRATA[1]], STRATA[0])
    m = (test.family == "cell").to_numpy()
    e = clouds["cell_k_edges"]
    test.loc[m, "stratum"] = pd.cut(test.k[m], e, right=False,
                                    labels=[f"k {a}-{b - 1}" for a, b in zip(e[:-1], e[1:])]).astype(str)
    if clouds.get("strata"):
        test = test[(test.family == "poisson") | test.stratum.isin(clouds["strata"])]
    frames = []
    for f in families:
        g = test[test.family == f]
        if f == "poisson":
            frames.append(g.sample(clouds.get("poisson", 3 * clouds["per_bin"]), random_state=clouds["seed"]))
            continue
        frames += [h.sample(min(clouds["per_bin"], len(h)), random_state=clouds["seed"]) for _, h in g.groupby("stratum")]
    return pd.concat(frames)


def partner(case_id: str) -> str:
    """The other replicate of the same theta: <family>-<theta>-<rep>."""
    head, rep = case_id.rsplit("-", 1)
    return f"{head}-{1 - int(rep)}"


# ------------------------------------------------------------------------------------ pipelines

def assignment(cfg: dict, spec, best: dict) -> dict:
    if spec == "best":
        return dict(best)
    if isinstance(spec, dict):
        return {k: v for k, v in spec.items() if k != "name"}
    return {f: spec for f in estimated_families(cfg)}


def fits(cfg: dict, name: str, spec: dict, best: dict, chosen: pd.DataFrame) -> dict:
    """case_id (the replicate the fit comes from) -> {variant, family_hat, theta_hat}."""
    post = load_classifier(cfg, spec["classifier"])
    if post is None:
        raise SystemExit(f"pipeline {name}: classifier {spec['classifier']} not trained")
    fams = [c for c in post.columns if c != "split"]
    call = pd.Series(np.array(fams)[post.loc[chosen.index, fams].to_numpy().argmax(1)], index=chosen.index)
    a = assignment(cfg, spec["estimators"], best)
    theta = {f: load_estimator(cfg, f, m) for f, m in a.items()}
    missing = [f"{f}:{a[f]}" for f, t in theta.items() if t is None]
    if missing:
        raise SystemExit(f"pipeline {name}: estimators not trained ({', '.join(missing)})")
    out = {}
    for cid, f in call.items():
        th = ({"nbar": float(chosen.n[cid])} if f == "poisson"
              else {k: float(v) for k, v in theta[f].loc[cid, TARGETS[f]].items()})
        out[cid] = {"variant": name, "family_hat": f, "theta_hat": th}
    return out


# -------------------------------------------------------------------------------------- summary

def fmt(c: dict, s: str) -> str:
    v = c[s]
    skill = "—" if v["skill"] is None else f"{v['skill']:.2f}"
    failed = f", {c['failed']} failed" if c.get("failed") else ""
    return f"{skill} (regret {v['regret']:+.2g} ± {v['regret_se']:.1g}{failed})"


def render(rep: dict, cfg: dict, set_name: str, scores: list[str], families: list[str]) -> str:
    variants = list(rep)
    L = [f"# End-to-end evaluation: {cfg['name']}, set {set_name}", "",
         "Skill = 1 − Σregret / Σgain: 1 = as good as the true model, 0 = no better than CSR; shown only "
         "where the gain (true model vs CSR) is resolved. Regret = S(fit) − S(true), ≥ 0 in expectation. "
         "Fits come from replicate 0 and are scored on the independent replicate 1. "
         "A fit the sampler cannot realise (\"failed\") is scored as CSR, regret = gain.", ""]
    for s in scores:
        L += [f"## {s} score", "", "| | " + " | ".join(variants) + " |", "|---|" + "---|" * len(variants)]
        L.append("| overall | " + " | ".join(fmt(rep[v]["overall"], s) for v in variants) + " |")
        L.append("| structured clouds | " + " | ".join(fmt(rep[v]["structured"], s) for v in variants) + " |")
        for f in families:
            L.append(f"| {f} | " + " | ".join(fmt(rep[v]["by_family"][f], s) if f in rep[v]["by_family"] else "—"
                                            for v in variants) + " |")
        L += ["", "By family and stratum (regime: P(detected | u); cell: k):", "",
              "| family / stratum | n | gain (true vs CSR) | " + " | ".join(variants) + " |",
              "|---|---|---|" + "---|" * len(variants)]
        for key in rep[variants[0]]["by_family_stratum"]:
            c0 = rep[variants[0]]["by_family_stratum"][key]
            L.append(f"| {key} | {c0['n']} | {c0[s]['gain']:+.2g} ± {c0[s]['gain_se']:.1g} | "
                     + " | ".join(fmt(rep[v]["by_family_stratum"][key], s) for v in variants) + " |")
        L += ["", "Structured clouds the pipeline sent to poisson (\"might as well be Poisson\" ⇔ gain ≈ 0):", "",
              "| pipeline | n | gain | n sent to a family | skill there |", "|---|---|---|---|---|"]
        for v in variants:
            e = rep[v]["structured_by_end"]
            p, fa = e.get("poisson"), e.get("family")
            L.append(f"| {v} | {p['n'] if p else 0} | "
                     + (f"{p[s]['gain']:+.2g} ± {p[s]['gain_se']:.1g}" if p else "—")
                     + f" | {fa['n'] if fa else 0} | " + (fmt(fa, s) if fa else "—") + " |")
        L.append("")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------------------- main

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    config_arg(p)
    p.add_argument("--set", dest="set_name", default="main", help="an evaluation.sets entry (default: %(default)s)")
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    p.add_argument("--limit", type=int, help="score only N clouds (smoke test)")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    ev = cfg["evaluation"]
    s = ev["sets"][args.set_name]
    pipelines, clouds = s["pipelines"], {**ev["clouds"], **s.get("clouds", {})}
    out = run_dir(cfg, "evaluation", args.set_name + (f"_limit{args.limit}" if args.limit else ""))
    save_config(args.config, out)
    t0 = time.time()

    report = json.loads((run_dir(cfg, "compare") / "report.json").read_text())
    cuts, best = report["cutoffs"], report["best"]
    r = load_rows(cfg)
    chosen = pick(clouds, cfg["families"], r, cuts)
    if args.limit:
        chosen = chosen.sample(args.limit, random_state=0)
    scored = r.loc[[partner(c) for c in chosen.index]]
    log(f"{len(chosen)} clouds ({chosen.groupby('family').size().to_dict()})")

    per_cloud = {cid: [] for cid in chosen.index}
    for name, spec in pipelines.items():
        for cid, v in fits(cfg, name, spec, best, chosen).items():
            per_cloud[cid].append(v)
    xs = observed(scored)
    ev_score = {k: ev[k] for k in ("kernel", "dss", "seed")}
    jobs = []
    for cid, sid in zip(chosen.index, scored.index):
        jobs.append({"case_id": sid, "x": xs[sid], "oracle": true_model(r.loc[cid]), "variants": per_cloud[cid],
                     "ev": ev_score})
    log(f"{len(jobs)} jobs, pipelines {list(pipelines)}, {args.workers} workers ({time.time() - t0:.0f}s)")

    recs = []
    with mp.get_context("fork").Pool(args.workers) as pool:
        for i, rows_ in enumerate(pool.imap_unordered(score_cloud, jobs), 1):
            recs += rows_
            if i % 25 == 0 or i == len(jobs):
                log(f"  {i}/{len(jobs)} clouds")

    back = dict(zip(scored.index, chosen.index))
    df = pd.DataFrame(recs)
    df["fit_case_id"] = df.case_id.map(back)
    df = df.rename(columns={"case_id": "scored_case_id"}).join(
        chosen[["family", "stratum", "n"]].rename(columns={"n": "n_fit"}), on="fit_case_id")
    df["n_scored"] = df.scored_case_id.map(scored.n)
    df.to_csv(out / "clouds.csv", index=False)
    scores = ["kernel"] + (["dss"] if ev["dss"]["enabled"] else [])
    rep = summarize(df, scores, ev["skill_min_z"])
    write_json(out / "report.json", {"set": args.set_name, "pipelines": pipelines, "best": best,
                                     "clouds": clouds, "variants": rep})
    (out / "summary.md").write_text(render(rep, cfg, args.set_name, scores, cfg["families"]))
    for v, d in rep.items():
        log(f"{v:10s} " + " | ".join(f"{s}: skill overall {fmt(d['overall'], s)}, structured {fmt(d['structured'], s)}"
                                     for s in scores))
    log(f"-> {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
