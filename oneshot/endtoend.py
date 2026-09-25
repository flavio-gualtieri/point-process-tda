#!/usr/bin/env python3
"""End-to-end score of whole one-shot pipelines: does the fitted model reproduce the pattern?

    python oneshot/endtoend.py [--config ...] [--score-on heldout|same] [--limit N]    # sbatch oneshot/endtoend.sh

A pipeline (config evaluation.pipelines) is a classifier plus an estimator assignment, exactly as in
compare.py (`best` = compare's validation choice per family, read from compare/report.json). Per
test cloud it gives a fit: poisson at nbar_hat = n, or (family_hat, theta_hat).

Scoring is cascade/evaluate.py's score_cloud, unchanged: kernel score on local configurations
(primary) and Dawid-Sebastiani on non-PH geometric statistics (secondary), both proper, lower is
better, and three models simulated per cloud:
    oracle  the true family at the true theta        -- the floor
    csr     poisson at nbar = n(x)                   -- no structure
    fit     each pipeline's fit (identical fits share one simulation set)
    regret = S(fit) - S(oracle)    gain = S(csr) - S(oracle)    skill = 1 - sum(regret) / sum(gain)

score_on (config evaluation.score_on, or --score-on):
    heldout  the fit comes from replicate 0 and is scored against replicate 1 of the same theta --
             an independent pattern, so the fit cannot be flattered by having seen it
    same     scored against replicate 0 itself (the cascade's evaluation, for comparison)

Clouds: test split, stratified without delta-tilde (config evaluation.clouds): per family with a
regime coordinate, per_bin clouds in each of P(detected | u) < 0.5, 0.5-0.9, >= 0.9 (compare's frozen
cutoffs); cell by k bins; per_bin x 3 poisson clouds. Every pipeline is scored on the same clouds
with the same oracle and CSR simulations.

Output  oneshot/results/<run>/evaluation/<score_on>/{clouds.csv, report.json, summary.md}
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

from compare import in_regime
from core import (ROOT, TARGETS, config_arg, estimated_families, load_classifier, load_config, load_estimator,
                  rows as load_rows, run_dir, save_config, write_json)

sys.path.insert(0, str(ROOT / "cascade" / "scoring"))
import bank as sbank                                         # noqa: E402  (cascade/scoring)
from evaluate import score_cloud                             # noqa: E402  (cascade/evaluate.py)

STRATA = ("below 0.5", "0.5-0.9", "above 0.9")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------- clouds

def pick(cfg: dict, r: pd.DataFrame, cuts: dict) -> pd.DataFrame:
    """Test clouds of replicate 0, with a `stratum` column; see the module doc."""
    c = cfg["evaluation"]["clouds"]
    test = r[(r.split == "test") & (r.rep == 0)].copy()
    test["stratum"] = "all"
    for f in cuts:
        m = (test.family == f).to_numpy()
        hi, mid = in_regime(cuts, test[m], 0.9), in_regime(cuts, test[m], 0.5)
        test.loc[m, "stratum"] = np.select([hi, mid], [STRATA[2], STRATA[1]], STRATA[0])
    m = (test.family == "cell").to_numpy()
    e = c["cell_k_edges"]
    test.loc[m, "stratum"] = pd.cut(test.k[m], e, right=False,
                                    labels=[f"k {a}-{b - 1}" for a, b in zip(e[:-1], e[1:])]).astype(str)
    frames = []
    for f in cfg["families"]:
        g = test[test.family == f]
        if f == "poisson":
            frames.append(g.sample(3 * c["per_bin"], random_state=c["seed"]))
            continue
        frames += [h.sample(min(c["per_bin"], len(h)), random_state=c["seed"]) for _, h in g.groupby("stratum")]
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

def summarize(df: pd.DataFrame, scores: list[str], min_z: float) -> dict:
    def se(v):
        return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("inf")

    def cell(g):
        out = {"n": int(len(g))}
        for s in scores:
            reg, gain = g[f"{s}_regret"], g[f"{s}_gain"]
            resolved = gain.mean() > min_z * se(gain)
            out[s] = {"regret": float(reg.mean()), "regret_se": se(reg), "gain": float(gain.mean()),
                      "gain_se": se(gain), "skill": float(1 - reg.sum() / gain.sum()) if resolved else None,
                      "fit_beats_csr": float((g[f"{s}_fit"] < g[f"{s}_csr"]).mean())}
        return out

    out = {}
    for variant, g in df.groupby("variant"):
        structured = g[g.family != "poisson"]
        out[variant] = {
            "overall": cell(g),
            "structured": cell(structured),
            "by_family": {f: cell(h) for f, h in g.groupby("family")},
            "by_family_stratum": {f"{f} | {s}": cell(h) for (f, s), h in g.groupby(["family", "stratum"])},
            "structured_by_end": {e: cell(h) for e, h in structured.groupby("ended")},
            "identified": {str(k): cell(h) for k, h in structured.groupby(structured.family_hat == structured.family)},
        }
    return out


def fmt(c: dict, s: str) -> str:
    v = c[s]
    skill = "—" if v["skill"] is None else f"{v['skill']:.2f}"
    return f"{skill} (regret {v['regret']:+.2g} ± {v['regret_se']:.1g})"


def render(rep: dict, cfg: dict, score_on: str, scores: list[str], families: list[str]) -> str:
    variants = list(rep)
    L = [f"# End-to-end evaluation: {cfg['name']}, scored on {score_on}", "",
         "Skill = 1 − Σregret / Σgain: 1 = as good as the true model, 0 = no better than CSR; shown only "
         "where the gain (true model vs CSR) is resolved. Regret = S(fit) − S(true), ≥ 0 in expectation. "
         + ("Fits come from replicate 0 and are scored on the independent replicate 1."
            if score_on == "heldout" else "Fits are scored on the pattern they were estimated from."), ""]
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
    p.add_argument("--score-on", choices=["heldout", "same"])
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    p.add_argument("--limit", type=int, help="score only N clouds (smoke test)")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    ev = cfg["evaluation"]
    score_on = args.score_on or ev["score_on"]
    out = run_dir(cfg, "evaluation", score_on + (f"_limit{args.limit}" if args.limit else ""))
    save_config(args.config, out)
    t0 = time.time()

    report = json.loads((run_dir(cfg, "compare") / "report.json").read_text())
    cuts, best = report["cutoffs"], report["best"]
    r = load_rows(cfg)
    chosen = pick(cfg, r, cuts)
    if args.limit:
        chosen = chosen.sample(args.limit, random_state=0)
    scored = chosen if score_on == "same" else r.loc[[partner(c) for c in chosen.index]]
    log(f"{len(chosen)} clouds ({chosen.groupby('family').size().to_dict()}), scored on {score_on}")

    per_cloud = {cid: [] for cid in chosen.index}
    for name, spec in ev["pipelines"].items():
        for cid, v in fits(cfg, name, spec, best, chosen).items():
            per_cloud[cid].append(v)
    xs = sbank.observed(scored)
    ev_score = {k: ev[k] for k in ("kernel", "dss", "seed")}
    jobs = []
    for cid, sid in zip(chosen.index, scored.index):
        fam, kw = sbank.true_model(r.loc[cid])
        jobs.append({"case_id": sid, "x": xs[sid], "oracle": (fam, kw), "variants": per_cloud[cid], "ev": ev_score})
    log(f"{len(jobs)} jobs, pipelines {list(ev['pipelines'])}, {args.workers} workers ({time.time() - t0:.0f}s)")

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
    write_json(out / "report.json", {"score_on": score_on, "pipelines": ev["pipelines"], "best": best,
                                     "evaluation": ev, "variants": rep})
    (out / "summary.md").write_text(render(rep, cfg, score_on, scores, cfg["families"]))
    for v, d in rep.items():
        log(f"{v:10s} " + " | ".join(f"{s}: skill overall {fmt(d['overall'], s)}, structured {fmt(d['structured'], s)}"
                                     for s in scores))
    log(f"-> {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
