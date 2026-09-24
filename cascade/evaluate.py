#!/usr/bin/env python3
"""End-to-end score of the assembled cascade: does its fitted model reproduce each test pattern?

    python cascade/evaluate.py [--config ...] [--workers 16] [--limit N]      # sbatch cascade/evaluate.sh

The scores are cascade/scoring's (self-contained; see its README and module docstrings). Both are
proper scoring rules, lower is better, and compare MODELS ON THE SAME PATTERN:
    kernel  primary. local_kernel.Component(x, R, kind, h, grid, centres), built ONCE per cloud and
            reused for every model (its kernel width is fixed from x, which is what makes the models'
            scores comparable), then .scores(sims)[tau_mult]. Strictly proper for the law of
            radius-R local configurations.
    dss     secondary. Dawid-Sebastiani on dss.statistics, restricted to the NO_PH groups so no
            persistent-homology statistic overlaps with the PersLay models being scored.

Clouds: scoring.bank.pick -- test split, replicate 0, config evaluation.clouds.per_bin per
(family, delta-tilde bin), plus a Poisson noise floor. Every pipeline variant is scored on the SAME
clouds with the SAME oracle and CSR simulations; only the fit differs.

Per cloud, models simulated with scoring.bank.simulate (bank samplers, n in [20, 2000], never the
bank's seeds):
    oracle   the true family at the true theta (scoring.bank.true_model)       -- the floor
    csr      poisson at nbar = n                                                -- no structure
    fit      each variant's (family_hat, theta_hat), from pipeline.assemble + pipeline.clouds.
             Identical fits share one simulation set; a fit that ends at poisson IS csr.
Variants: config evaluation.variants is a set of NAMED variant sets, each
    {stage2: <model>, stage3: <model> | {family: <model>}, taus: all | [...], stage3_taus: [same | t]}
i.e. one stage-2 model and, per family, one stage-3 model (stage 1 is always the run's), crossed
with the taus. `--set NAME` scores one set, written to evaluation/NAME/; sets are independent runs.

Per variant and score:
    regret = S(fit) - S(oracle)          >= 0 in expectation; 0 = as good as the truth
    gain   = S(csr) - S(oracle)          how much structure there is to find (same for every variant)
    skill  = 1 - sum(regret) / sum(gain) pooled over clouds, only where gain is resolved
             (mean > skill_min_z standard errors): 1 = as good as the truth, 0 = no better than CSR
reported overall, by true family, by family x delta-tilde bin, and split by where the pipeline
ended: `poisson` (stage 1 or a reject sent it there) vs `family`. On structured clouds sent to
poisson, regret = gain, so "might as well be poisson" is the claim that their gain is ~0.

Output  cascade/results/<run>/evaluation/<set>/clouds.csv    one row per (cloud, variant)
        cascade/results/<run>/evaluation/<set>/report.json   the summaries above
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from functools import lru_cache

import numpy as np
import pandas as pd

from common import DEFAULT_CONFIG, RESULTS, load_config, load_predictions, manifest, write_json
import pipeline

sys.path.insert(0, str(RESULTS.parent / "scoring"))
import bank as sbank                                         # noqa: E402  (cascade/scoring)
import dss                                                   # noqa: E402
from local_kernel import Component                           # noqa: E402

M_FALLBACK = 1024          # LGCP grid when an estimate has no admissible one (bank rule, lgcp_grid)


# ------------------------------------------------------------------------------ fitted -> sampler

@lru_cache(maxsize=1)
def _tables():
    from cloudforger.departure.tables import Tables
    return Tables()


def sampler_kwargs(family: str, theta: dict) -> dict:
    """theta_hat (keyed by the TARGETS) -> the bank sampler's arguments. LGCP is estimated as
    (nbar, sigma2, s); the sampler wants mu_log = log nbar - sigma2 / 2 and a grid M, picked as the
    bank picks it."""
    if family != "lgcp":
        return dict(theta)
    from cloudforger.simulation.lgcp_grid import grid_size
    nbar, sigma2, s = theta["nbar"], theta["sigma2"], theta["s"]
    M = grid_size(sigma2, s, nbar, _tables()) or M_FALLBACK
    return {"mu_log": float(np.log(nbar) - sigma2 / 2), "sigma2": sigma2, "s": s, "M": int(M)}


def fit_key(family: str, theta: dict) -> str:
    return f"{family}:" + json.dumps({k: round(v, 10) for k, v in sorted(theta.items())})


# ------------------------------------------------------------------------------------- scoring

def score_cloud(job: dict) -> list[dict]:
    ev, cid, x = job["ev"], job["case_id"], job["x"]
    kc, dc = ev["kernel"], ev["dss"]
    rng = sbank.rng_for(ev["seed"], cid, "component")
    comp = Component(x, kc["R"], kc["kind"], kc["h"], kc["grid"], kc["centres"], rng)
    cols = dss.columns(getattr(dss, dc["groups"].upper()) if isinstance(dc["groups"], str) else tuple(dc["groups"]))
    t_x = dss.statistics(x)[cols] if dc["enabled"] else None
    k = max(kc["sims"], dc["sims"] if dc["enabled"] else 0)

    def score(key, family, kwargs):
        sims = sbank.simulate(family, kwargs, k, sbank.rng_for(ev["seed"], cid, key))
        out = {"kernel": comp.scores(sims[:kc["sims"]], rng, kc["pool"])[kc["tau_mult"]]}
        if dc["enabled"]:
            T = np.stack([dss.statistics(s) for s in sims[:dc["sims"]]])[:, cols]
            out["dss"] = dss.score(t_x, T)
        return out

    csr_key = fit_key("poisson", {"nbar": float(len(x))})
    S = {"oracle": score("oracle", *job["oracle"]), csr_key: score("csr", "poisson", {"nbar": float(len(x))})}
    rows = []
    for v in job["variants"]:
        key = csr_key if v["family_hat"] == "poisson" else fit_key(v["family_hat"], v["theta_hat"])
        if key not in S:
            S[key] = score(key, v["family_hat"], sampler_kwargs(v["family_hat"], v["theta_hat"]))
        row = {"case_id": cid, "variant": v["variant"], "family_hat": v["family_hat"],
               "ended": "poisson" if v["family_hat"] == "poisson" else "family"}
        for s in S["oracle"]:
            row |= {f"{s}_fit": S[key][s], f"{s}_oracle": S["oracle"][s], f"{s}_csr": S[csr_key][s],
                    f"{s}_regret": S[key][s] - S["oracle"][s], f"{s}_gain": S[csr_key][s] - S["oracle"][s]}
        rows.append(row)
    return rows


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

    bins = pd.cut(df.delta_tilde, [-np.inf, 0, 1, 4, np.inf], right=False).astype(str)
    out = {}
    for variant, g in df.groupby("variant"):
        b = bins[g.index]
        structured = g[g.family != "poisson"]
        out[variant] = {
            "overall": cell(g),
            "by_family": {f: cell(h) for f, h in g.groupby("family")},
            "by_family_delta": {f"{f} {d}": cell(h) for (f, d), h in g.groupby([g.family, b])},
            # the test of "might as well be poisson": structured clouds the pipeline sent to poisson
            "structured_by_end": {e: cell(h) for e, h in structured.groupby("ended")},
            "structured_by_end_delta": {f"{e} {d}": cell(h) for (e, d), h in structured.groupby([structured.ended, b[structured.index]])},
        }
    return out


# ----------------------------------------------------------------------------------------- main

def variants(cfg: dict, name: str) -> list[tuple[str, str, object, float, float]]:
    """(label, stage-2 model, stage-3 model or {family: model}, stage-2 tau, stage-3 tau)."""
    v = cfg["evaluation"]["variants"][name]
    taus = cfg["regime"]["taus"] if v["taus"] == "all" else v["taus"]
    return [(f"{name}|tau{t:g}|s3tau{(t if s3 == 'same' else float(s3)):g}", v["stage2"], v["stage3"], t,
             t if s3 == "same" else float(s3)) for t in taus for s3 in v["stage3_taus"]]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    p.add_argument("--set", required=True, help="a name under evaluation.variants")
    p.add_argument("--limit", type=int, help="score only the first N clouds (smoke test)")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    ev = cfg["evaluation"]
    out = RESULTS / cfg["name"] / "evaluation" / args.set
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    c = ev["clouds"]
    chosen = sbank.pick(c["per_bin"], c["delta_edges"], c["seed"])
    if args.limit:
        chosen = chosen.sample(args.limit, random_state=0)
    rows = manifest()
    s1 = load_predictions(RESULTS / cfg["name"] / "stage1" / "predictions.npz").loc[chosen.index]

    per_cloud = {cid: [] for cid in chosen.index}
    for name, m2, m3, t2, t3 in variants(cfg, args.set):
        r = pipeline.assemble(cfg, t2, m2, s1, rows)
        missing = [f for f in ("thomas", "nested", "matern2", "lgcp")
                   if pipeline.load(cfg, t3, f, pipeline.stage3_model(m3, f)) is None]
        if r is None or missing:
            print(f"skip {name}: components missing ({'stage 2' if r is None else ', '.join(missing)})", flush=True)
            continue
        th = pipeline.clouds(cfg, r, t3, m3)
        for cid, row in th.iterrows():
            per_cloud[cid].append({"variant": name, "family_hat": row.family_hat, "theta_hat": json.loads(row.theta_hat)})
    xs = sbank.observed(chosen)
    jobs = [{"case_id": cid, "x": xs[cid], "oracle": sbank.true_model(chosen.loc[cid]), "variants": per_cloud[cid],
             "ev": ev} for cid in chosen.index if per_cloud[cid]]
    n_fits = np.mean([len({(v["family_hat"], json.dumps(v["theta_hat"])) for v in j["variants"]}) for j in jobs])
    print(f"{len(jobs)} clouds, {len(jobs[0]['variants'])} variants, {n_fits:.1f} distinct fits per cloud on "
          f"average, {args.workers} workers ({time.time() - t0:.0f}s)", flush=True)

    recs = []
    with mp.get_context("fork").Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(score_cloud, jobs), 1):
            recs += r
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} clouds, {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(recs).join(chosen[["family", "delta_tilde", "n"]], on="case_id")
    df.to_csv(out / "clouds.csv", index=False)
    scores = ["kernel"] + (["dss"] if ev["dss"]["enabled"] else [])
    rep = summarize(df, scores, ev["skill_min_z"])
    write_json(out / "report.json", {"evaluation": ev | {"variants": ev["variants"][args.set]}, "variants": rep})

    for variant, v in rep.items():
        o, e = v["overall"], v["structured_by_end"]
        line = f"{variant:34s} n={o['n']}"
        for s in scores:
            line += (f" | {s}: regret {o[s]['regret']:+.4g} skill {o[s]['skill'] if o[s]['skill'] is None else round(o[s]['skill'], 3)}"
                     + (f", sent-to-poisson gain {e['poisson'][s]['gain']:+.3g}±{e['poisson'][s]['gain_se']:.2g}" if "poisson" in e else ""))
        print(line)
    print(f"-> {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
