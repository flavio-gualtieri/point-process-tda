#!/usr/bin/env python3
"""End-to-end evaluation: Wasserstein energy score of the pipeline's fit on test clouds.

    python cascade/evaluate.py [--config ...] [--workers 8]      # after pipeline.py

For a sample of test clouds (config evaluation.clouds_per_family per TRUE family, replicate 0 of
random test thetas) three models are simulated and scored against the observed pattern with
wasserstein.energy_score (read its docstring for every choice):

    fit      the pipeline's (family_hat, theta_hat)
    oracle   the true family at the true theta          -- the floor
    csr      poisson at nbar = n                        -- the no-structure baseline

When the pipeline ends at poisson, `fit` IS `csr` (same model, nbar = n), so its score is copied
rather than re-simulated.

Output  cascade/results/<run>/evaluation/clouds.csv    per cloud: es/cross/within per model,
                                                       regret, gain, family, family_hat, delta
        cascade/results/<run>/evaluation/report.json   regret and skill by true family, by
                                                       family x delta-tilde bin, and overall
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

import simulate
import wasserstein
from common import BANK, config_arg, load_config, manifest, run_dir, write_json
from pipeline import stage_dir

DELTA_EDGES = [-np.inf, 0.0, 1.0, 4.0, np.inf]


def pick(clouds: pd.DataFrame, per_family: int, seed: int) -> pd.DataFrame:
    test0 = clouds[(clouds.split == "test") & clouds.index.str.endswith("-0")]
    return pd.concat([g.sample(min(per_family, len(g)), random_state=seed)
                      for _, g in test0.groupby("family")])


def observed(case_ids: pd.Index, rows: pd.DataFrame) -> dict[str, np.ndarray]:
    out = {}
    for family, ids in pd.Series(case_ids, index=case_ids).groupby(rows.loc[case_ids, "family"]):
        index = pd.read_csv(BANK / family / "manifest.csv", usecols=["case_id"]).case_id
        pos = pd.Series(np.arange(len(index)), index=index).loc[ids.index]
        z = np.load(BANK / family / "points.npz")
        points, offsets = z["points"], z["offsets"]
        out |= {c: points[offsets[i]:offsets[i + 1]] for c, i in pos.items()}
    return out


def score_cloud(job: dict) -> dict:
    ev = job["ev"]
    kw = {"ground": ev["ground"], "p": ev["p"], "max_points": ev["max_points"]}
    x, cid = job["x"], job["case_id"]
    models = {"fit": (job["family_hat"], simulate.from_estimate(job["family_hat"], job["theta_hat"])),
              "oracle": (job["family"], job["oracle"]),
              "csr": ("poisson", {"nbar": float(len(x))})}
    out = {"case_id": cid}
    for name in ["oracle", "csr", "fit"]:
        if name == "fit" and job["family_hat"] == "poisson":
            out |= {k.replace("csr_", "fit_"): v for k, v in out.items() if k.startswith("csr_")}
            continue
        family, kwargs = models[name]
        rng = simulate.rng_for(ev["seed"], cid, name)
        sims = simulate.patterns(family, kwargs, ev["sims"], rng, ev["condition_n"])
        out |= {f"{name}_{k}": v for k, v in wasserstein.energy_score(x, sims, rng, **kw).items()}
    return out


def summarize(df: pd.DataFrame, min_z: float) -> dict:
    """skill = 1 - sum(regret) / sum(gain) is a ratio, and wild when the denominator is noise (near
    CSR the truth and CSR are the same model, so gain ~ 0 +- Monte Carlo error). It is reported only
    where the cell's mean gain exceeds min_z standard errors (config evaluation.skill_min_z)."""
    def se(v):
        return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.inf

    def cell(g):
        resolved = g.gain.mean() > min_z * se(g.gain)
        return {"n": int(len(g)), "regret_mean": float(g.regret.mean()), "regret_se": se(g.regret),
                "gain_mean": float(g.gain.mean()), "gain_se": se(g.gain),
                "skill": float(1 - g.regret.sum() / g.gain.sum()) if resolved else None,
                "routed_right": float((g.family_hat == g.family).mean())}
    bins = pd.cut(df.delta_tilde, DELTA_EDGES, right=False).astype(str)
    return {"overall": cell(df),
            "by_family": {f: cell(g) for f, g in df.groupby("family")},
            "by_family_delta": {f"{f} {b}": cell(g) for (f, b), g in df.groupby([df.family, bins])}}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    config_arg(p)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    ev = cfg["evaluation"]
    if set(ev["references"]) != {"oracle", "csr"}:
        raise SystemExit("evaluation.references: only [oracle, csr] is implemented")
    out = run_dir(cfg, "evaluation", args.config)

    clouds = pd.read_csv(stage_dir(cfg, "pipeline") / "clouds.csv", index_col="case_id")
    chosen = pick(clouds, ev["clouds_per_family"], ev["seed"])
    rows = manifest()
    xs = observed(chosen.index, rows)
    jobs = [{"case_id": c, "x": xs[c], "family": r.family, "family_hat": r.family_hat,
             "theta_hat": json.loads(r.theta_hat), "oracle": simulate.from_manifest(rows.loc[c]),
             "ev": ev} for c, r in chosen.iterrows()]

    t0 = time.time()
    with mp.get_context("fork").Pool(args.workers) as pool:
        scores = []
        for i, s in enumerate(pool.imap_unordered(score_cloud, jobs), 1):
            scores.append(s)
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} clouds, {time.time() - t0:.0f}s", flush=True)

    df = chosen[["family", "family_hat", "n", "delta_tilde"]].join(pd.DataFrame(scores).set_index("case_id"))
    df["regret"] = df.fit_es - df.oracle_es
    df["gain"] = df.csr_es - df.oracle_es
    df.to_csv(out / "clouds.csv")
    rep = summarize(df, ev["skill_min_z"]) | {"evaluation": ev}
    write_json(out / "report.json", rep)

    o = rep["overall"]
    print(f"\n{cfg['name']}: {o['n']} clouds   regret {o['regret_mean']:.4g} ± {o['regret_se']:.2g}   "
          f"gain {o['gain_mean']:.4g}   skill {o['skill']}")
    for f, c in rep["by_family"].items():
        print(f"   {f:8s} n={c['n']:3d}  regret {c['regret_mean']:+.4g}  gain {c['gain_mean']:+.4g}  "
              f"skill {c['skill'] if c['skill'] is None else round(c['skill'], 3)}  "
              f"routed right {c['routed_right']:.2f}")


if __name__ == "__main__":
    main()
