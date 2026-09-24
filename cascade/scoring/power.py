#!/usr/bin/env python3
"""Power check: does a candidate score prefer the true model to CSR where it obviously should?

    python cascade/scoring/power.py [--config cascade/scoring/config.yaml] [--workers 16] [--limit N]

Per cloud, two models are simulated -- oracle (true family, true theta) and csr (poisson at
nbar = n) -- and every score variant is computed for both. gain = S(csr) - S(oracle), lower S being
better, so gain > 0 means the score prefers the truth. Reported per score variant and per
(family, delta-tilde bin):

    mean gain, sd, z = mean / se, d' = mean / sd (per-cloud separability), fraction of clouds > 0

z grows with the number of clouds, d' does not: d' is the property of the score, z says whether
this sample resolves it. Poisson clouds are the noise floor (z should be ~0: any systematic gain
there is bias, e.g. from finite-M estimation, not power).

Variants:
  energy_w                  the whole-pattern W1 energy score that failed (baseline)
  dss/all, dss/no_ph        Dawid-Sebastiani on all 14 statistics / the non-PH subset
  dss/-<group>              leaving one statistic group out
  kernel/<R>/<kind>/<tau>   one local-configuration component
  kernel/sum/<tau>          all components summed; tau in {0.5,1,2,4} x median, or lin (limit)

Output  cascade/scoring/results/<name>/clouds.csv   per cloud: every variant's oracle, csr, gain
        cascade/scoring/results/<name>/report.json  the table above, per variant
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bank                                             # noqa: E402
import dss                                              # noqa: E402
import energy_w                                         # noqa: E402
from local_kernel import TAU_MULTS, Component           # noqa: E402


def score_cloud(job: dict) -> dict:
    cfg, cid, x, row = job["cfg"], job["case_id"], job["x"], job["row"]
    t0 = time.time()
    models = {"oracle": bank.true_model(row), "csr": ("poisson", {"nbar": float(len(x))})}
    k = max(cfg["sims"][s] for s in ("dss", "kernel", "energy_w") if cfg[s]["enabled"])
    sims = {m: bank.simulate(f, kw, k, bank.rng_for(cfg["seed"], cid, m)) for m, (f, kw) in models.items()}
    out: dict[str, float] = {}

    def put(variant: str, model: str, value: float):
        out[f"{variant}|{model}"] = value

    if cfg["energy_w"]["enabled"]:
        for m, s in sims.items():
            rng = bank.rng_for(cfg["seed"], cid, f"energy_w/{m}")
            put("energy_w", m, energy_w.score(x, s[:cfg["sims"]["energy_w"]], rng, cfg["energy_w"]["max_points"]))

    if cfg["dss"]["enabled"]:
        t_x = dss.statistics(x)
        subsets = {"all": tuple(dss.GROUPS), "no_ph": dss.NO_PH}
        subsets |= {f"-{g}": tuple(o for o in dss.GROUPS if o != g) for g in dss.GROUPS}
        for m, s in sims.items():
            T = np.stack([dss.statistics(p) for p in s[:cfg["sims"]["dss"]]])
            for name, groups in subsets.items():
                c = dss.columns(groups)
                put(f"dss/{name}", m, dss.score(t_x[c], T[:, c]))

    if cfg["kernel"]["enabled"]:
        kc = cfg["kernel"]
        rng = bank.rng_for(cfg["seed"], cid, "kernel")
        comps = {(R, kind): Component(x, R, kind, kc["h"], kc["grid"], kc["centres"], rng)
                 for R in kc["radii"] for kind in kc["kinds"]}
        for m, s in sims.items():
            total: dict[str, float] = {}
            for (R, kind), comp in comps.items():
                for tau, v in comp.scores(s[:cfg["sims"]["kernel"]], rng, kc["pool"]).items():
                    put(f"kernel/{R:g}/{kind}/{tau}", m, v)
                    total[tau] = total.get(tau, 0.0) + (0.0 if np.isnan(v) else v)
            for tau, v in total.items():
                put(f"kernel/sum/{tau}", m, v)

    out["seconds"] = time.time() - t0
    return {"case_id": cid, **out}


def summarize(df: pd.DataFrame, variants: list[str], edges: list[float]) -> dict:
    bins = pd.cut(df.delta_tilde, edges, right=False).astype(str)
    cell_key = np.where(df.family == "poisson", "poisson", df.family + " " + bins)
    rep = {}
    for v in variants:
        g = df[f"{v}|csr"] - df[f"{v}|oracle"]
        cells = {}
        for key, s in g.groupby(cell_key):
            s = s.dropna()
            sd = s.std(ddof=1) if len(s) > 1 else np.nan
            cells[key] = {"n": int(len(s)), "mean": float(s.mean()), "sd": float(sd),
                          "z": float(s.mean() / (sd / np.sqrt(len(s)))) if sd > 0 else np.nan,
                          "d_prime": float(s.mean() / sd) if sd > 0 else np.nan,
                          "frac_pos": float((s > 0).mean())}
        rep[v] = cells
    return rep


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(HERE / "config.yaml"))
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    p.add_argument("--limit", type=int, help="score only the first N clouds (smoke test)")
    args = p.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = HERE / "results" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(Path(args.config).read_text())

    c = cfg["clouds"]
    chosen = bank.pick(c["per_bin"], c["delta_edges"], c["seed"])
    print(f"picked {len(chosen)} clouds; loading patterns", flush=True)
    if args.limit:
        chosen = chosen.groupby("family").head(max(1, args.limit // len(bank.FAMILIES)))
    xs = bank.observed(chosen)
    jobs = [{"cfg": cfg, "case_id": cid, "x": xs[cid], "row": row} for cid, row in chosen.iterrows()]
    print(f"{len(jobs)} clouds, {args.workers} workers", flush=True)

    t0, results = time.time(), []
    with mp.get_context("fork").Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(score_cloud, jobs), 1):
            results.append(r)
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} clouds, {time.time() - t0:.0f}s", flush=True)

    df = chosen[["family", "delta_tilde", "n"]].join(pd.DataFrame(results).set_index("case_id"))
    variants = sorted({c.split("|")[0] for c in df.columns if c.endswith("|oracle")})
    for v in variants:
        df[f"{v}|gain"] = df[f"{v}|csr"] - df[f"{v}|oracle"]
    df.to_csv(out / "clouds.csv")
    rep = summarize(df, variants, c["delta_edges"])
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=float))

    top = f"[{c['delta_edges'][-2]}, {c['delta_edges'][-1]})"
    heads = ["poisson"] + [f"{f} {top}" for f in bank.FAMILIES if f != "poisson"]
    print(f"\nd' = mean gain / sd  (z in brackets); columns: Poisson floor, then delta-tilde {top}")
    print(f"{'variant':28s}" + "".join(f"{h.split()[0]:>16s}" for h in heads))
    for v in variants:
        cells = rep[v]
        print(f"{v:28s}" + "".join(
            f"{cells[h]['d_prime']:>+8.2f} ({cells[h]['z']:>+5.1f})" if h in cells else f"{'-':>16s}"
            for h in heads))
    print(f"\nmedian {df.seconds.median():.1f}s per cloud, max {df.seconds.max():.0f}s")


if __name__ == "__main__":
    main()
