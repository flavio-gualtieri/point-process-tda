#!/usr/bin/env python3
"""Routing through the cascade, and the assembled end-to-end output.

    python cascade/pipeline.py [--config ...]      # after stage1, stage2, stage3

                    stage 1                  stage 2 (per group)             stage 3 (per family)
    cloud  ->  poisson ----------------------------------------------------> nbar_hat = n
           ->  clustered  -> thomas | nested | lgcp | reject(->poisson) ---> theta_hat
           ->  repulsive  -> matern2 | reject(->poisson) ------------------> theta_hat

Every arrow is `route_group` / `route_family` below; stages 2 and 3 import them, so training-time
routing (cascade training) and test-time routing are the same code.

Output  cascade/results/<run>/pipeline/clouds.csv   one row per val/test cloud: true family,
            group_hat, family_hat, n, theta_hat (json), delta_tilde
        cascade/results/<run>/pipeline/report.json  end-to-end family accuracy under the prior
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (CLASSES, REGIME, REJECT, RESULTS, TARGETS, config_arg, load_config,
                    load_predictions, manifest, run_dir, sample_weights, write_json)


def stage_dir(cfg: dict, stage: str) -> Path:
    d = RESULTS / cfg["name"] / stage
    if not d.exists():
        raise SystemExit(f"{d} missing -- run cascade/{stage}.py for this config first")
    return d


def route_group(cfg: dict) -> pd.DataFrame:
    """case_id -> split, group_hat. config stage2.routing: argmax of the stage-1 posterior."""
    s1 = load_predictions(stage_dir(cfg, "stage1") / "predictions.npz")
    if cfg["stage2"]["routing"] != "argmax":
        raise SystemExit(f"unknown stage2.routing `{cfg['stage2']['routing']}`")
    return s1[["split"]].assign(group_hat=s1.pred)


def route_family(cfg: dict) -> pd.DataFrame:
    """case_id -> split, group_hat, family_hat. poisson group -> poisson; otherwise stage 2's call,
    with `reject` -> poisson. Train rows use out-of-fold predictions at both stages."""
    r = route_group(cfg)
    r["family_hat"] = np.where(r.group_hat == "poisson", "poisson", None)
    for group in CLASSES[1:]:
        s2 = load_predictions(stage_dir(cfg, "stage2") / f"{group}.npz")
        r.loc[s2.index, "family_hat"] = s2.pred.replace({REJECT: "poisson"})
    if r.family_hat.isna().any():
        raise SystemExit(f"{r.family_hat.isna().sum()} routed clouds have no stage-2 prediction")
    return r


def theta_hat(cfg: dict, routes: pd.DataFrame, rows: pd.DataFrame) -> dict[str, dict]:
    """case_id -> {target: value} for the routed family. Poisson: config stage3.poisson = mle."""
    if cfg["stage3"]["poisson"] != "mle":
        raise SystemExit(f"unknown stage3.poisson `{cfg['stage3']['poisson']}`")
    pois = routes.index[routes.family_hat == "poisson"]
    out = {c: {"nbar": float(n)} for c, n in zip(pois, rows.loc[pois, "n"])}
    for family in TARGETS:
        if family == "poisson":
            continue
        z = np.load(stage_dir(cfg, "stage3") / f"{family}.npz")
        names = list(z["targets"])
        out |= {c: dict(zip(names, map(float, t))) for c, t in zip(z["case_id"], z["theta_hat"])}
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    config_arg(p)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    out = run_dir(cfg, "pipeline", args.config)

    rows = manifest()
    r = route_family(cfg)
    r = r[r.split != "train"].copy()
    r["family"] = rows.loc[r.index, "family"]
    r["n"] = rows.loc[r.index, "n"]
    r["delta_tilde"] = rows.loc[r.index, "delta_tilde"]
    th = theta_hat(cfg, r, rows)
    r["theta_hat"] = [json.dumps(th[c]) for c in r.index]
    r.to_csv(out / "clouds.csv")

    rep = {}
    for split, g in r.groupby("split"):
        w = sample_weights(g.family)
        rep[split] = {
            "n": int(len(g)),
            "family_balanced_accuracy": float(np.average(g.family_hat == g.family, weights=w)),
            "group_balanced_accuracy": float(np.average(g.group_hat == g.family.map(REGIME), weights=w)),
            "calls": {f: s.family_hat.value_counts(normalize=True).round(4).to_dict()
                      for f, s in g.groupby("family")}}
    write_json(out / "report.json", rep)
    t = rep["test"]
    print(f"{cfg['name']}: test family acc {t['family_balanced_accuracy']:.4f}, "
          f"group acc {t['group_balanced_accuracy']:.4f}  (n = {t['n']})")
    for f, calls in t["calls"].items():
        print(f"   {f:8s} " + "  ".join(f"{k}:{v:.3f}" for k, v in sorted(calls.items())))


if __name__ == "__main__":
    main()
