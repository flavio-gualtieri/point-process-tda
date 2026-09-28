"""Throwaway negative control for the end-to-end kernel score (delete this folder when done).

Uses the package's scoring code unchanged: kernel.Component with the pipeline's evaluation
settings (configs/pipeline.yaml -> evaluation.kernel) and simulate.simulate for model draws.

Per scenario: draw N independent observed clouds x from the TRUE model; on each x score a ladder of
candidate models with the same Component (same tau), 16 simulations each, as endtoend.score_cloud
does. Everything is reported as regret = S(model) - S(oracle) on the same cloud.
"""

from __future__ import annotations

import json
import os
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

from cloudforger.scores.kernel import Component
from cloudforger.scores.simulate import simulate

OUT = Path(__file__).parent
KC = {"R": 1.5, "kind": "point", "h": 0.1, "grid": 21, "centres": 300, "pool": 1000, "sims": 16,
      "tau_mult": "1.0"}
N_CLOUDS = int(os.environ.get("N_CLOUDS", 40))

# ---------------------------------------------------------------------------------- the models
# Thomas: 20 crisp clusters of 20 points (sigma = 0.025, omega = sigma sqrt(kappa) = 0.11), nbar 400.
TH = {"kappa": 20.0, "mu": 20.0, "sigma": 0.025}
# Matern II at nbar ~ 400: hard core R with pi R^2 nbar = 0.8 (strongly regular).
M2 = {"R": float(np.sqrt(0.8 / (400 * np.pi))), "lam_p": 400 * np.log(5) / 0.8}


def thomas(k=1.0, s=1.0):
    """Thomas at nbar 400 with k x as many clusters (mu / k points each) and sigma x s."""
    return ("thomas", {"kappa": TH["kappa"] * k, "mu": TH["mu"] / k, "sigma": TH["sigma"] * s})


def matern2(r=1.0):
    """Matern II with core R x r, lam_p adjusted to keep nbar ~ 400."""
    R = M2["R"] * r
    a = np.pi * R**2
    fill = 0.8 * r**2                      # pi R^2 nbar at nbar 400 ...
    return ("matern2", {"R": R, "lam_p": -np.log(1 - fill) / a})   # ... nbar = (1 - e^{-lam_p a}) / a


SIGMA_MULTS = [0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.7, 8.0, 16.0]
K_MULTS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 20.0]        # 20 -> clusters of 1 point = CSR
CORE_MULTS = [0.25, 0.5, 0.7, 0.85, 1.0, 1.1]                 # r 1.12 would be fill 1.0

SCENARIOS = {
    "thomas": {
        "truth": thomas(),
        "ladder": {"oracle": thomas(), "thomas, sigma x2": thomas(s=2), "thomas, 4x clusters": thomas(k=4),
                   "thomas, sigma x4": thomas(s=4), "CSR": None, "matern2 (wrong)": matern2()},
        "sweeps": {"sigma": {m: thomas(s=m) for m in SIGMA_MULTS},
                   "clusters": {m: thomas(k=m) for m in K_MULTS}},
    },
    "matern2": {
        "truth": matern2(),
        "ladder": {"oracle": matern2(), "matern2, core x0.7": matern2(0.7), "matern2, core x0.5": matern2(0.5),
                   "CSR": None, "thomas (wrong)": thomas()},
        "sweeps": {"core": {m: matern2(m) for m in CORE_MULTS}},
    },
}


def rng_for(*keys) -> np.random.Generator:
    return np.random.default_rng([zlib.crc32(str(k).encode()) for k in keys])


def one_cloud(args):
    scen, i = args
    S = SCENARIOS[scen]
    x = simulate(*S["truth"], 1, rng_for("x", scen, i))[0]
    rng = rng_for("component", scen, i)
    comp = Component(x, KC["R"], KC["kind"], KC["h"], KC["grid"], KC["centres"], rng)

    def score(tag, model):
        family, kw = model if model is not None else ("poisson", {"nbar": float(len(x))})
        sims = simulate(family, kw, KC["sims"], rng_for("sims", scen, i, tag))
        return comp.scores(sims, rng, KC["pool"])[KC["tau_mult"]]

    out = {"n": len(x), "ladder": {k: score(k, m) for k, m in S["ladder"].items()},
           "sweeps": {name: {str(m): score(f"{name}{m}", mod) for m, mod in sw.items()}
                      for name, sw in S["sweeps"].items()}}
    return scen, i, out


def main():
    tasks = [(s, i) for s in SCENARIOS for i in range(N_CLOUDS)]
    res = {s: [None] * N_CLOUDS for s in SCENARIOS}
    with Pool(int(os.environ.get("JOBS", 16))) as pool:
        for k, (s, i, out) in enumerate(pool.imap_unordered(one_cloud, tasks)):
            res[s][i] = out
            print(f"{k + 1}/{len(tasks)}", file=sys.stderr, flush=True)
    (OUT / "results.json").write_text(json.dumps({"kernel": KC, "scenarios": res}, indent=1))


if __name__ == "__main__":
    main()
