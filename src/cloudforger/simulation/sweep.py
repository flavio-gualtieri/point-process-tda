"""One theta per (family, index) from its PARAMS stream; replicate patterns from PATTERN streams.

Draw order (part of the dataset definition):
    1. nbar  ~ log-uniform
    2. delta ~ log-uniform          (not for poisson)
    3. shape ~ log-uniform on the shape box at nbar; redraw 3 until the shape is valid,
       the target delta is reachable, and (LGCP) a grid size exists
Amplitude: solved so that delta-tilde(theta, nbar) = delta.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from ..departure.tables import Tables
from .families import CONFIG, FAMILIES, Family, Rules, cv, solve
from .lgcp_grid import grid_size
from .processes import SAMPLERS, lgcp_eigenvalues
from .seeding import PARAMS, PATTERN, case_rng

DATA = Path(__file__).resolve().parents[3] / "data" / "simulation"


@dataclass(frozen=True)
class Config:
    root: int
    nbar: tuple[float, float]
    delta: tuple[float, float]
    thetas: int
    reps: int
    n_range: tuple[int, int]
    shard_size: int

    @classmethod
    def load(cls, path: Path = CONFIG) -> Config:
        c = yaml.safe_load(path.read_text())
        pair = lambda d: (d["low"], d["high"])
        return cls(int(c["root"]), pair(c["nbar"]), pair(c["delta"]), int(c["thetas"]), int(c["reps"]),
                   pair(c["n"]), int(c["shard_size"]))


def log_uniform(rng, lo, hi):
    return float(lo * (hi / lo) ** rng.random())


def draw_theta(fam: Family, tables: Tables, cfg: Config, index: int, max_tries: int = 1000) -> dict:
    rng = case_rng(cfg.root, "sweep", fam.name, index, PARAMS)
    nbar = log_uniform(rng, *cfg.nbar)
    if fam.name == "poisson":
        return {"nbar": nbar, "delta": 0.0, "shape": {}, "amp": None, "model": fam.model(nbar, {}, None), "shape_tries": 0}
    target = log_uniform(rng, *cfg.delta)
    for tries in range(1, max_tries + 1):
        shape = {k: log_uniform(rng, lo, hi) for k, (lo, hi) in fam.shape_box(nbar).items()}
        if not fam.valid_shape(nbar, shape):
            continue
        amp = solve(fam, tables, nbar, shape, target)
        if amp is None:
            continue
        model = fam.model(nbar, shape, amp)
        if fam.name == "lgcp":
            model["M"] = grid_size(amp, shape["s"], nbar, tables)
            if model["M"] is None:
                continue
        return {"nbar": nbar, "delta": target, "shape": shape, "amp": amp, "model": model, "shape_tries": tries,
                "cv": cv(fam, nbar, shape, amp)}
    raise RuntimeError(f"{fam.name} theta {index}: no feasible shape in {max_tries} draws")


def sample_patterns(fam_name: str, theta: dict, cfg: Config, index: int, max_tries: int = 10_000):
    model = dict(theta["model"])
    if fam_name == "lgcp":
        model["root_lam"] = lgcp_eigenvalues(model["sigma2"], model["s"], model["M"])
    lo, hi = cfg.n_range
    for rep in range(cfg.reps):
        rng = case_rng(cfg.root, "sweep", fam_name, cfg.reps * index + rep, PATTERN)
        for tries in range(1, max_tries + 1):
            pts = SAMPLERS[fam_name](rng, **model)
            if lo <= len(pts) <= hi:
                yield rep, pts, tries
                break
        else:
            raise RuntimeError(f"{fam_name} theta {index}: no pattern with n in [{lo}, {hi}]")


def run_shard(fam_name: str, shard: int, cfg: Config) -> Path:
    path = DATA / "shards" / fam_name / f"shard_{shard:04d}.npz"
    if path.exists():
        return path
    tables = Tables()
    fam = FAMILIES[fam_name](Rules.load())
    points, rows = [], []
    for index in range(shard * cfg.shard_size, min((shard + 1) * cfg.shard_size, cfg.thetas)):
        theta = draw_theta(fam, tables, cfg, index)
        for rep, pts, tries in sample_patterns(fam_name, theta, cfg, index):
            points.append(pts)
            rows.append({"family": fam_name, "theta": index, "rep": rep, "nbar": theta["nbar"], "delta": theta["delta"],
                         "cv": theta.get("cv", math.sqrt(1 / theta["nbar"])), "amplitude": theta["amp"],
                         "shape_tries": theta["shape_tries"], **theta["shape"], **theta["model"],
                         "n": len(pts), "pattern_tries": tries})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, points=np.concatenate(points), sizes=np.array([len(p) for p in points]),
             manifest=np.array(json.dumps(rows)))
    tmp.replace(path)
    return path
