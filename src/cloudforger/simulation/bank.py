"""The cloud bank: one theta per (family, index) from its PARAMS stream, replicate patterns from
PATTERN streams.

Parameters are drawn in model order with every bound conditioned on what is already drawn
(see simulation.families). Nothing here knows about departure from CSR: delta-tilde is a label
applied afterwards by scripts/relabel.py, so refitting the null tables never moves the bank.
Theta i is drawn from its own stream, so the bank can be extended without disturbing what exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from ..departure.tables import Tables
from .families import CONFIG, FAMILIES, Family, Rules, cv, log_uniform
from .lgcp_grid import grid_size
from .processes import SAMPLERS, lgcp_eigenvalues
from .seeding import PARAMS, PATTERN, case_rng

DATA = Path(__file__).resolve().parents[3] / "data" / "bank"


@dataclass(frozen=True)
class Config:
    root: int
    nbar: tuple[float, float]
    thetas: int
    reps: int
    n_range: tuple[int, int]
    shard_size: int

    @classmethod
    def load(cls, path: Path = CONFIG) -> Config:
        c = yaml.safe_load(path.read_text())
        pair = lambda d: (d["low"], d["high"])
        return cls(int(c["root"]), pair(c["nbar"]), int(c["thetas"]), int(c["reps"]),
                   pair(c["n"]), int(c["shard_size"]))


def draw_theta(fam: Family, tables: Tables, cfg: Config, index: int, max_tries: int = 1000,
               set_: str = "bank") -> dict:
    """Model parameters for one theta. LGCP also needs a grid size, the one place the bank consults
    the null tables; M is written to the manifest, so the pattern stays reproducible from it.
    `set_` picks the stream set (seeding.SET_ID): the bank, or a pilot drawn beside it."""
    rng = case_rng(cfg.root, set_, fam.name, index, PARAMS)
    for tries in range(1, max_tries + 1):
        nbar = log_uniform(rng, *cfg.nbar)
        model = fam.draw(rng, nbar)
        if model is None or cv(fam, model) > fam.rules.cv_max:
            continue
        if fam.name == "lgcp":
            model["M"] = grid_size(model["sigma2"], model["s"], nbar, tables)
            if model["M"] is None:
                continue
        return {"nbar": nbar, "cv": cv(fam, model), "model": model, "draw_tries": tries}
    raise RuntimeError(f"{fam.name} theta {index}: no feasible draw in {max_tries} tries")


def sample_patterns(fam_name: str, theta: dict, cfg: Config, index: int, max_tries: int = 10_000,
                    set_: str = "bank"):
    model = dict(theta["model"])
    if fam_name == "lgcp":
        model["root_lam"] = lgcp_eigenvalues(model["sigma2"], model["s"], model["M"])
    lo, hi = cfg.n_range
    for rep in range(cfg.reps):
        rng = case_rng(cfg.root, set_, fam_name, cfg.reps * index + rep, PATTERN)
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
            rows.append({"family": fam_name, "theta": index, "rep": rep, "nbar": theta["nbar"],
                         "cv": theta["cv"], "draw_tries": theta["draw_tries"], **theta["model"],
                         "n": len(pts), "pattern_tries": tries})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, points=np.concatenate(points), sizes=np.array([len(p) for p in points]),
             manifest=np.array(json.dumps(rows)))
    tmp.replace(path)
    return path
