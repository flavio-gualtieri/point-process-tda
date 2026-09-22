"""Diagrams for every simulated pattern, one shard of patterns at a time.

Input   data/bank/<family>/{points.npz, manifest.csv}
Shard   data/featurization/shards/<family>/<tag>/shard_<i>.npz
Merged  data/featurization/<family>/<tag>/diagrams.npz (shards deleted), rows in manifest.csv order:
            case_id          (P,)
            h<d>             (m, 2) finite pairs of every pattern, concatenated
            h<d>_offsets     (P + 1,) pattern p's pairs are h<d>[offsets[p]:offsets[p + 1]]
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .filtrations import FILTRATIONS, tag

from ..simulation.bank import DATA as BANK

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG = PROJECT_ROOT / "configs" / "featurization" / "config.yaml"
DATA = PROJECT_ROOT / "data" / "featurization"


@dataclass(frozen=True)
class Config:
    maxdim: int
    shard_size: int
    filtrations: tuple[dict, ...]

    @classmethod
    def load(cls, path: Path = CONFIG) -> Config:
        c = yaml.safe_load(path.read_text())
        return cls(int(c["maxdim"]), int(c["shard_size"]), tuple(c["filtrations"]))

    def spec(self, tag_: str) -> dict:
        return next(s for s in self.filtrations if tag(s) == tag_)


def families() -> list[str]:
    return sorted(p.parent.name for p in BANK.glob("*/points.npz"))


def n_shards(family: str, cfg: Config) -> int:
    return math.ceil(len(pd.read_csv(BANK / family / "manifest.csv")) / cfg.shard_size)


def _pack(diagrams: list[dict[int, np.ndarray]], maxdim: int) -> dict[str, np.ndarray]:
    out = {}
    for d in range(maxdim + 1):
        out[f"h{d}"] = np.concatenate([g[d] for g in diagrams]).reshape(-1, 2)
        out[f"h{d}_offsets"] = np.concatenate([[0], np.cumsum([len(g[d]) for g in diagrams])])
    return out


def run_shard(family: str, tag_: str, shard: int, cfg: Config) -> Path:
    path = DATA / "shards" / family / tag_ / f"shard_{shard:04d}.npz"
    if path.exists() or (DATA / family / tag_ / "diagrams.npz").exists():
        return path
    spec = {k: v for k, v in cfg.spec(tag_).items() if k != "name"}
    compute = FILTRATIONS[cfg.spec(tag_)["name"]]
    z = np.load(BANK / family / "points.npz")
    points, offsets = z["points"], z["offsets"]
    lo, hi = shard * cfg.shard_size, min((shard + 1) * cfg.shard_size, len(offsets) - 1)
    diagrams = [compute(points[offsets[i]:offsets[i + 1]], maxdim=cfg.maxdim, **spec) for i in range(lo, hi)]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, first=lo, spec=np.array(json.dumps(cfg.spec(tag_))), **_pack(diagrams, cfg.maxdim))
    tmp.replace(path)
    return path


def merge(family: str, tag_: str, cfg: Config) -> Path:
    path, shards = DATA / family / tag_ / "diagrams.npz", DATA / "shards" / family / tag_
    if path.exists() and not shards.exists():
        return path
    case_id = pd.read_csv(BANK / family / "manifest.csv").case_id.to_numpy(str)
    paths = sorted(shards.glob("shard_*.npz"))
    if len(paths) != n_shards(family, cfg):
        raise SystemExit(f"{family}/{tag_}: {len(paths)} of {n_shards(family, cfg)} shards")
    parts = [np.load(p) for p in paths]
    if [int(z["first"]) for z in parts] != list(range(0, len(case_id), cfg.shard_size)):
        raise SystemExit(f"{family}/{tag_}: shards inconsistent")
    out = {"case_id": case_id}
    for d in range(cfg.maxdim + 1):
        out[f"h{d}"] = np.concatenate([z[f"h{d}"] for z in parts])
        sizes = np.concatenate([np.diff(z[f"h{d}_offsets"]) for z in parts])
        out[f"h{d}_offsets"] = np.concatenate([[0], np.cumsum(sizes)])
    if len(out["h0_offsets"]) != len(case_id) + 1:
        raise SystemExit(f"{family}/{tag_}: {len(out['h0_offsets']) - 1} diagrams for {len(case_id)} patterns")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, spec=np.array(json.dumps(cfg.spec(tag_))), **out)
    tmp.replace(path)
    shutil.rmtree(shards)
    for empty in (shards.parent, shards.parent.parent):
        if empty.exists() and not any(empty.iterdir()):
            empty.rmdir()
    return path
