# src/cloudforger/generation/spec.py
"""Load and validate configs/generation/dv3.yaml."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .seeding import FAMILY_ID, SET_ID, check_root

PRIOR_SETS = ("train", "A")  # sets whose theta is drawn from the prior


@dataclass(frozen=True)
class Spec:
    path: Path
    sha256: str
    root: int
    nbar_low: float
    nbar_high: float
    families: dict[str, dict[str, Any]]
    sets: dict[str, dict[str, int]]
    shard_size: int

    def set_size(self, set_: str) -> int:
        return int(self.sets[set_]["size"])

    def n_val(self, set_: str) -> int:
        return int(self.sets[set_].get("n_val", 0))


def load_spec(path: Path | str) -> Spec:
    path = Path(path)
    raw = path.read_bytes()
    cfg = yaml.safe_load(raw)

    if cfg.get("dataset") != "DV3":
        raise ValueError(f"{path}: expected dataset: DV3, got {cfg.get('dataset')!r}")
    root = check_root(cfg["root"])

    nbar = cfg["nbar"]
    lo, hi = float(nbar["low"]), float(nbar["high"])
    if not 0 < lo < hi:
        raise ValueError(f"{path}: need 0 < nbar.low < nbar.high")

    families = {name: dict(block or {}) for name, block in cfg["families"].items()}
    for name in families:
        if name not in FAMILY_ID:
            raise ValueError(f"{path}: unknown family {name!r}")

    sets = {name: dict(block) for name, block in cfg["sets"].items()}
    for name, block in sets.items():
        if name not in PRIOR_SETS:
            raise ValueError(f"{path}: set {name!r} is not prior-drawn; B and C come from dv3_cells.csv")
        if name not in SET_ID:
            raise ValueError(f"{path}: unknown set {name!r}")
        if not 0 <= int(block.get("n_val", 0)) < int(block["size"]):
            raise ValueError(f"{path}: set {name!r} needs 0 <= n_val < size")

    shard_size = int(cfg["shard_size"])
    if shard_size < 1:
        raise ValueError(f"{path}: shard_size must be >= 1")

    return Spec(
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        root=root,
        nbar_low=lo,
        nbar_high=hi,
        families=families,
        sets=sets,
        shard_size=shard_size,
    )
