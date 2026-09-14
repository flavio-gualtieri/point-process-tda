# src/cloudforger/generation/store.py
"""On-disk layout of DV3 (generation.tex, "Storage"), and the low-level I/O.

    data/dv3/
      plan.csv, plan.json                 every case, with theta; checksums
      shards/<set>/<family>/shard_KKKK.{npz,csv,done.json}
      <set>/<family>/points.npz           all coordinates (float64) + int64 offsets
      <set>/<family>/manifest.csv         one row per case
      <set>/<family>/clouds.pkl           compatibility copy for featurize/diagrams
      dataset_card.json                   spec hash, git commit, versions, checksums

Every file is written to a temporary name and renamed into place, so a
killed job never leaves a half-written file under a valid name.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# --- manifest schema (Table "Manifest columns"); fixed for every family ------

IDENTITY = ["case_id", "dv", "set", "family", "index", "cell_id", "level_id", "rep",
            "spawn_key", "r_seed", "split", "shard"]
DESIGN = ["nbar", "mu", "s", "mu1", "mu2", "s2", "rho", "tau", "gamma", "sigma2", "sp"]
MODEL = ["kappa", "sigma", "sigma1", "lam_p", "R", "beta", "mu_log", "s_abs"]
REGIME = ["delta_tilde", "tau_K", "prior_tries"]
SAMPLER = ["n", "sampler", "buffer", "n_parents", "grid_M", "pad_P", "min_eig",
           "cftp_start", "mh_steps", "burnin", "wall_s", "sha1"]

PLAN_COLUMNS = IDENTITY + DESIGN + MODEL + REGIME
MANIFEST_COLUMNS = PLAN_COLUMNS + SAMPLER

_INT = {"dv", "index", "cell_id", "level_id", "rep", "r_seed", "shard", "prior_tries",
        "n", "n_parents", "grid_M", "pad_P", "cftp_start", "mh_steps", "burnin"}
_STR = {"case_id", "set", "family", "spawn_key", "split", "sampler", "sha1"}


def fmt(value: Any) -> str:
    # repr() of a float is its shortest exact round-trip form, so a theta read
    # back from CSV is bit-identical to the one that was drawn.
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("booleans have no manifest column")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    return str(value)


def parse(column: str, text: str) -> Any:
    if text == "":
        return None
    if column in _STR:
        return text
    if column in _INT:
        return int(text)
    return float(text)


# --- atomic writes and checksums ---------------------------------------------

def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_bytes(path, (json.dumps(payload, indent=2, sort_keys=False) + "\n").encode())


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def csv_bytes(columns: list[str], rows: Iterable[dict[str, Any]]) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        unknown = set(row) - set(columns)
        if unknown:
            raise KeyError(f"row has columns outside the schema: {sorted(unknown)}")
        writer.writerow([fmt(row.get(c)) for c in columns])
    return buf.getvalue().encode()


def read_csv(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="") as f:
        return [{k: parse(k, v) for k, v in row.items()} for row in csv.DictReader(f)]


# --- ragged point arrays -------------------------------------------------------

def points_npz_bytes(clouds: list[np.ndarray], index: list[int]) -> bytes:
    offsets = np.zeros(len(clouds) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(c) for c in clouds])
    coords = np.concatenate(clouds, axis=0) if clouds else np.empty((0, 2))
    buf = io.BytesIO()
    np.savez(buf, coords=coords.astype(np.float64), offsets=offsets, index=np.asarray(index, dtype=np.int64))
    return buf.getvalue()


def load_points(path: Path) -> tuple[list[np.ndarray], np.ndarray]:
    with np.load(path) as z:
        coords, offsets, index = z["coords"], z["offsets"], z["index"]
    return [coords[offsets[i]:offsets[i + 1]] for i in range(len(index))], index


# --- layout ---------------------------------------------------------------------

class DV3Paths:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    @property
    def plan_csv(self) -> Path:
        return self.root / "plan.csv"

    @property
    def plan_meta(self) -> Path:
        return self.root / "plan.json"

    @property
    def card(self) -> Path:
        return self.root / "dataset_card.json"

    def shard_stem(self, set_: str, family: str, shard: int) -> Path:
        return self.root / "shards" / set_ / family / f"shard_{shard:04d}"

    def shard_files(self, set_: str, family: str, shard: int) -> tuple[Path, Path, Path]:
        stem = self.shard_stem(set_, family, shard)
        return stem.with_suffix(".npz"), stem.with_suffix(".csv"), stem.with_name(stem.name + ".done.json")

    def out_dir(self, set_: str, family: str) -> Path:
        return self.root / set_ / family

    def points(self, set_: str, family: str) -> Path:
        return self.out_dir(set_, family) / "points.npz"

    def manifest(self, set_: str, family: str) -> Path:
        return self.out_dir(set_, family) / "manifest.csv"

    def clouds(self, set_: str, family: str) -> Path:
        # Same path DataPaths(family, root=<root>/<set>).clouds() resolves to.
        return self.out_dir(set_, family) / "clouds.pkl"
