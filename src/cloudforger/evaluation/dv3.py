# src/cloudforger/evaluation/dv3.py
"""DV3 as the *evaluation* layer sees it: which sets exist, where a
(set, family)'s clouds/diagrams live, what regime each pattern sits in, and
which rows are train vs validation.

WHY THIS EXISTS. Before DV3 every process was one flat `clouds.pkl` that
`core.splits.train_val_test_indices(n, seed)` cut into a random 70/15/15,
and "performance" was one scalar loss on that random test slice (plus an
"adversarial" slice that docs/status/audit.md showed was a second i.i.d.
draw from the same prior, not out-of-design). DV3 replaces that with four
purpose-built products (docs/generation_procedure.tex, Table "The four data
products"):

    train   10,000 per family from the prior, with the train/val split
            ASSIGNED AT GENERATION (manifest column `split`), so every
            method and every seed trains on exactly the same rows.
    A       5,000 per family from the same prior, own streams
              -> risk integrated over the prior (the old "test loss")
    B       fixed-theta cells, 400 replicates each
              -> conditional error and bias AT a given (nbar, scale, delta)
    C       fixed-shape ladders whose amplitude walks down to CSR, plus
            exact-CSR anchors
              -> error and detection power AS A FUNCTION OF delta-tilde

A random split of a pooled dataset cannot answer any of the last two: it
averages every regime together, which is precisely what "performance across
regimes" is asking to take apart. So the split stops being random (it is
read off the manifest) and the test set stops being one slice (it is three
named sets, each analysed on its own terms).

Coordinates. Regime is (nbar, scale, delta-tilde), all dimensionless:

    nbar          expected count -- how much data, independent of geometry
    scale         the family's length coordinate in point spacings
                  (thomas s, nested s2, lgcp sp, matern2 tau)
    delta_tilde   distance from CSR in units of the 5% critical value of the
                  classical studentised L-envelope test, so delta = 0 is
                  exactly CSR and delta ~ 1 is the edge of detectability

Every one of these is already on disk, per pattern, in the manifest written
by the generator -- this module only reads and bins it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..generation.seeding import case_id as _seeding_case_id
from ..generation.store import read_csv
from ..paths import PROJECT_ROOT, DataPaths

DEFAULT_DV3_ROOT = PROJECT_ROOT / "data" / "dv3"

TRAIN_SET = "train"
PRIOR_SETS = ("train", "A")
FIXED_SETS = ("B", "C")
EVAL_SETS: tuple[str, ...] = ("A", "B", "C")
ALL_SETS: tuple[str, ...] = ("train", "A", "B", "C")

# Ordered once here so a class index never depends on directory iteration
# order: index i is class i everywhere (bundles, confusion matrices, configs).
FAMILIES: tuple[str, ...] = ("poisson", "thomas", "nested", "matern2", "lgcp")

# The family's length coordinate, in point spacings (docs/generation_procedure.tex
# Table "Models and prior"). CSR has no scale.
SCALE_COORDINATE: dict[str, str | None] = {
    "poisson": None,
    "thomas": "s",
    "nested": "s2",
    "matern2": "tau",
    "lgcp": "sp",
}

# The coordinate the ladders/cells move to change the distance from CSR.
AMPLITUDE_COORDINATE: dict[str, str | None] = {
    "poisson": None,
    "thomas": "mu",
    "nested": "mu2",
    "matern2": "tau",
    "lgcp": "sigma2",
}

# ---------------------------------------------------------------------------
# Regime bands. Fixed here, once, so every table/figure in the project cuts
# the axes at the same places and rows stay comparable between methods.
# delta edges are the B grid's own targets {0.5, 1, 2, 4, 8}; nbar edges
# split the LogU[100, 800] prior into the three octaves the B/C grids sit at
# (125 / 250 / 500).
# ---------------------------------------------------------------------------

DELTA_EDGES: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, np.inf)
NBAR_EDGES: tuple[float, ...] = (100.0, 200.0, 400.0, 800.0 + 1e-9)
SCALE_EDGES: tuple[float, ...] = (0.0, 0.15, 0.35, 0.7, 1.5, np.inf)


def band_labels(edges: Sequence[float], fmt: str = "{:g}") -> list[str]:
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        lo_s = fmt.format(lo)
        hi_s = "inf" if not np.isfinite(hi) else fmt.format(hi)
        out.append(f"[{lo_s},{hi_s})")
    return out


def band_index(values: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """Half-open bin index per value; -1 for NaN or out-of-range values, so a
    caller can tell "no band" from "first band" instead of silently pooling
    them together."""
    values = np.asarray(values, dtype=float)
    idx = np.digitize(values, np.asarray(edges[1:-1], dtype=float), right=False)
    bad = ~np.isfinite(values) | (values < edges[0]) | (values >= edges[-1])
    idx = idx.astype(int)
    idx[bad] = -1
    return idx


# ---------------------------------------------------------------------------
# Paths and identity
# ---------------------------------------------------------------------------

def set_root(set_: str, root: Path | str = DEFAULT_DV3_ROOT) -> Path:
    return Path(root) / set_


def data_paths(set_: str, family: str, root: Path | str = DEFAULT_DV3_ROOT) -> DataPaths:
    """The DataPaths every existing stage (featurize/diagrams/train) already
    speaks, pointed at one DV3 (set, family): clouds.pkl and the per-filtration
    diagrams.pkl land exactly where scripts/processing/dv3_diagrams.py writes
    them."""
    return DataPaths(family, root=set_root(set_, root))


def manifest_path(set_: str, family: str, root: Path | str = DEFAULT_DV3_ROOT) -> Path:
    return set_root(set_, root) / family / "manifest.csv"


def case_id(set_: str, family: str, index: int) -> str:
    """The generator's own pattern identity (generation/seeding.py). Unique
    across the whole dataset, which the per-(set, family) `index` -- what the
    legacy cloud record calls `seed` -- is NOT."""
    return _seeding_case_id(set_, family, int(index))


def case_ids(set_: str, family: str, indices: Iterable[int]) -> np.ndarray:
    return np.array([case_id(set_, family, i) for i in indices], dtype=object)


# ---------------------------------------------------------------------------
# Regimes
# ---------------------------------------------------------------------------

REGIME_FIELDS = (
    "case_id", "set", "family", "index", "cell_id", "level_id", "rep",
    "nbar", "delta_tilde", "tau_K", "scale", "amplitude", "n", "split",
)


@dataclass(frozen=True)
class Regimes:
    """Per-pattern regime coordinates for one or more (set, family) groups,
    addressable by case_id. Columns are plain numpy arrays, aligned."""

    columns: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.columns["case_id"])

    def __getitem__(self, key: str) -> np.ndarray:
        return self.columns[key]

    @property
    def case_id(self) -> np.ndarray:
        return self.columns["case_id"]

    def index_of(self) -> dict[str, int]:
        return {str(c): i for i, c in enumerate(self.columns["case_id"])}

    def select(self, wanted_case_ids: Sequence[str]) -> "Regimes":
        """Reorder/restrict to `wanted_case_ids` (the order a prediction
        bundle stored its rows in). Raises on an unknown id rather than
        silently dropping a row, so a join that lost patterns is loud."""
        pos = self.index_of()
        missing = [c for c in wanted_case_ids if str(c) not in pos]
        if missing:
            raise KeyError(
                f"{len(missing)} case_id(s) absent from the DV3 manifests "
                f"(first few: {missing[:5]}) -- predictions and manifest disagree."
            )
        rows = np.array([pos[str(c)] for c in wanted_case_ids], dtype=np.int64)
        return Regimes({k: v[rows] for k, v in self.columns.items()})

    def mask(self, **equals: Any) -> np.ndarray:
        out = np.ones(len(self), dtype=bool)
        for key, value in equals.items():
            out &= self.columns[key] == value
        return out

    # -- derived bands --------------------------------------------------

    def delta_band(self) -> np.ndarray:
        return band_index(self.columns["delta_tilde"], DELTA_EDGES)

    def nbar_band(self) -> np.ndarray:
        return band_index(self.columns["nbar"], NBAR_EDGES)

    def scale_band(self) -> np.ndarray:
        return band_index(self.columns["scale"], SCALE_EDGES)


def _row_regime(row: dict[str, Any]) -> dict[str, Any]:
    family = row["family"]
    scale_key = SCALE_COORDINATE.get(family)
    amp_key = AMPLITUDE_COORDINATE.get(family)
    return {
        "case_id": row["case_id"],
        "set": row["set"],
        "family": family,
        "index": int(row["index"]),
        "cell_id": -1 if row.get("cell_id") is None else int(row["cell_id"]),
        "level_id": -1 if row.get("level_id") is None else int(row["level_id"]),
        "rep": -1 if row.get("rep") is None else int(row["rep"]),
        "nbar": float(row["nbar"]),
        # CSR is delta = 0 by construction; the generator leaves the column
        # empty for poisson rows rather than writing a 0 it did not compute.
        "delta_tilde": 0.0 if row.get("delta_tilde") is None else float(row["delta_tilde"]),
        "tau_K": np.nan if row.get("tau_K") is None else float(row["tau_K"]),
        "scale": np.nan if (scale_key is None or row.get(scale_key) is None) else float(row[scale_key]),
        "amplitude": np.nan if (amp_key is None or row.get(amp_key) is None) else float(row[amp_key]),
        "n": -1 if row.get("n") is None else int(row["n"]),
        "split": row.get("split") or "",
    }


def _stack(records: list[dict[str, Any]]) -> Regimes:
    if not records:
        return Regimes({f: np.array([]) for f in REGIME_FIELDS})
    cols: dict[str, np.ndarray] = {}
    for field in REGIME_FIELDS:
        values = [r[field] for r in records]
        if field in ("case_id", "set", "family", "split"):
            cols[field] = np.array(values, dtype=object)
        elif field in ("index", "cell_id", "level_id", "rep", "n"):
            cols[field] = np.array(values, dtype=np.int64)
        else:
            cols[field] = np.array(values, dtype=float)
    return Regimes(cols)


@lru_cache(maxsize=64)
def _cached_manifest(path_str: str) -> tuple[dict[str, Any], ...]:
    return tuple(read_csv(Path(path_str)))


def load_regimes(
    set_: str,
    families: Sequence[str] | str,
    root: Path | str = DEFAULT_DV3_ROOT,
) -> Regimes:
    """Regime coordinates for every pattern of `families` in one DV3 set,
    read straight from the generator's manifests (no point data touched)."""
    if isinstance(families, str):
        families = [families]
    records: list[dict[str, Any]] = []
    for family in families:
        path = manifest_path(set_, family, root)
        if not path.exists():
            raise FileNotFoundError(
                f"no DV3 manifest at {path} -- generate it with "
                f"`python scripts/generation/dv3.py all`."
            )
        records += [_row_regime(row) for row in _cached_manifest(str(path))]
    return _stack(records)


def load_regimes_multi(
    sets: Sequence[str],
    families: Sequence[str] | str,
    root: Path | str = DEFAULT_DV3_ROOT,
) -> Regimes:
    parts = [load_regimes(s, families, root) for s in sets]
    return Regimes({
        field: np.concatenate([p.columns[field] for p in parts]) for field in REGIME_FIELDS
    })


# ---------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------

def split_indices_from_records(
    records: Sequence[dict[str, Any]],
    reshuffle_seed: int | None = None,
    val_fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """(train_idx, val_idx) from the per-cloud `split` field DV3 assigned at
    generation -- NOT a fresh random cut.

    Fixing the split at generation is the point: every method and every seed
    then fits on identical rows, so a seed-to-seed difference is training
    randomness alone and a method-to-method difference is not confounded
    with a luckier draw. Test rows do not appear here at all: under DV3 the
    test sets are separate products (A, B, C), never a slice of the training
    pool.

    reshuffle_seed re-draws train/val *within the training pool only* (the
    test sets are untouched either way) for a "does the fixed split matter?"
    variance check; leave it None for the pre-registered split.
    """
    splits = np.array([str(r.get("split") or "") for r in records], dtype=object)
    if reshuffle_seed is not None:
        n = len(records)
        perm = np.random.default_rng(int(reshuffle_seed)).permutation(n)
        n_val = int(round(val_fraction * n))
        return np.sort(perm[n_val:]), np.sort(perm[:n_val])

    train_idx = np.flatnonzero(splits == "train")
    val_idx = np.flatnonzero(splits == "val")
    unknown = sorted(set(map(str, splits)) - {"train", "val"})
    if unknown:
        raise ValueError(
            f"the training pool carries split value(s) {unknown}; DV3 assigns only "
            "'train' and 'val' there (B/C rows are 'test' and must not be trained on)."
        )
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(
            f"degenerate DV3 split: {len(train_idx)} train / {len(val_idx)} val rows."
        )
    return train_idx, val_idx


def split_indices(
    set_: str,
    families: Sequence[str] | str,
    root: Path | str = DEFAULT_DV3_ROOT,
    reshuffle_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    regimes = load_regimes(set_, families, root)
    records = [{"split": s} for s in regimes["split"]]
    return split_indices_from_records(records, reshuffle_seed=reshuffle_seed)
