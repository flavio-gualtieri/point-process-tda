# src/cloudforger/evaluation/testset.py
"""The stratified test set: one table, balanced over (family x distance from CSR).

WHY THIS EXISTS. The generator emits three evaluation products -- a draw from
the prior, a fixed-theta grid, and ladders walking down to CSR. Reporting them
as three separately-named sets made the headline number depend on which one a
reader looked at, and the prior draw's aggregate is an artifact of a
log-uniform prior nobody defends on scientific grounds: roughly half its mass
sits below delta = 1, i.e. where the classical 5% test provably cannot see the
departure at all.

This module replaces those three headline numbers with ONE test set, assembled
by SELECTION from the patterns that already exist. Nothing is generated and
nothing is re-featurized: a row here is a case_id that already has clouds,
diagrams and predictions on disk. The three source sets become a provenance
column, not three answers.

THE DESIGN. A balanced factorial, `per_cell` cases in every cell of

    family  x  stratum          5 families x 4 strata = 20 cells

with the stratum cut on delta-tilde, the generator's distance from CSR in
units of the 5% critical value of the classical studentised L-envelope test
(generation/nulls.py). So the strata are

    below      delta <  1     the classical test does not reject CSR
    weak       1 <= delta < 2  just past the detection threshold
    moderate   2 <= delta < 4
    strong     4 <= delta      the regime where the departure is unambiguous

Equal cells mean the aggregate over the table is an explicit, stated weighting
rather than a property of the prior, and every per-stratum number is read off
the same table by `groupby(stratum)`.

CSR IS A CLASS, NOT A STRATUM. Poisson has delta = 0 by construction, so on the
delta axis every CSR pattern would land in `below`. A classification stratum
with no CSR cases is a 4-way problem whose chance level is 25%, which cannot be
compared with a stratum that has them. So the CSR quota is held constant across
strata: each stratum gets `per_cell` CSR patterns, drawn disjointly from the
CSR pool, and the class prior stays uniform at 1/5 everywhere. A CSR row's
`stratum` therefore names the band it is PAIRED WITH, not a delta band it
falls in -- its `delta_tilde` is 0.0 and its `is_null` flag is set. The
per-family parameter tasks never see these rows.

WHAT IS NOT BALANCED. nbar cannot also be balanced inside a cell, and the
allocator does not pretend otherwise. Reaching a given delta at fixed shape
needs points, so delta and nbar are structurally entangled: the pool holds
only 142 Matern II patterns at nbar ~ 250 in the `weak` stratum, and 120 at
nbar ~ 125 in `strong`. Each cell is therefore filled as evenly across the
three nbar octaves as availability allows, and the realised composition is
recorded per row (`nbar_band`) and summarised by `composition()` so that any
nbar imbalance is visible rather than assumed away.

REPLICATES ARE NOT INDEPENDENT DRAWS. The fixed-theta grid and the ladders
carry hundreds of replicates at ONE theta, while the prior draw has a fresh
theta per row. Pooling them puts repeated thetas in the table, so a naive
standard error over rows understates uncertainty. Every row therefore carries
`theta_id`, constant within a replicate group and unique per prior-draw row;
cluster on it (`scripts/evaluate_testset.py` does) and the error bars are
honest.

The selection is deterministic: a fixed seed, case ids sorted before sampling,
and the largest-remainder allocation below. Rebuilding the table on any machine
reproduces it byte for byte.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .dv3 import (
    DEFAULT_DV3_ROOT,
    EVAL_SETS,
    FAMILIES,
    NBAR_EDGES,
    Regimes,
    band_index,
    band_labels,
    load_regimes,
    manifest_path,
)

# The stratum cut. Anchored at delta = 1 because that is not a round number
# chosen after the fact: `nulls.departure` normalises by c95, so delta <= 1 is
# exactly "not rejected at 5% by the Monte Carlo CSR test on L(r) - r".
STRATUM_EDGES: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, np.inf)
STRATUM_NAMES: tuple[str, ...] = ("below", "weak", "moderate", "strong")

# The null family. Held at a constant share of every stratum; see the module
# docstring.
NULL_FAMILY = "poisson"

# How the generator's three source products are named in this table. The
# letters they carry on disk say nothing about what they are; these do.
SOURCE_NAMES: dict[str, str] = {
    "A": "prior",    # one fresh theta per pattern, drawn from the prior
    "B": "grid",     # fixed-theta cells, hundreds of replicates each
    "C": "ladder",   # fixed shape, amplitude walking down to CSR
}


DEFAULT_PER_CELL = 2500
DEFAULT_SEED = 20260916

COLUMNS: tuple[str, ...] = (
    "case_id", "family", "stratum", "nbar_band", "theta_id", "is_null",
    "source_set", "cell_id", "level_id", "index", "nbar", "n",
    "delta_tilde", "scale", "amplitude",
)

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "configs" / "testset.csv"


def stratum_index(delta: np.ndarray) -> np.ndarray:
    """Index into STRATUM_NAMES for each delta-tilde."""
    return band_index(np.asarray(delta, dtype=float), STRATUM_EDGES)


def nbar_band_labels() -> list[str]:
    return band_labels(NBAR_EDGES)


def theta_id(source_set: str, family: str, cell_id: int, level_id: int, index: int) -> str:
    """Identifier of the parameter vector behind a row, constant within a
    replicate group. The fixed-theta sets address theta by (cell, level); the
    prior draw has one theta per row, so its own index is the identifier."""
    if cell_id >= 0 or level_id >= 0:
        return f"{source_set}:{family}:c{cell_id}:l{level_id}"
    return f"{source_set}:{family}:i{index}"


def available_families(set_: str, root: Path | str = DEFAULT_DV3_ROOT) -> list[str]:
    """Which families the generator actually wrote for `set_`. The fixed-theta
    grid has no CSR cells, so asking for every family there is a FileNotFound."""
    return [f for f in FAMILIES if manifest_path(set_, f, root).exists()]


def pool(
    sets: Sequence[str] = EVAL_SETS,
    root: Path | str = DEFAULT_DV3_ROOT,
) -> Regimes:
    """Every evaluation pattern that exists, as one addressable table. Reads
    the generator's manifests only -- no point data, no diagrams."""
    parts = [load_regimes(s, available_families(s, root), root) for s in sets]
    if not parts:
        raise ValueError("no sets to pool")
    fields = parts[0].columns.keys()
    return Regimes({f: np.concatenate([p.columns[f] for p in parts]) for f in fields})


def allocate(available: Sequence[int], total: int) -> list[int]:
    """Split `total` over bins of the given capacities, as evenly as the
    capacities allow: give every bin still under quota an equal share, spill
    what a full bin cannot take onto the others, repeat. Returns a short
    allocation (sum < total) only when the capacities genuinely cannot meet it,
    which `build` reports rather than silently accepting."""
    capacity = [int(c) for c in available]
    take = [0] * len(capacity)
    remaining = int(total)
    active = {i for i, c in enumerate(capacity) if c > 0}
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        moved = False
        for i in sorted(active):
            if remaining <= 0:
                break
            give = min(share, capacity[i] - take[i], remaining)
            if give > 0:
                take[i] += give
                remaining -= give
                moved = True
            if take[i] >= capacity[i]:
                active.discard(i)
        if not moved:
            break
    return take


@dataclass(frozen=True)
class TestSet:
    """The assembled table. `rows` are dicts keyed by COLUMNS."""

    rows: tuple[dict[str, Any], ...]
    per_cell: int
    seed: int

    def __len__(self) -> int:
        return len(self.rows)

    def column(self, key: str) -> np.ndarray:
        return np.array([r[key] for r in self.rows], dtype=object if isinstance(
            self.rows[0][key], str) else None)

    @property
    def case_ids(self) -> np.ndarray:
        return np.array([r["case_id"] for r in self.rows], dtype=object)

    def counts(self) -> dict[str, dict[str, int]]:
        """cells[family][stratum] -> n, the balance check."""
        out: dict[str, dict[str, int]] = {}
        for r in self.rows:
            out.setdefault(r["family"], {}).setdefault(r["stratum"], 0)
            out[r["family"]][r["stratum"]] += 1
        return out

    def composition(self) -> dict[str, dict[str, dict[str, int]]]:
        """cells[family][stratum][nbar_band] -> n: what the allocator could
        not make even, laid out so it can be read off rather than assumed."""
        out: dict[str, dict[str, dict[str, int]]] = {}
        for r in self.rows:
            cell = out.setdefault(r["family"], {}).setdefault(r["stratum"], {})
            cell[r["nbar_band"]] = cell.get(r["nbar_band"], 0) + 1
        return out

    def provenance(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.rows:
            out[r["source_set"]] = out.get(r["source_set"], 0) + 1
        return out


def build(
    per_cell: int = DEFAULT_PER_CELL,
    seed: int = DEFAULT_SEED,
    sets: Sequence[str] = EVAL_SETS,
    root: Path | str = DEFAULT_DV3_ROOT,
) -> tuple[TestSet, list[str]]:
    """Select the balanced table out of the existing patterns.

    Returns the table and a list of human-readable warnings -- one per cell the
    pool could not fill to `per_cell`. A short cell is reported, never padded
    and never silently dropped: the count travels with the row group so a
    thin cell shows up as a wider error bar instead of a missing one."""
    reg = pool(sets, root)
    families = np.asarray(reg["family"], dtype=object)
    delta = np.asarray(reg["delta_tilde"], dtype=float)
    nbar = np.asarray(reg["nbar"], dtype=float)
    strat = stratum_index(delta)
    nband = band_index(nbar, NBAR_EDGES)
    nband_names = nbar_band_labels()

    rng = np.random.default_rng(seed)
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []

    def emit(positions: Iterable[int], stratum_name: str) -> None:
        for p in positions:
            p = int(p)
            cid = int(reg["cell_id"][p])
            lid = int(reg["level_id"][p])
            idx = int(reg["index"][p])
            src = SOURCE_NAMES.get(str(reg["set"][p]), str(reg["set"][p]))
            fam = str(reg["family"][p])
            rows.append({
                "case_id": str(reg["case_id"][p]),
                "family": fam,
                "stratum": stratum_name,
                "nbar_band": nband_names[int(nband[p])],
                "theta_id": theta_id(src, fam, cid, lid, idx),
                "is_null": int(fam == NULL_FAMILY),
                "source_set": src,
                "cell_id": cid,
                "level_id": lid,
                "index": idx,
                "nbar": float(reg["nbar"][p]),
                "n": int(reg["n"][p]),
                "delta_tilde": float(delta[p]),
                "scale": float(reg["scale"][p]),
                "amplitude": float(reg["amplitude"][p]),
            })

    def draw(candidates: np.ndarray, stratum_name: str, label: str) -> None:
        """Fill one cell from `candidates`, spreading over nbar octaves as
        evenly as their capacities allow."""
        by_band = [candidates[nband[candidates] == b] for b in range(len(nband_names))]
        quota = allocate([len(c) for c in by_band], per_cell)
        got = sum(quota)
        if got < per_cell:
            warnings.append(
                f"{label}: pool holds {got} of {per_cell} requested "
                f"(by nbar octave: {dict(zip(nband_names, [len(c) for c in by_band]))})"
            )
        for band_rows, take in zip(by_band, quota):
            if take <= 0:
                continue
            # Sort before sampling so the draw depends on the seed alone,
            # never on manifest row order.
            ordered = np.sort(band_rows)
            emit(rng.choice(ordered, size=take, replace=False), stratum_name)

    # -- the four structured families: cut on their own delta ---------------
    for fam in FAMILIES:
        if fam == NULL_FAMILY:
            continue
        for s, stratum_name in enumerate(STRATUM_NAMES):
            cand = np.flatnonzero((families == fam) & (strat == s))
            draw(cand, stratum_name, f"{fam}/{stratum_name}")

    # -- CSR: one constant quota per stratum, drawn disjointly --------------
    null_pool = np.flatnonzero(families == NULL_FAMILY)
    null_order = rng.permutation(np.sort(null_pool))
    cursor = 0
    for stratum_name in STRATUM_NAMES:
        chunk = null_order[cursor:cursor + per_cell]
        cursor += len(chunk)
        if len(chunk) < per_cell:
            warnings.append(
                f"{NULL_FAMILY}/{stratum_name}: CSR pool exhausted, "
                f"{len(chunk)} of {per_cell} requested"
            )
        draw(chunk, stratum_name, f"{NULL_FAMILY}/{stratum_name}")

    rows.sort(key=lambda r: (r["family"], r["stratum"], r["case_id"]))
    return TestSet(tuple(rows), per_cell=per_cell, seed=seed), warnings


def write_csv(ts: TestSet, path: Path | str = DEFAULT_PATH) -> Path:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COLUMNS))
        writer.writeheader()
        for row in ts.rows:
            writer.writerow(row)
    return path


def load(path: Path | str = DEFAULT_PATH) -> TestSet:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no test set at {path} -- build it with `python scripts/build_testset.py`."
        )
    import csv

    rows = []
    with path.open(newline="") as fh:
        raw_rows = list(csv.DictReader(fh))
    for raw in raw_rows:
        rows.append({
            "case_id": str(raw["case_id"]),
            "family": str(raw["family"]),
            "stratum": str(raw["stratum"]),
            "nbar_band": str(raw["nbar_band"]),
            "theta_id": str(raw["theta_id"]),
            "is_null": int(raw["is_null"]),
            "source_set": str(raw["source_set"]),
            "cell_id": int(raw["cell_id"]),
            "level_id": int(raw["level_id"]),
            "index": int(raw["index"]),
            "nbar": float(raw["nbar"]),
            "n": int(raw["n"]),
            "delta_tilde": float(raw["delta_tilde"]),
            "scale": float("nan") if raw["scale"] in (None, "") else float(raw["scale"]),
            "amplitude": float("nan") if raw["amplitude"] in (None, "") else float(raw["amplitude"]),
        })
    per_cell = max(
        sum(1 for r in rows if r["family"] == f and r["stratum"] == s)
        for f in {r["family"] for r in rows}
        for s in STRATUM_NAMES
    )
    return TestSet(tuple(rows), per_cell=per_cell, seed=DEFAULT_SEED)
