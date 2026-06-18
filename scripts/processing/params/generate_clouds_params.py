#!/usr/bin/env python3
"""Generate a labelled series of point clouds for a parameter-estimation pipeline.

Currently supports the Thomas cluster process in 2D. Every cloud produced is a
``cloudforger.core.cloud.PointCloud``, which already records the generator name,
the exact parameters it was sampled from, and its seed -- so each cloud in the
saved collection is self-labelling; no separate label array is needed.

Two design modes:
  * ``random`` (default): draw ``--n-samples`` distinct parameter vectors from
    continuous ranges. Best for estimation -- every cloud has its own label and
    the parameter space is densely covered.
  * ``grid``: sweep the discrete grid, ``--reps`` realisations per combination.

Output: a pickle file containing a ``list[dict]`` (default). Each dict is
labelled by its generating parameters under the ``"params"`` key:

    {"points": (N, D) array, "params": {...}, "process": "thomas",
     "seed": int, "n_points": int, "dimension": int, "region": {...}}

These dicts hold only numpy arrays and built-ins, so the pickle reloads with
numpy alone -- no need to import ``cloudforger``. Pass ``--format object`` to
pickle ``PointCloud`` instances instead (which then requires the package).

Examples
--------
    python scripts/processing/generate_clouds_params.py                 # 500 distinct
    python scripts/processing/generate_clouds_params.py --n-samples 100 --reps 5  # 100 settings x 5 seeds
    python scripts/processing/generate_clouds_params.py --mode grid --reps 20
"""

from __future__ import annotations

import argparse
import itertools
import pickle
import sys
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

# --- make the cloudforger package importable when run as a plain script ------
# scripts/processing/<this file>  ->  parents[2] == project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SRC = PROJECT_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cloudforger.core.base import PointProcess          # noqa: E402
from cloudforger.core.cloud import PointCloud           # noqa: E402
from cloudforger.core.region import Box, Region         # noqa: E402
from cloudforger.processes.thomas import ThomasProcess  # noqa: E402

ProcessBuilder = Callable[[dict[str, Any]], PointProcess]

# --- process registry --------------------------------------------------------
# Each process registers:
#   build  : param dict -> PointProcess
#   grid   : discrete values per parameter (used by --mode grid)
#   ranges : (low, high, scale) per parameter, scale in {"log", "linear"}
#            (used by --mode random)
# To add a process later, register an entry here; the pipeline is otherwise
# process-agnostic.
PROCESS_REGISTRY: dict[str, dict[str, Any]] = {
    "thomas": {
        "build": lambda p: ThomasProcess(**p),
        "grid": {
            "parent_intensity": [20.0, 50.0, 100.0],
            "mean_offspring": [5.0, 15.0, 30.0],
            "cluster_scale": [0.01, 0.03, 0.07],
        },
        "ranges": {
            # intensity & scale span orders of magnitude -> log-uniform
            "parent_intensity": (10.0, 200.0, "log"),
            "mean_offspring": (2.0, 40.0, "log"),
            "cluster_scale": (0.005, 0.1, "log"),
        },
    },
}


def iter_param_grid(grid: dict[str, list[Any]]) -> Iterator[dict[str, Any]]:
    """Yield each combination of parameters in `grid` as a dict."""
    keys = list(grid)
    for combo in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, combo))


def sample_param_vector(
    ranges: dict[str, tuple[float, float, str]], rng: np.random.Generator
) -> dict[str, float]:
    """Draw one parameter vector from the given ranges."""
    params: dict[str, float] = {}
    for key, (low, high, scale) in ranges.items():
        if scale == "log":
            value = np.exp(rng.uniform(np.log(low), np.log(high)))
        elif scale == "linear":
            value = rng.uniform(low, high)
        else:
            raise ValueError(f"Unknown scale {scale!r} for parameter {key!r}.")
        params[key] = float(value)
    return params


def build_design(
    entry: dict[str, Any],
    mode: str,
    n_samples: int,
    reps: int,
    design_rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Produce the list of parameter dicts to generate one cloud each for."""
    if mode == "grid":
        design: list[dict[str, Any]] = []
        for params in iter_param_grid(entry["grid"]):
            design.extend(dict(params) for _ in range(reps))
        return design
    if mode == "random":
        vectors = [sample_param_vector(entry["ranges"], design_rng) for _ in range(n_samples)]
        # `reps` realisations (each a distinct seed) per parameter vector;
        # realisations of the same vector are kept contiguous so the design can
        # be split by parameter group without leakage.
        return [dict(v) for v in vectors for _ in range(reps)]
    raise ValueError(f"Unknown mode {mode!r}.")


def default_region(dimension: int) -> Region:
    """Unit box [0, 1]^d."""
    return Box(low=np.zeros(dimension), high=np.ones(dimension))


def cloud_to_record(cloud: PointCloud) -> dict[str, Any]:
    """Flatten a PointCloud into a plain, picklable dict labelled by its params.

    Contains only numpy arrays and built-in types, so the pickle can be loaded
    without importing ``cloudforger`` (numpy alone suffices). The ``params`` key
    holds the generating parameters -- the label for training/validation.
    """
    record: dict[str, Any] = {
        "points": cloud.points,                 # (N, D) float array
        "params": dict(cloud.generator_params),  # <-- label
        "process": cloud.generator_name,
        "seed": cloud.seed,
        "n_points": cloud.n_points,
        "dimension": cloud.dimension,
    }
    region = cloud.region
    if isinstance(region, Box):
        record["region"] = {"low": region.low, "high": region.high}
    return record


def generate_clouds(
    process_name: str,
    region: Region,
    mode: str,
    n_samples: int,
    reps: int,
    base_seed: int,
    n_hint: int = 0,
) -> list[PointCloud]:
    """Sample one cloud per entry in the parameter design.

    `n_hint` is forwarded to `PointProcess.sample`'s `n` argument. The Thomas
    process derives its own point count from its intensity parameters and so
    ignores it; it is kept for processes that need a target count.
    """
    if process_name not in PROCESS_REGISTRY:
        raise ValueError(
            f"Unknown process {process_name!r}. Available: {sorted(PROCESS_REGISTRY)}"
        )
    entry = PROCESS_REGISTRY[process_name]
    build: ProcessBuilder = entry["build"]

    # Separate RNG for the parameter design so the *choice of parameters* is
    # reproducible and independent of the per-cloud sampling seeds.
    design_rng = np.random.default_rng(base_seed)
    design = build_design(entry, mode, n_samples, reps, design_rng)

    clouds: list[PointCloud] = []
    for offset, params in enumerate(design):
        process = build(params)
        # sample() stamps the cloud with name, params and seed -> labelled.
        clouds.append(process.sample(n=n_hint, region=region, seed=base_seed + offset))
    return clouds


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--process", default="thomas", choices=sorted(PROCESS_REGISTRY),
        help="Point process to generate (default: thomas).",
    )
    parser.add_argument(
        "--mode", default="random", choices=("random", "grid"),
        help="random: draw --n-samples param vectors; grid: sweep grid x --reps.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=5000,
        help="Number of clouds in random mode (default: 500).",
    )
    parser.add_argument(
        "--reps", type=int, default=1,
        help="Realisations (distinct seeds) per parameter vector, both modes "
             "(default: 1). Total clouds = n_samples*reps (random) or "
             "grid_size*reps (grid).",
    )
    parser.add_argument(
        "--dimension", type=int, default=2,
        help="Spatial dimension of the unit-box region (default: 2).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Base seed; drives both the parameter design and per-cloud seeds.",
    )
    parser.add_argument(
        "--format", default="dict", choices=("dict", "object"),
        help="dict: list of plain dicts labelled by 'params' (numpy-only to "
             "reload). object: list of PointCloud instances (needs cloudforger "
             "importable). Default: dict.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output .pkl path (default: data/params/<process>/clouds.pkl).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    region = default_region(args.dimension)
    clouds = generate_clouds(
        process_name=args.process,
        region=region,
        mode=args.mode,
        n_samples=args.n_samples,
        reps=args.reps,
        base_seed=args.seed,
    )

    output = args.output or (PROJECT_ROOT / "data" / "params" / args.process / "clouds.pkl")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = [cloud_to_record(c) for c in clouds] if args.format == "dict" else clouds
    with open(output, "wb") as f:
        pickle.dump(payload, f)

    total_pts = sum(c.n_points for c in clouds)
    n_distinct = len({tuple(sorted(c.generator_params.items())) for c in clouds})
    print(
        f"Generated {len(clouds)} '{args.process}' clouds "
        f"({args.mode} mode: {n_distinct} distinct param vectors x {args.reps} rep(s)) "
        f"in {args.dimension}D, {total_pts} points total "
        f"(min {min(c.n_points for c in clouds)}, max {max(c.n_points for c in clouds)})."
    )
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()