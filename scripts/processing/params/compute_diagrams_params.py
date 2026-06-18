#!/usr/bin/env python3
"""Compute persistence diagrams for a saved collection of point clouds.

Reads the clouds produced by ``generate_clouds_params.py`` -- a ``list`` of dict
records, each labelled by its generating parameters under the ``"params"`` key --
runs a Rips filtration over every cloud, and saves the resulting diagrams
together with the aligned parameter labels.

Input  (default): data/params/<process>/clouds.pkl
Output (default): data/params/<process>/diagrams.pkl

The output pickle is a dict:
    {
        "diagrams":          list, one entry per cloud (aligned with labels);
                             each is a record {"diagrams": {dim: (M,2) array},
                             "params": {...}, "seed", "process", "filtration",
                             "filtration_params"} under --format dict, or a
                             PersistenceDiagram under --format object,
        "labels":            (N, P) float array of generating parameters,
        "label_names":       list[str] naming the P label columns,
        "params":            list[dict], the raw per-cloud parameters,
        "seeds":             list[int],
        "process":           str,
        "filtration_params": dict,
    }

With --format dict (default) the file reloads with numpy alone; --format object
stores PersistenceDiagram instances and needs cloudforger importable.

Examples
--------
    python scripts/processing/params/compute_diagrams_params.py
    python scripts/processing/params/compute_diagrams_params.py --maxdim 2 --thresh 0.5
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

# --- make the cloudforger package importable when run as a plain script ------
# scripts/processing/params/<this file>  ->  parents[3] == project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SRC = PROJECT_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cloudforger.core.cloud import PointCloud           # noqa: E402
from cloudforger.core.diagram import PersistenceDiagram  # noqa: E402
from cloudforger.core.region import Box                 # noqa: E402
from cloudforger.tda.filtration.rips import RipsFiltration  # noqa: E402


def diagram_to_record(pd: PersistenceDiagram) -> dict[str, Any]:
    """Flatten a PersistenceDiagram into a plain, picklable dict labelled by params.

    Holds only numpy arrays and built-ins, so the diagrams pickle reloads with
    numpy alone -- no need to import ``cloudforger``. Birth/death arrays are kept
    as returned by the filtration (the H0 infinite class keeps its ``inf``).
    """
    return {
        "diagrams": {int(dim): np.asarray(pairs) for dim, pairs in pd.diagrams.items()},
        "params": dict(pd.generator_params),   # <-- label
        "seed": pd.seed,
        "process": pd.generator_name,
        "filtration": pd.filtration_name,
        "filtration_params": dict(pd.filtration_params),
    }


# --- record accessors --------------------------------------------------------
# Work whether the cloud was saved as a dict (default --format dict) or as a
# PointCloud instance (--format object).

def to_pointcloud(cloud: Any) -> PointCloud:
    """Return a PointCloud for the filtration.

    The filtration consumes a PointCloud (it reads ``cloud.points`` internally),
    so dict records are rebuilt into PointCloud instances; instances are passed
    through unchanged.
    """
    if not isinstance(cloud, dict):
        return cloud
    region = None
    reg = cloud.get("region")
    if reg is not None:
        try:
            region = Box(low=np.asarray(reg["low"], dtype=float),
                         high=np.asarray(reg["high"], dtype=float))
        except Exception:
            region = None  # region is optional; don't fail the whole run over it
    return PointCloud(
        points=np.asarray(cloud["points"]),
        generator_name=cloud.get("process", ""),
        generator_params=dict(cloud.get("params", {})),
        seed=cloud.get("seed"),
        region=region,
    )


def cloud_params(cloud: Any) -> dict[str, Any]:
    """The generating parameters (the label) for one cloud."""
    if isinstance(cloud, dict):
        return dict(cloud["params"])
    return dict(cloud.generator_params)


def cloud_seed(cloud: Any) -> Any:
    return cloud["seed"] if isinstance(cloud, dict) else cloud.seed


# --- loading & labels --------------------------------------------------------

def load_clouds(path: Path) -> list:
    with open(path, "rb") as f:
        data = pickle.load(f)
    # Accept the flat list-of-records format, or a {"clouds": [...]} wrapper.
    if isinstance(data, dict) and "clouds" in data:
        return data["clouds"]
    return data


def build_labels(clouds: list) -> tuple[np.ndarray, list[str], list[dict]]:
    """Stack the per-cloud parameter dicts into an (N, P) label array."""
    params = [cloud_params(c) for c in clouds]
    label_names = list(params[0].keys())
    for p in params:
        if list(p.keys()) != label_names:
            raise ValueError(
                "Clouds have inconsistent parameter keys; cannot stack labels."
            )
    labels = np.array([[p[k] for k in label_names] for p in params], dtype=float)
    return labels, label_names, params


# --- main computation --------------------------------------------------------

def compute_diagrams(
    clouds_path: Path,
    out_path: Path,
    filtration: RipsFiltration,
    process: str,
    diagram_format: str = "dict",
) -> None:
    clouds = load_clouds(clouds_path)
    n = len(clouds)
    labels, label_names, params = build_labels(clouds)
    seeds = [cloud_seed(c) for c in clouds]

    computed = []
    print(f"  Computing {n} diagrams from {clouds_path} ...")
    for i, cloud in enumerate(clouds):
        # The filtration reads cloud.points internally, so pass a PointCloud.
        computed.append(filtration.compute(to_pointcloud(cloud)))
        print(f"\r    {i + 1}/{n}", end="", flush=True)
    print()

    # dict: portable records labelled by params (numpy-only to reload).
    # object: the PersistenceDiagram instances (needs cloudforger to reload).
    diagrams = (
        [diagram_to_record(pd) for pd in computed]
        if diagram_format == "dict"
        else computed
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "diagrams": diagrams,
                "labels": labels,
                "label_names": label_names,
                "params": params,
                "seeds": seeds,
                "process": process,
                "filtration_params": filtration.params,
            },
            f,
        )
    print(f"  Saved {len(diagrams)} diagrams \u2192 {out_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--process", default="thomas",
        help="Process name; sets default in/out paths (default: thomas).",
    )
    parser.add_argument(
        "--clouds", type=Path, default=None,
        help="Input clouds .pkl (default: data/params/<process>/clouds.pkl).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output diagrams .pkl (default: data/params/<process>/diagrams.pkl).",
    )
    parser.add_argument(
        "--maxdim", type=int, default=1,
        help="Max homology dimension for the Rips filtration (default: 1).",
    )
    parser.add_argument(
        "--format", default="dict", choices=("dict", "object"),
        help="dict: portable records labelled by 'params' (numpy-only to "
             "reload). object: PersistenceDiagram instances (needs cloudforger "
             "importable). Default: dict.",
    )
    parser.add_argument(
        "--thresh", type=float, default=None,
        help="Distance threshold for the Rips filtration (default: none).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    clouds_path = args.clouds or (
        PROJECT_ROOT / "data" / "params" / args.process / "clouds.pkl"
    )
    out_path = args.output or (
        PROJECT_ROOT / "data" / "params" / args.process / "diagrams.pkl"
    )
    filtration = RipsFiltration(maxdim=args.maxdim, thresh=args.thresh)
    compute_diagrams(clouds_path, out_path, filtration, args.process, args.format)
    print("\nDone.")


if __name__ == "__main__":
    main()