# scripts/processing/params/compute_diagrams_params.py

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

# --- make the cloudforger package importable when run as a plain script ------
# scripts/processing/params/<this file> -> parents[3] == project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SRC = PROJECT_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cloudforger.core.cloud import PointCloud
from cloudforger.core.diagram import PersistenceDiagram
from cloudforger.core.region import Box
from cloudforger.tda.filtration.rips import RipsFiltration


CONFIG: dict[str, Any] = {
    # Process/dataset identity.
    "process": "thomas",

    # Accepts either an int or a list[int].
    # Examples:
    #   "dimensions": 2
    #   "dimensions": [2, 3, 5]
    "dimensions": [2],

    # Output representation.
    # "dict" is portable and reloads with numpy alone.
    # "object" stores PersistenceDiagram instances.
    "format": "dict",

    # Rips filtration settings.
    "filtration": {
        "maxdim": 1,
        "thresh": None,
    },

    # Files to process under each data/params/{dim}d/{process}/ directory.
    "splits": {
        "train_test": {
            "input": "clouds.pkl",
            "output": "diagrams.pkl",
        },
        "adversarial": {
            "input": "adversarial_clouds.pkl",
            "output": "adversarial_diagrams.pkl",
        },
    },

    # If True, missing split files are skipped.
    # If False, missing split files raise FileNotFoundError.
    "skip_missing": True,
}


def normalize_dimensions(value: int | list[int]) -> list[int]:
    if isinstance(value, int):
        return [value]

    if isinstance(value, list) and all(isinstance(v, int) for v in value):
        return value

    raise TypeError(
        "CONFIG['dimensions'] must be an int, e.g. 2, "
        "or a list of ints, e.g. [2, 3, 5]."
    )


def diagram_to_record(diagram: PersistenceDiagram) -> dict[str, Any]:
    return {
        "diagrams": {
            int(dim): np.asarray(pairs)
            for dim, pairs in diagram.diagrams.items()
        },
        "params": dict(diagram.generator_params),
        "seed": diagram.seed,
        "process": diagram.generator_name,
        "filtration": diagram.filtration_name,
        "filtration_params": dict(diagram.filtration_params),
    }


def to_pointcloud(cloud: Any) -> PointCloud:
    if not isinstance(cloud, dict):
        return cloud

    region = None
    reg = cloud.get("region")
    if reg is not None:
        try:
            region = Box(
                low=np.asarray(reg["low"], dtype=float),
                high=np.asarray(reg["high"], dtype=float),
            )
        except Exception:
            region = None

    return PointCloud(
        points=np.asarray(cloud["points"]),
        generator_name=cloud.get("process", ""),
        generator_params=dict(cloud.get("params", {})),
        seed=cloud.get("seed"),
        region=region,
    )


def cloud_params(cloud: Any) -> dict[str, Any]:
    if isinstance(cloud, dict):
        return dict(cloud["params"])
    return dict(cloud.generator_params)


def cloud_seed(cloud: Any) -> Any:
    if isinstance(cloud, dict):
        return cloud.get("seed")
    return cloud.seed


def load_clouds(path: Path) -> list:
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "clouds" in data:
        return data["clouds"]

    return data


def build_labels(clouds: list) -> tuple[np.ndarray, list[str], list[dict]]:
    if not clouds:
        raise ValueError("Cannot build labels from an empty cloud list.")

    params = [cloud_params(c) for c in clouds]
    label_names = list(params[0].keys())

    for p in params:
        if list(p.keys()) != label_names:
            raise ValueError(
                "Clouds have inconsistent parameter keys; cannot stack labels."
            )

    labels = np.array(
        [[p[k] for k in label_names] for p in params],
        dtype=float,
    )
    return labels, label_names, params


def compute_diagrams(
        clouds_path: Path,
        out_path: Path,
        filtration: RipsFiltration, process: str,
        diagram_format: str = "dict"
) -> None:
    clouds = load_clouds(clouds_path)
    labels, label_names, params = build_labels(clouds)
    seeds = [cloud_seed(c) for c in clouds]

    n = len(clouds)
    computed: list[PersistenceDiagram] = []

    print(f"  Computing {n} diagrams from {clouds_path} ...")
    for i, cloud in enumerate(clouds):
        point_cloud = to_pointcloud(cloud)
        computed.append(filtration.compute(point_cloud))
        print(f"\r    {i + 1}/{n}", end="", flush=True)
    print()

    diagrams = (
        [diagram_to_record(diagram) for diagram in computed]
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

    print(f"  Saved {len(diagrams)} diagrams → {out_path}")


def build_filtration(config: dict[str, Any]) -> RipsFiltration:
    """Build the configured filtration."""
    filtration_cfg = config["filtration"]
    return RipsFiltration(
        maxdim=int(filtration_cfg.get("maxdim", 1)),
        thresh=filtration_cfg.get("thresh", None),
    )


def validate_config(config: dict[str, Any]) -> None:
    """Validate top-level script config."""
    if config.get("format", "dict") not in {"dict", "object"}:
        raise ValueError("CONFIG['format'] must be 'dict' or 'object'.")

    if "process" not in config:
        raise ValueError("CONFIG must include 'process'.")

    if "dimensions" not in config:
        raise ValueError("CONFIG must include 'dimensions'.")

    if "splits" not in config or not config["splits"]:
        raise ValueError("CONFIG must include at least one split in 'splits'.")


def main() -> None:
    validate_config(CONFIG)

    process = CONFIG["process"]
    dimensions = normalize_dimensions(CONFIG["dimensions"])
    diagram_format = CONFIG.get("format", "dict")
    skip_missing = bool(CONFIG.get("skip_missing", True))

    filtration = build_filtration(CONFIG)

    for dim in dimensions:
        base_dir = PROJECT_ROOT / "data" / "params" / f"{dim}d" / process
        print(f"\n[dim={dim}] Base directory: {base_dir}")

        for split_name, split_cfg in CONFIG["splits"].items():
            clouds_path = base_dir / split_cfg["input"]
            out_path = base_dir / split_cfg["output"]

            if not clouds_path.exists():
                message = f"[dim={dim}] Missing {split_name} clouds: {clouds_path}"
                if skip_missing:
                    print(f"  Skipping: {message}")
                    continue
                raise FileNotFoundError(message)

            print(f"  Split: {split_name}")
            compute_diagrams(
                clouds_path=clouds_path,
                out_path=out_path,
                filtration=filtration,
                process=process,
                diagram_format=diagram_format,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()