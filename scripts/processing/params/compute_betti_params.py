# scripts/processing/params/compute_betti_params.py

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

from cloudforger.core.diagram import PersistenceDiagram
from cloudforger.core.betti import BettiCurveFeature
from cloudforger.tda.features.betti_curve import BettiCurve


CONFIG: dict[str, Any] = {
    "process": "thomas",

    "dimensions": [2],

    "format": "dict",

    "betti": {
        "homology_dims": (0, 1),
        "grid_size": 128,
        "grid_range": (0.0, 1.0),
        "drop_infinite": True,
        "normalize": False,
    },

    "splits": {
        "train_test": {
            "input": "diagrams.pkl",
            "output": "betti.pkl",
        },
        "adversarial": {
            "input": "adversarial_diagrams.pkl",
            "output": "adversarial_betti.pkl",
        },
    },

    "skip_missing": True,
}


def validate_config(config: dict[str, Any]) -> None:
    if config.get("format", "dict") not in {"dict", "object"}:
        raise ValueError("CONFIG['format'] must be 'dict' or 'object'.")

    if "process" not in config:
        raise ValueError("CONFIG must include 'process'.")

    if "dimensions" not in config:
        raise ValueError("CONFIG must include 'dimensions'.")

    if "splits" not in config or not config["splits"]:
        raise ValueError("CONFIG must include at least one split in 'splits'.")
    

def normalize_dimensions(value: int | list[int]) -> list[int]:
    if isinstance(value, int):
        return [value]

    if isinstance(value, list) and all(isinstance(v, int) for v in value):
        return value

    raise TypeError(
        "CONFIG['dimensions'] must be an int, e.g. 2, "
        "or a list of ints, e.g. [2, 3, 5]."
    )


def betti_features_to_record(curve: BettiCurveFeature) -> dict[str, Any]:
    return {
        "curves": {
            int(dim): np.asarray(curve)
            for dim, curve in curve.curves.items()
        },
        "vector": curve.vector(),
        "params": dict(curve.generator_params),
        "seed": curve.seed,
        "process": curve.generator_name,
        "filtration": curve.filtration_name,
        "filtration_params": dict(curve.filtration_params),
        "feature": curve.feature_name,
        "feature_params": dict(curve.feature_params),
    }


def load_diagrams(path: Path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "diagrams" in data:
        return data["diagrams"]

    return data


def diagram_seed(diagram: Any) -> Any:
    if isinstance(diagram, dict):
        return diagram.get("seed")
    return diagram.seed


def build_betti(config: dict[str, Any]) -> BettiCurve:
    betti_cfg = config["betti"]
    return BettiCurve(
        homology_dims=betti_cfg.get("homology_dims", (0, 1)),
        grid_size=betti_cfg.get("grid_size", 128),
        grid_range=betti_cfg.get("grid_range", (0.0, 1.0)),
        drop_infinite=betti_cfg.get("drop_infinite", True),
        normalize=betti_cfg.get("normalize", False),
    )
    

def diagram_params(diagram: Any) -> dict[str, Any]:
    if isinstance(diagram, dict):
        return dict(diagram["params"])
    return dict(diagram.generator_params)


def build_labels(diagrams: list) -> tuple[np.ndarray, list[str], list[dict]]:
    if not diagrams:
        raise ValueError("Cannot build labels from an empty diagram list.")

    params = [diagram_params(d) for d in diagrams]
    label_names = list(params[0].keys())

    for p in params:
        if list(p.keys()) != label_names:
            raise ValueError(
                "Diagrams have inconsistent parameter keys; cannot stack labels."
            )

    labels = np.array(
        [[p[k] for k in label_names] for p in params],
        dtype=float,
    )
    return labels, label_names, params


def to_diagram(diagram: Any) -> PersistenceDiagram:
    if isinstance(diagram, dict):
        return PersistenceDiagram(
            diagrams={
                int(dim): np.asarray(pairs)
                for dim, pairs in diagram["diagrams"].items()
            },
            generator_name=diagram.get("process", "thomas"),
            generator_params=dict(diagram.get("params", {})),
            seed=diagram.get("seed"),
            filtration_name=diagram.get("filtration", ""),
            filtration_params=dict(diagram.get("filtration_params", {})),
        )

    return diagram


def compute_curves(
        diagrams_path: Path,
        out_path: Path,
        betti: BettiCurve,
        process: str,
        curve_format: str = "dict"
) -> None:
    diagrams = load_diagrams(diagrams_path)
    labels, label_names, params = build_labels(diagrams)
    seeds = [diagram_seed(d) for d in diagrams]

    n = len(diagrams)
    computed: list[BettiCurveFeature] = []

    print(f"  Computing {n} curves from {diagrams_path} ...")
    for i, d in enumerate(diagrams):
        diagram = to_diagram(d)
        computed.append(betti.compute(diagram))

        print(f"\r    {i + 1}/{n}", end="", flush=True)
    print()

    curves = (
        [betti_features_to_record(curve) for curve in computed]
        if curve_format == "dict"
        else computed
    )

    betti0_matrix = np.stack([curve.curves[0] for curve in computed])
    betti1_matrix = np.stack([curve.curves[1] for curve in computed])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "curves": curves,
                "betti0_matrix": betti0_matrix,
                "betti1_matrix": betti1_matrix,
                "labels": labels,
                "label_names": label_names,
                "params": params,
                "seeds": seeds,
                "process": process,
                "betti_params": betti.params,
            },
            f,
        )

    print(f"  Saved {len(diagrams)} curves → {out_path}")


def main() -> None:
    validate_config(CONFIG)

    process = CONFIG["process"]
    dimensions = normalize_dimensions(CONFIG["dimensions"])
    curve_format = CONFIG.get("format", "dict")
    skip_missing = bool(CONFIG.get("skip_missing", True))

    betti = build_betti(CONFIG)

    for dim in dimensions:
        base_dir = PROJECT_ROOT / "data" / "params" / f"{dim}d" / process
        print(f"\n[dim={dim}] Base directory: {base_dir}")

        for split_name, split_cfg in CONFIG["splits"].items():
            diagrams_path = base_dir / split_cfg["input"]
            out_path = base_dir / split_cfg["output"]

            if not diagrams_path.exists():
                message = f"[dim={dim}] Missing {split_name} diagrams: {diagrams_path}"
                if skip_missing:
                    print(f"  Skipping: {message}")
                    continue
                raise FileNotFoundError(message)

            print(f"  Split: {split_name}")
            compute_curves(
                diagrams_path=diagrams_path,
                out_path=out_path,
                betti=betti,
                process=process,
                curve_format=curve_format,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()