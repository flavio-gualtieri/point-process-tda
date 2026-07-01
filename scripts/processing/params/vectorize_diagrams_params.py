# scripts/processing/params/vectorize_diagrams_params.py

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SRC = PROJECT_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cloudforger.core.diagram import PersistenceDiagram
from cloudforger.tda.calibration import calibrate, calibrate_report
from cloudforger.tda.vectorizer.persistence_image import PersistenceImager
from cloudforger.tda.vectorizer.multi_channel import MultiChannelImager


CONFIG: dict[str, Any] = {
    "data_root": "data/params",
    "process": "thomas",
    "dimensions": [2],
    "homology_dims": [0, 1],
    "resolution": 64,
    "sigma": 0.05,
    "calibration_split": "train_test",
    "skip_missing": True,
    "splits": {
        "train_test": {
            "input": "diagrams.pkl",
            "output": "images.pkl",
        },
        "adversarial": {
            "input": "adversarial_diagrams.pkl",
            "output": "adversarial_images.pkl",
        },
    },
}


def normalize_dimensions(value: int | list[int]) -> list[int]:
    if isinstance(value, int):
        return [value]

    if isinstance(value, list) and all(isinstance(v, int) for v in value):
        return value

    raise TypeError("CONFIG['dimensions'] must be an int or a list of ints.")


def normalize_homology_dims(value: list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(value, tuple):
        value = list(value)

    if isinstance(value, list) and all(isinstance(v, int) for v in value):
        return value

    raise TypeError("CONFIG['homology_dims'] must be a list or tuple of ints.")


def base_dir(process: str, dim: int) -> Path:
    return PROJECT_ROOT / CONFIG["data_root"] / f"{dim}d" / process


def load_diagram_bundle(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "diagrams" in data:
        return data

    return {"diagrams": data}


def record_to_diagram(record: dict[str, Any]) -> PersistenceDiagram:
    return PersistenceDiagram(
        diagrams={
            int(dim): np.asarray(pairs, dtype=float)
            for dim, pairs in record["diagrams"].items()
        },
        generator_name=record.get("process", ""),
        generator_params=dict(record.get("params", {})),
        seed=record.get("seed"),
        filtration_name=record.get("filtration", ""),
        filtration_params=dict(record.get("filtration_params", {})),
    )


def as_diagram(value: Any) -> PersistenceDiagram:
    if isinstance(value, PersistenceDiagram):
        return value

    if isinstance(value, dict):
        return record_to_diagram(value)

    raise TypeError(f"Expected PersistenceDiagram or dict, got {type(value)}.")


def load_diagrams(path: Path) -> tuple[list[PersistenceDiagram], dict[str, Any]]:
    bundle = load_diagram_bundle(path)
    diagrams = [as_diagram(d) for d in bundle["diagrams"]]
    return diagrams, bundle


def build_imagers(
    diagrams: list[PersistenceDiagram],
    homology_dims: list[int],
) -> MultiChannelImager:
    print(calibrate_report(diagrams))

    stats = calibrate(diagrams)
    imagers: dict[int, PersistenceImager] = {}

    for dim in homology_dims:
        axes = stats.get(dim)

        if axes is None:
            birth_hi = 1.0
            persistence_hi = 1.0
        else:
            birth_hi = axes["birth"].get(99.0, 1.0)
            persistence_hi = axes["persistence"].get(99.0, 1.0)

            if birth_hi <= 0:
                birth_hi = 1.0

            if persistence_hi <= 0:
                persistence_hi = 1.0

        imagers[dim] = PersistenceImager(
            birth_range=(0.0, float(birth_hi)),
            pers_range=(0.0, float(persistence_hi)),
            resolution=int(CONFIG["resolution"]),
            sigma=float(CONFIG["sigma"]),
        )

    return MultiChannelImager(imagers)


def stack_images_by_dim(
    images: list[dict[int, np.ndarray]],
    homology_dims: list[int],
) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}

    for dim in homology_dims:
        if images:
            result[dim] = np.stack([img[dim] for img in images])
        else:
            resolution = int(CONFIG["resolution"])
            result[dim] = np.empty((0, resolution, resolution), dtype=float)

    return result


def output_payload(
    images: list[dict[int, np.ndarray]],
    image_tensors: dict[int, np.ndarray],
    bundle: dict[str, Any],
    imager: MultiChannelImager,
    dim: int,
    split_name: str,
    diagrams_path: Path,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "images": images,
        "image_tensors": image_tensors,
        "homology_dims": imager.dimensions,
        "imager_params": imager.params,
        "dimension": dim,
        "split": split_name,
        "source_diagrams": str(diagrams_path),
    }

    for key in (
        "labels",
        "label_names",
        "params",
        "seeds",
        "process",
        "filtration_params",
    ):
        if key in bundle:
            payload[key] = bundle[key]

    return payload


def vectorize_split(
    diagrams_path: Path,
    out_path: Path,
    imager: MultiChannelImager,
    homology_dims: list[int],
    dim: int,
    split_name: str,
) -> None:
    diagrams, bundle = load_diagrams(diagrams_path)

    print(f"  Vectorizing {split_name} ({len(diagrams)} diagrams) from {diagrams_path} ...")

    images = [imager.transform(d) for d in diagrams]
    image_tensors = stack_images_by_dim(images, homology_dims)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(
            output_payload(
                images=images,
                image_tensors=image_tensors,
                bundle=bundle,
                imager=imager,
                dim=dim,
                split_name=split_name,
                diagrams_path=diagrams_path,
            ),
            f,
        )

    shapes = {k: v.shape for k, v in image_tensors.items()}
    print(f"  Saved {len(images)} images → {out_path}")
    print(f"  Image tensors by dimension: {shapes}")


def build_calibration_imager(
    dim: int,
    process: str,
    homology_dims: list[int],
) -> MultiChannelImager | None:
    split_name = CONFIG["calibration_split"]
    split_cfg = CONFIG["splits"][split_name]
    diagrams_path = base_dir(process, dim) / split_cfg["input"]

    if not diagrams_path.exists():
        message = f"[dim={dim}] Missing calibration diagrams: {diagrams_path}"
        if CONFIG.get("skip_missing", True):
            print(f"  Skipping dimension: {message}")
            return None
        raise FileNotFoundError(message)

    diagrams, _ = load_diagrams(diagrams_path)
    return build_imagers(diagrams, homology_dims)


def main() -> None:
    process = CONFIG["process"]
    dimensions = normalize_dimensions(CONFIG["dimensions"])
    homology_dims = normalize_homology_dims(CONFIG["homology_dims"])
    skip_missing = bool(CONFIG.get("skip_missing", True))

    for dim in dimensions:
        data_path = base_dir(process, dim)
        print(f"\n[dim={dim}] Base directory: {data_path}")

        imager = build_calibration_imager(
            dim=dim,
            process=process,
            homology_dims=homology_dims,
        )

        if imager is None:
            continue

        for split_name, split_cfg in CONFIG["splits"].items():
            diagrams_path = data_path / split_cfg["input"]
            out_path = data_path / split_cfg["output"]

            if not diagrams_path.exists():
                message = f"[dim={dim}] Missing {split_name} diagrams: {diagrams_path}"
                if skip_missing:
                    print(f"  Skipping: {message}")
                    continue
                raise FileNotFoundError(message)

            vectorize_split(
                diagrams_path=diagrams_path,
                out_path=out_path,
                imager=imager,
                homology_dims=homology_dims,
                dim=dim,
                split_name=split_name,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()