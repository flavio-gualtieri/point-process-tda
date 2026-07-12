# scripts/processing/params/pipeline_lib/images.py
"""Stage: vectorize_diagrams — persistence images from persistence diagrams."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cloudforger.core.diagram import PersistenceDiagram
from cloudforger.tda.calibration import calibrate, calibrate_report
from cloudforger.tda.vectorizer.multi_channel import MultiChannelImager
from cloudforger.tda.vectorizer.persistence_image import PersistenceImager

from pipeline_lib.io import dump_pickle
from pipeline_lib.records import load_diagrams


def build_imagers(
    diagrams: list[PersistenceDiagram],
    homology_dims: list[int],
    resolution: int,
    sigma: float,
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
            birth_hi = 4.0 * sigma if birth_hi <= 0 else birth_hi
            persistence_hi = 1.0 if persistence_hi <= 0 else persistence_hi

        imagers[dim] = PersistenceImager(
            birth_range=(0.0, float(birth_hi)),
            pers_range=(0.0, float(persistence_hi)),
            resolution=int(resolution),
            sigma=float(sigma),
        )

    return MultiChannelImager(imagers)


def stack_images_by_dim(
    images: list[dict[int, np.ndarray]],
    homology_dims: list[int],
    resolution: int,
) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for dim in homology_dims:
        if images:
            result[dim] = np.stack([img[dim] for img in images])
        else:
            result[dim] = np.empty((0, resolution, resolution), dtype=float)
    return result


def vectorize_diagram_split(
    diagrams_path: Path,
    out_path: Path,
    imager: MultiChannelImager,
    homology_dims: list[int],
    resolution: int,
    dim: int,
    split_name: str,
) -> None:
    diagrams, bundle = load_diagrams(diagrams_path)
    print(f"  Vectorizing {split_name} ({len(diagrams)} diagrams) from {diagrams_path} ...")

    images = [imager.transform(d) for d in diagrams]
    image_tensors = stack_images_by_dim(images, homology_dims, resolution)

    payload: dict[str, Any] = {
        "images": images,
        "image_tensors": image_tensors,
        "homology_dims": imager.dimensions,
        "imager_params": imager.params,
        "dimension": dim,
        "split": split_name,
        "source_diagrams": str(diagrams_path),
    }

    for key in ("labels", "label_names", "params", "seeds", "process", "filtration_params"):
        if key in bundle:
            payload[key] = bundle[key]

    dump_pickle(out_path, payload)
    shapes = {k: v.shape for k, v in image_tensors.items()}
    print(f"  Saved {len(images)} images → {out_path}")
    print(f"  Image tensors by dimension: {shapes}")
