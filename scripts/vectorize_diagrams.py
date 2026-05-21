# scripts/vectorize_diagrams.py
"""Stage 3: persistence diagrams -> persistence images. Reads diagrams.pkl,
writes images.pkl. Rerun this when sweeping sigma / weighting / resolution."""
import pickle
from pathlib import Path

import numpy as np

from cloudforger.tda.calibration import calibrate, calibrate_report
from cloudforger.tda.vectorizer.persistence_image import PersistenceImager
from cloudforger.tda.vectorizer.multi_channel import MultiChannelImager

DATA_DIR = Path("data/experiment_01")

RESOLUTION = 64
SIGMA = 0.05  # research knob — sweep this


def build_imagers(diagrams) -> MultiChannelImager:
    """Calibrate bounds from the data, then construct one imager per dim."""
    print(calibrate_report(diagrams))
    stats = calibrate(diagrams)

    imagers = {}
    for dim, axes in stats.items():
        # Birth floor at 0 (births are non-negative; H0 births are all 0).
        # Ceiling from the 99th percentile so a rare giant feature doesn't
        # blow out the range — its kernel tail still lands in-grid.
        b_hi = axes["birth"][99.0]
        p_hi = axes["persistence"][99.0]
        imagers[dim] = PersistenceImager(
            birth_range=(0.0, b_hi if b_hi > 0 else 1.0),
            pers_range=(0.0, p_hi),
            resolution=RESOLUTION,
            sigma=SIGMA,
        )
    return MultiChannelImager(imagers)


def main():
    with open(DATA_DIR / "diagrams.pkl", "rb") as f:
        bundle = pickle.load(f)

    diagrams = bundle["diagrams"]
    imager = build_imagers(diagrams)

    # images[i] is {0: (R,R) array, 1: (R,R) array} for diagram i.
    images = [imager.transform(d) for d in diagrams]

    with open(DATA_DIR / "images.pkl", "wb") as f:
        pickle.dump({
            "images": images,
            "labels": bundle["labels"],
            "label_names": bundle["label_names"],
            "imager_params": imager.params,
        }, f)

    print(f"Vectorized {len(images)} diagrams -> {DATA_DIR / 'images.pkl'}")


if __name__ == "__main__":
    main()