# scripts/processing/vectorize_diagrams.py
"""Stage 3: persistence diagrams -> persistence images.
Reads {dim}d/diagrams.pkl, writes {dim}d/images.pkl for each ambient dimension.
Rerun this when sweeping sigma / weighting / resolution."""

import pickle
from pathlib import Path
from cloudforger.tda.calibration import calibrate, calibrate_report
from cloudforger.tda.vectorizer.persistence_image import PersistenceImager
from cloudforger.tda.vectorizer.multi_channel import MultiChannelImager

CONFIG = {
    "dims":       list(range(2, 18)),
    "data_dir":   Path("data"),
    "resolution": 64,
    "sigma":      0.05,   # research knob — sweep this
}

def build_imagers(diagrams) -> MultiChannelImager:
    """Calibrate bounds from the data, then construct one imager per dim."""
    print(calibrate_report(diagrams))
    stats = calibrate(diagrams)
    imagers = {}
    for dim, axes in stats.items():
        b_hi = axes["birth"][99.0]
        p_hi = axes["persistence"][99.0]
        imagers[dim] = PersistenceImager(
            birth_range=(0.0, b_hi if b_hi > 0 else 1.0),
            pers_range=(0.0, p_hi),
            resolution=CONFIG["resolution"],
            sigma=CONFIG["sigma"],
        )
    return MultiChannelImager(imagers)

def vectorize_for_dim(dim: int, data_dir: Path) -> None:
    diagrams_path = data_dir / f"{dim}d" / "diagrams.pkl"
    out_path      = data_dir / f"{dim}d" / "images.pkl"

    with open(diagrams_path, "rb") as f:
        bundle = pickle.load(f)

    diagrams = bundle["diagrams"]

    print(f"  Vectorizing dim={dim} ({len(diagrams)} diagrams) ...")
    imager = build_imagers(diagrams)
    images = [imager.transform(d) for d in diagrams]

    with open(out_path, "wb") as f:
        pickle.dump({
            "images":        images,
            "labels":        bundle["labels"],
            "label_names":   bundle["label_names"],
            "imager_params": imager.params,
            "config":        bundle["config"],   # carry through from diagrams
        }, f)

    print(f"  Saved {len(images)} images → {out_path}")

def main():
    for dim in CONFIG["dims"]:
        vectorize_for_dim(dim, CONFIG["data_dir"])
    print("\nDone.")

if __name__ == "__main__":
    main()