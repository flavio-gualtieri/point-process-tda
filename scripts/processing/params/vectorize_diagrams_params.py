import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SRC = PROJECT_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

CONFIG = {
    "data_dir": Path("data/params"),
    "resolution": 64,
    "sigma": 0.05,   # research knob — sweep this
    "process": "thomas"
}

from cloudforger.core.diagram import PersistenceDiagram
from cloudforger.tda.calibration import calibrate, calibrate_report
from cloudforger.tda.vectorizer.persistence_image import PersistenceImager
from cloudforger.tda.vectorizer.multi_channel import MultiChannelImager


def _record_to_diagram(record: dict) -> PersistenceDiagram:
    """Convert a plain dict record (from --format dict) to a PersistenceDiagram."""
    return PersistenceDiagram(
        diagrams=record["diagrams"],
        generator_name=record.get("process", ""),
        generator_params=dict(record.get("params", {})),
        seed=record.get("seed"),
        filtration_name=record.get("filtration", ""),
        filtration_params=dict(record.get("filtration_params", {})),
    )


def _as_diagram(d) -> PersistenceDiagram:
    return _record_to_diagram(d) if isinstance(d, dict) else d

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

def vectorize(data_dir: Path) -> None:
    diagrams_path = data_dir / CONFIG["process"] / "diagrams.pkl"
    out_path = data_dir / CONFIG["process"] / "images.pkl"

    with open(diagrams_path, "rb") as f:
        bundle = pickle.load(f)

    diagrams = [_as_diagram(d) for d in bundle["diagrams"]]

    print(f"  Vectorizing ({len(diagrams)} diagrams) ...")
    imager = build_imagers(diagrams)
    images = [imager.transform(d) for d in diagrams]

    with open(out_path, "wb") as f:
        pickle.dump({
            "images": images,
            "labels": bundle["labels"],
            "label_names": bundle["label_names"],
            "imager_params": imager.params,
            "config": {k: bundle[k] for k in ("params", "seeds", "process", "filtration_params") if k in bundle},
        }, f)

    print(f" Saved {len(images)} images → {out_path}")


def main():
    vectorize(CONFIG["data_dir"])
    print("\nDone.")

if __name__ == "__main__":
    main()