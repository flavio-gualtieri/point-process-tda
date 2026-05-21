# scripts/compute_diagrams.py
"""Stage 2: clouds -> persistence diagrams. Reads clouds.pkl, writes
diagrams.pkl. Rerun only if the filtration config changes."""
import pickle
from pathlib import Path

from cloudforger.tda.filtration.rips import RipsFiltration

DATA_DIR = Path("/Users/qp252676/Desktop/point-process-tda/data/experiment_01")

FILTRATION = RipsFiltration(maxdim=1, thresh=None)


def main():
    with open(DATA_DIR / "clouds.pkl", "rb") as f:
        bundle = pickle.load(f)

    clouds = bundle["clouds"]
    diagrams = [FILTRATION.compute(c) for c in clouds]

    # Labels/names carried straight through, kept index-aligned with diagrams.
    with open(DATA_DIR / "diagrams.pkl", "wb") as f:
        pickle.dump({
            "diagrams": diagrams,
            "labels": bundle["labels"],
            "label_names": bundle["label_names"],
            "filtration_params": FILTRATION.params,
        }, f)

    print(f"Computed {len(diagrams)} diagrams -> {DATA_DIR / 'diagrams.pkl'}")


if __name__ == "__main__":
    main()