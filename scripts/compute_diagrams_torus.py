# scripts/compute_diagrams_torus.py
import pickle
from pathlib import Path

from cloudforger.tda.filtration.rips import RipsFiltration

DATA_DIR = Path("/Users/qp252676/Desktop/point-process-tda/data")

FILTRATION = RipsFiltration(maxdim=1, thresh=None)


def main():
    with open(DATA_DIR / "clouds_torus.pkl", "rb") as f:
        data = pickle.load(f)

    clouds = data["clouds"]
    diagrams = []
    n = len(clouds)
    for i, c in enumerate(clouds):
        diagrams.append(FILTRATION.compute(c))
        print(f"\r  {i + 1}/{n}", end="", flush=True)
    print()

    # Labels/names carried straight through, kept index-aligned with diagrams.
    with open(DATA_DIR / "diagrams_torus.pkl", "wb") as f:
        pickle.dump({
            "diagrams": diagrams,
            "labels": data["labels"],
            "label_names": data["label_names"],
            "filtration_params": FILTRATION.params,
        }, f)

    print(f"Computed {len(diagrams)} diagrams -> {DATA_DIR / 'diagrams_torus.pkl'}")


if __name__ == "__main__":
    main()