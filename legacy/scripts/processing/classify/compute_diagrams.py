# scripts/processing/compute_diagrams.py
import pickle
from pathlib import Path
from cloudforger.tda.filtration.rips import RipsFiltration

CONFIG = {
    "dims": list(range(2, 18)),
    "data_dir": Path("data"),
    "maxdim": 1,
    "thresh": None,
}

FILTRATION = RipsFiltration(maxdim=CONFIG["maxdim"], thresh=CONFIG["thresh"])

def compute_diagrams_for_dim(dim: int, data_dir: Path) -> None:
    clouds_path = data_dir / f"{dim}d" / "clouds.pkl"
    out_path = data_dir / f"{dim}d" / "diagrams.pkl"

    with open(clouds_path, "rb") as f:
        data = pickle.load(f)

    clouds = data["clouds"]
    n = len(clouds)
    diagrams = []

    print(f"  Computing {n} diagrams for dim={dim} ...")
    for i, cloud in enumerate(clouds):
        diagrams.append(FILTRATION.compute(cloud))
        print(f"\r    {i + 1}/{n}", end="", flush=True)
    print()

    with open(out_path, "wb") as f:
        pickle.dump({
            "diagrams": diagrams,
            "labels": data["labels"],
            "label_names": data["label_names"],
            "filtration_params": FILTRATION.params,
            "config": data["config"],   # carry through from clouds
        }, f)

    print(f"  Saved {len(diagrams)} diagrams → {out_path}")

def main():
    for dim in CONFIG["dims"]:
        compute_diagrams_for_dim(dim, CONFIG["data_dir"])
    print("\nDone.")

if __name__ == "__main__":
    main()