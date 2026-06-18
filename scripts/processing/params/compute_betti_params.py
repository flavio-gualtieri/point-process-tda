import argparse
import os
import pickle
import numpy as np
from pathlib import Path

CONFIG = {
    "data_dir": Path("data/params"),
    "process": "thomas"
}

def compute_betti_curves(diagrams: list, maxdim: int, n_steps: int = 100) -> list:
    all_finite = []
    for dgm in diagrams:
        for k in range(maxdim + 1):
            arr = dgm["diagrams"].get(k)
            if arr is None or len(arr) == 0:
                continue
            arr = np.asarray(arr)
            all_finite.extend(arr[:, 0].tolist())                        # births
            finite_deaths = arr[np.isfinite(arr[:, 1]), 1]
            all_finite.extend(finite_deaths.tolist())

    if not all_finite:
        t_min, t_max = 0.0, 1.0
    else:
        t_min = float(np.min(all_finite))
        t_max = float(np.max(all_finite))

    t_grid = np.linspace(t_min, t_max, n_steps)

    betti_curves = []
    for dgm in diagrams:
        betti = np.zeros((maxdim + 1, n_steps), dtype=np.int32)
        for k in range(maxdim + 1):
            arr = dgm["diagrams"].get(k)
            if arr is None or len(arr) == 0:
                continue
            arr = np.asarray(arr)
            births = arr[:, 0]
            deaths = arr[:, 1]
            # treat infinite deaths as alive past the end of the grid
            deaths = np.where(np.isfinite(deaths), deaths, t_max + 1e-9)
            for j, t in enumerate(t_grid):
                betti[k, j] = int(np.sum((births <= t) & (t < deaths)))
        betti_curves.append({"t": t_grid, "betti": betti})

    return betti_curves


def run_betti_pipeline(
    maxdim: int = 1,
    n_steps: int = 100,
) -> None:
    diagrams_path = CONFIG["data_dir"] / CONFIG["process"] / "diagrams.pkl"
    out_path = CONFIG["data_dir"] / CONFIG["process"] / "betti.pkl"

    if not diagrams_path.exists():
        raise FileNotFoundError(f"Missing input file: {diagrams_path}")

    with open(diagrams_path, "rb") as f:
        data = pickle.load(f)

    diagrams = data["diagrams"]
    n = len(diagrams)

    print(f"Computing Betti curves for {n} diagrams")
    print(f"Input:  {diagrams_path}")
    print(f"Output: {out_path}")

    betti_curves = []
    for i, dgm in enumerate(diagrams):
        betti_curves.extend(compute_betti_curves([dgm], maxdim=maxdim, n_steps=n_steps))
    print()

    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "betti_curves": betti_curves,   # list of {"t": ..., "betti": ...}
                "labels": data["labels"],
                "label_names": data["label_names"],
                "filtration_params": data.get("filtration_params"),
                "config": data.get("config"),
                "n_steps": n_steps,
                "maxdim": maxdim,
            },
            f,
        )
    print(f"Saved {len(betti_curves)} Betti curves -> {out_path}")


def main():
    run_betti_pipeline()

if __name__ == "__main__":
    main()