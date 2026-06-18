# scripts/processing/compute_betti.py
import argparse
import os
import pickle
import numpy as np
from pathlib import Path

DEFAULT_DIMS = list(range(2, 18))


def compute_betti_curves(diagrams: list, maxdim: int, n_steps: int = 100) -> list:
    all_finite = []
    for dgm in diagrams:
        for k in range(maxdim + 1):
            arr = dgm.diagrams.get(k)
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

    # ------------------------------------------------------------------ #
    # 2. For every diagram compute the Betti curve                        #
    # ------------------------------------------------------------------ #
    betti_curves = []
    for dgm in diagrams:
        betti = np.zeros((maxdim + 1, n_steps), dtype=np.int32)
        for k in range(maxdim + 1):
            arr = dgm.diagrams.get(k)
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


def compute_betti_for_dim(
    dim: int,
    data_dir: Path,
    maxdim: int = 1,
    n_steps: int = 100,
    overwrite: bool = False,
) -> None:
    diagrams_path = data_dir / f"{dim}d" / "diagrams.pkl"
    out_path = data_dir / f"{dim}d" / "betti.pkl"

    if out_path.exists() and not overwrite:
        print(f"[dim={dim}] Skipping because output already exists: {out_path}")
        return

    if not diagrams_path.exists():
        raise FileNotFoundError(f"Missing input file: {diagrams_path}")

    with open(diagrams_path, "rb") as f:
        data = pickle.load(f)

    diagrams = data["diagrams"]
    n = len(diagrams)

    print(f"[dim={dim}] Computing Betti curves for {n} diagrams")
    print(f"[dim={dim}] Input:  {diagrams_path}")
    print(f"[dim={dim}] Output: {out_path}")

    betti_curves = []
    for i, dgm in enumerate(diagrams):
        betti_curves.extend(compute_betti_curves([dgm], maxdim=maxdim, n_steps=n_steps))
        print(f"\r[dim={dim}] {i + 1}/{n}", end="", flush=True)
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
    print(f"[dim={dim}] Saved {len(betti_curves)} Betti curves -> {out_path}")


def resolve_dim_from_array_index(dims):
    idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if idx < 0 or idx >= len(dims):
        raise IndexError(
            f"SLURM_ARRAY_TASK_ID={idx} out of range for dims={dims}"
        )
    return dims[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--maxdim", type=int, default=1)
    parser.add_argument(
        "--n-steps", type=int, default=100,
        help="Number of points in the filtration grid for Betti curves",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dims = DEFAULT_DIMS

    if args.all:
        for dim in dims:
            compute_betti_for_dim(
                dim=dim,
                data_dir=args.data_dir,
                maxdim=args.maxdim,
                n_steps=args.n_steps,
                overwrite=args.overwrite,
            )
        print("\nDone.")
        return

    if args.dim is not None:
        dim = args.dim
    else:
        dim = resolve_dim_from_array_index(dims)

    compute_betti_for_dim(
        dim=dim,
        data_dir=args.data_dir,
        maxdim=args.maxdim,
        n_steps=args.n_steps,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()