# scripts/processing/generate_clouds.py

import pickle
from pathlib import Path
from math import pi, gamma

import numpy as np

from cloudforger.processes.matern import MaternHardCoreProcess
from cloudforger.processes.poisson import PoissonProcess
from cloudforger.processes.thomas import ThomasProcess
from cloudforger.core.region import Box


CONFIG = {
    "n_clouds_per_class": 500,
    "n_points": 500,
    "dimensions": list(range(2, 17)),  # 2 through 16 inclusive
    "seed_base": 20260519,
    "output_dir": Path("data"),
    "output_file": "clouds.pkl",

    # Calibration settings
    "n_probe": 8,
    "matern_target_ball_vol": 5e-4,
    "thomas_mean_offspring": 10.0,
    "thomas_cluster_scale": 0.05,
}


def unit_box(d: int) -> Box:
    return Box(low=np.zeros(d), high=np.ones(d))


def ball_vol_coeff(d: int) -> float:
    """Volume coefficient of the unit d-ball: vol(B_r) = coeff * r**d."""
    return pi ** (d / 2) / gamma(d / 2 + 1)


def matern_radius(d: int, target_ball_vol: float) -> float:
    """
    Choose r so the hard-core ball has a small fixed volume in every dimension,
    keeping thinning behaviour more comparable across d.
    """
    return (target_ball_vol / ball_vol_coeff(d)) ** (1.0 / d)


def mean_count(process, box: Box, n_probe: int, seed0: int) -> float:
    """
    Estimate the mean number of sampled points.

    For Matérn and Thomas, the public sample(n, region, seed) method accepts n,
    but these processes determine their own count from intensity, so n=0 is fine.
    """
    counts = [
        process.sample(n=0, region=box, seed=seed0 + s).n_points
        for s in range(n_probe)
    ]
    return float(np.mean(counts))


def calibrate_matern(
    d: int,
    box: Box,
    target: int,
    n_probe: int,
    seed0: int,
    target_ball_vol: float,
) -> MaternHardCoreProcess:
    """Tune parent_intensity so mean retained count is approximately target."""
    r = matern_radius(d, target_ball_vol)

    def make(parent_intensity: float) -> MaternHardCoreProcess:
        return MaternHardCoreProcess(
            parent_intensity=float(parent_intensity),
            hardcore_radius=float(r),
        )

    lo, hi = 1.0, 1.0

    while mean_count(make(hi), box, n_probe=n_probe, seed0=seed0) < target:
        lo = hi
        hi *= 1.6
        if hi > 1e6:
            break

    for _ in range(18):
        mid = 0.5 * (lo + hi)
        mc = mean_count(make(mid), box, n_probe=n_probe, seed0=seed0)
        if mc < target:
            lo = mid
        else:
            hi = mid

    return make(0.5 * (lo + hi))


def calibrate_thomas(
    d: int,
    box: Box,
    target: int,
    n_probe: int,
    seed0: int,
    mean_offspring: float,
    cluster_scale: float,
) -> ThomasProcess:
    """Tune parent_intensity so mean clipped count is approximately target."""

    def make(parent_intensity: float) -> ThomasProcess:
        return ThomasProcess(
            parent_intensity=float(parent_intensity),
            mean_offspring=float(mean_offspring),
            cluster_scale=float(cluster_scale),
        )

    lo, hi = 1.0, 1.0

    while mean_count(make(hi), box, n_probe=n_probe, seed0=seed0) < target:
        lo = hi
        hi *= 1.6
        if hi > 1e6:
            break

    for _ in range(18):
        mid = 0.5 * (lo + hi)
        mc = mean_count(make(mid), box, n_probe=n_probe, seed0=seed0)
        if mc < target:
            lo = mid
        else:
            hi = mid

    return make(0.5 * (lo + hi))


def processes_for_dimension(d: int, box: Box, seed0: int) -> dict:
    """Create dimension-specific point processes."""
    return {
        "poisson": PoissonProcess(intensity=CONFIG["n_points"] / box.volume),
        "matern": calibrate_matern(
            d=d,
            box=box,
            target=CONFIG["n_points"],
            n_probe=CONFIG["n_probe"],
            seed0=seed0,
            target_ball_vol=CONFIG["matern_target_ball_vol"],
        ),
        "thomas": calibrate_thomas(
            d=d,
            box=box,
            target=CONFIG["n_points"],
            n_probe=CONFIG["n_probe"],
            seed0=seed0 + 50_000,
            mean_offspring=CONFIG["thomas_mean_offspring"],
            cluster_scale=CONFIG["thomas_cluster_scale"],
        ),
    }


def print_count_summary(clouds, labels, dimensions, label_names):
    labels_arr = np.asarray(labels)
    dims_arr = np.asarray(dimensions)

    for d in np.unique(dims_arr):
        for class_idx, name in enumerate(label_names):
            idx = np.flatnonzero((labels_arr == class_idx) & (dims_arr == d))

            if idx.size == 0:
                print(
                    f"d={d:2d} {name:7s}: no clouds found "
                    f"(check diagnostic placement/bookkeeping)"
                )
                continue

            counts = np.array([clouds[i].n_points for i in idx], dtype=int)

            print(
                f"d={d:2d} {name:7s}: "
                f"mean={counts.mean():.1f}, std={counts.std():.1f}, "
                f"min={counts.min()}, max={counts.max()}"
            )


def main():
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)

    label_names = ["poisson", "matern", "thomas"]
    seed_counter = 0

    for d in CONFIG["dimensions"]:
        box = unit_box(d)

        clouds = []
        labels = []
        dimensions = []

        print(f"Calibrating processes for d={d}...")
        processes = processes_for_dimension(
            d=d,
            box=box,
            seed0=CONFIG["seed_base"] + 1_000_000 * d,
        )

        process_params = {
            name: proc.params for name, proc in processes.items()
        }

        print(f"Generating clouds for d={d}...")
        for class_idx, name in enumerate(label_names):
            proc = processes[name]

            for _ in range(CONFIG["n_clouds_per_class"]):
                seed = CONFIG["seed_base"] + seed_counter
                seed_counter += 1

                cloud = proc.sample(
                    n=CONFIG["n_points"],
                    region=box,
                    seed=seed,
                )

                clouds.append(cloud)
                labels.append(class_idx)
                dimensions.append(d)

        print_count_summary(clouds, labels, dimensions, label_names)

        output_dir = CONFIG["output_dir"] / f"{d}d"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / CONFIG["output_file"]

        payload = {
            "clouds": clouds,
            "labels": np.array(labels, dtype=np.int64),
            "dimensions": np.array(dimensions, dtype=np.int64),
            "dimension": d,
            "label_names": label_names,
            "process_params": process_params,
            "config": {
                k: v
                for k, v in CONFIG.items()
                if k not in {"output_dir"}
            },
        }

        with open(output_path, "wb") as f:
            pickle.dump(payload, f)

        print(
            f"Saved {len(clouds)} clouds "
            f"({len(label_names)} classes, dimension {d}) "
            f"to {output_path}"
        )


if __name__ == "__main__":
    main()