from __future__ import annotations

import pickle
import shutil
from pathlib import Path

import numpy as np


ROOT = Path("/Users/qp252676/Desktop/point-process-tda/data/params/2d/inhom_thomas")

SPLITS = [
    {
        "name": "train_test",
        "clouds": "clouds.pkl",
        "diagrams": "diagrams.pkl",
        "betti": "betti.pkl",
    },
    {
        "name": "adversarial",
        "clouds": "adversarial_clouds.pkl",
        "diagrams": "adversarial_diagrams.pkl",
        "betti": "adversarial_betti.pkl",
    },
]


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def dump_pickle(path: Path, obj) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def backup_once(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"Backed up {path.name} -> {backup.name}")


def get_items(bundle, key: str):
    if isinstance(bundle, dict) and key in bundle:
        return bundle[key]
    return bundle


def get_seed(record):
    if isinstance(record, dict):
        return record.get("seed")
    return getattr(record, "seed", None)


def get_points(record):
    if isinstance(record, dict):
        return record.get("points")
    return getattr(record, "points", None)


def get_covariates(record):
    if isinstance(record, dict):
        return record.get("covariates")
    return getattr(record, "covariates", None)


def attach_covariates_to_records(records, covariates, n_points, seeds, kind: str) -> None:
    for i, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(
                f"{kind}[{i}] is not a dict record. "
                "This backfill script assumes dict-format outputs."
            )

        if get_seed(record) != seeds[i]:
            raise ValueError(
                f"Seed mismatch in {kind}[{i}]: "
                f"{get_seed(record)=}, expected cloud seed {seeds[i]}"
            )

        record["covariates"] = covariates[i]
        record["n_points"] = int(n_points[i])


def process_split(split: dict[str, str]) -> None:
    print(f"\nProcessing {split['name']}")

    clouds_path = ROOT / split["clouds"]
    diagrams_path = ROOT / split["diagrams"]
    betti_path = ROOT / split["betti"]

    for path in [clouds_path, diagrams_path, betti_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    clouds_bundle = load_pickle(clouds_path)
    diagrams_bundle = load_pickle(diagrams_path)
    betti_bundle = load_pickle(betti_path)

    clouds = get_items(clouds_bundle, "clouds")
    diagrams = get_items(diagrams_bundle, "diagrams")
    curves = get_items(betti_bundle, "curves")

    if not (len(clouds) == len(diagrams) == len(curves)):
        raise ValueError(
            f"Length mismatch for {split['name']}: "
            f"{len(clouds)=}, {len(diagrams)=}, {len(curves)=}"
        )

    seeds = [get_seed(c) for c in clouds]
    covariates = [get_covariates(c) for c in clouds]
    n_points = [
        np.asarray(get_points(c)).shape[0]
        if get_points(c) is not None
        else 0
        for c in clouds
    ]

    missing = [i for i, cov in enumerate(covariates) if cov is None]
    if missing:
        raise ValueError(
            f"{split['name']} has {len(missing)} clouds without covariates. "
            f"First missing index: {missing[0]}"
        )

    attach_covariates_to_records(diagrams, covariates, n_points, seeds, "diagrams")
    attach_covariates_to_records(curves, covariates, n_points, seeds, "curves")

    if isinstance(diagrams_bundle, dict):
        diagrams_bundle["covariates"] = covariates
        diagrams_bundle["n_points"] = np.asarray(n_points, dtype=int)

    if isinstance(betti_bundle, dict):
        betti_bundle["covariates"] = covariates
        betti_bundle["n_points"] = np.asarray(n_points, dtype=int)

    backup_once(diagrams_path)
    backup_once(betti_path)

    dump_pickle(diagrams_path, diagrams_bundle)
    dump_pickle(betti_path, betti_bundle)

    print(f"Updated {diagrams_path}")
    print(f"Updated {betti_path}")


def main() -> None:
    for split in SPLITS:
        process_split(split)


if __name__ == "__main__":
    main()
