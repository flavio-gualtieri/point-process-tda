# dtm_experiment/compute_features_delete.py
#
# One-off script: computes only the new weighted-betti-curve feature
# (weight_by_persistence=True) for the k5 DTM diagrams. All other features
# (betti curves, persistence images, persistence entropy) already exist on
# disk from a prior run of compute_features.py -- this script does not
# recompute them.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "processing" / "params"))

from cloudforger.tda.calibration import calibrate, calibrate_report
from cloudforger.tda.features.betti_curve import BettiCurve
from cloudforger.core.diagram import PersistenceDiagram

from pipeline_lib.records import load_diagrams
from pipeline_lib.io import dump_pickle

DATA_DIR = ROOT / "data" / "params" / "2d" / "thomas"

HOMOLOGY_DIMS = [0, 1]
BETTI_GRID_SIZE = 512          # matches the Rips-based betti_cnn resolution fix
PI_SIGMA = 0.05
DEGENERATE_BIRTH_SIGMA_MULT = 4.0   # same fallback as pipeline_lib/images.py::build_imagers
GRID_RANGE_PAD = 1.1                # small safety margin over the calibrated range

REQUIRED_BUNDLE_KEYS = ("labels", "label_names", "seeds", "params")


def _axis_bounds(stats: dict, dim: int) -> tuple[float, float]:
    """99th-percentile birth/persistence upper bounds for one homology dim,
    with the same degenerate-birth fallback used for Rips H0 (every bar has
    birth==0). DTM shouldn't hit this in practice -- DTM births vary with
    local density -- but the guard costs nothing and keeps this robust."""
    axes = stats.get(dim)
    if axes is None:
        return 1.0, 1.0
    birth_hi = axes["birth"].get(99.0, 1.0)
    persistence_hi = axes["persistence"].get(99.0, 1.0)
    birth_hi = DEGENERATE_BIRTH_SIGMA_MULT * PI_SIGMA if birth_hi <= 0 else birth_hi
    persistence_hi = 1.0 if persistence_hi <= 0 else persistence_hi
    return float(birth_hi), float(persistence_hi)


def build_calibrated_betti(stats: dict, weight_by_persistence: bool = False) -> dict[int, BettiCurve]:
    betti_by_dim = {}
    for dim in HOMOLOGY_DIMS:
        birth_hi, pers_hi = _axis_bounds(stats, dim)
        grid_range = (0.0, (birth_hi + pers_hi) * GRID_RANGE_PAD)
        betti_by_dim[dim] = BettiCurve(
            homology_dims=(dim,), grid_size=BETTI_GRID_SIZE, grid_range=grid_range,
            drop_infinite=True, normalize=False,
            weight_by_persistence=weight_by_persistence,
        )
    return betti_by_dim


def _validate_bundle(bundle: dict, diagrams_path: Path) -> None:
    missing = [k for k in REQUIRED_BUNDLE_KEYS if k not in bundle]
    if missing:
        raise KeyError(
            f"{diagrams_path} is missing {missing!r} -- expected the same schema "
            "pipeline_lib.diagrams.compute_diagrams_from_clouds produces "
            "(labels/label_names/seeds/params/process alongside 'diagrams')."
        )


def _is_broken(diagram: PersistenceDiagram) -> bool:
    """A diagram with zero H0 bars is topologically impossible for any
    non-empty point cloud (even isolated points register an essential H0
    bar) -- this shows up for a small, n_points-correlated but not
    n_points-deterministic fraction of clouds under k=10 DTM (see
    docs/diagrams_summary.md's filtration_params and the investigation that
    found this: failure rate ~100% at n_points=10 down to ~0% only by
    n_points~=33, not a clean cutoff), so checking diagram content directly
    is the only reliable way to catch it."""
    h0 = diagram.diagrams.get(0)
    return h0 is None or len(h0) == 0


def _filter_broken_diagrams(
    diagrams: list[PersistenceDiagram], bundle: dict[str, Any], tag: str
) -> tuple[list[PersistenceDiagram], dict[str, Any]]:
    keep_mask = np.array([not _is_broken(d) for d in diagrams], dtype=bool)
    n_dropped = int((~keep_mask).sum())
    if n_dropped:
        print(
            f"  [{tag}] dropping {n_dropped}/{len(diagrams)} clouds with empty H0 diagrams "
            "(DTM k-NN degeneracy for sparse clouds -- not fixable here, only filterable)"
        )

    filtered_diagrams = [d for d, keep in zip(diagrams, keep_mask) if keep]
    filtered_bundle = dict(bundle)
    filtered_bundle["labels"] = np.asarray(bundle["labels"])[keep_mask]
    filtered_bundle["seeds"] = [s for s, keep in zip(bundle["seeds"], keep_mask) if keep]
    filtered_bundle["params"] = [p for p, keep in zip(bundle["params"], keep_mask) if keep]
    return filtered_diagrams, filtered_bundle


def compute_weighted_betti_for_split(
    diagrams: list[PersistenceDiagram],
    bundle: dict[str, Any],
    betti_weighted_by_dim: dict[int, BettiCurve],
    betti_weighted_out: Path,
    tag: str,
) -> None:
    n = len(diagrams)

    betti_weighted_matrices = {dim: np.empty((n, BETTI_GRID_SIZE), dtype=np.float32) for dim in HOMOLOGY_DIMS}

    for i, d in enumerate(diagrams):
        for dim in HOMOLOGY_DIMS:
            betti_weighted_matrices[dim][i] = betti_weighted_by_dim[dim].compute(d).curves[dim]
        if (i + 1) % 1000 == 0 or i + 1 == n:
            print(f"\r    [{tag}] {i + 1}/{n}", end="", flush=True)
    print()

    dump_pickle(betti_weighted_out, {
        "labels": bundle["labels"],
        "label_names": bundle["label_names"],
        "params": bundle["params"],
        "seeds": bundle["seeds"],
        "process": bundle.get("process", ""),
        "betti_params": {dim: b.params for dim, b in betti_weighted_by_dim.items()},
        "betti0_matrix": betti_weighted_matrices[0],
        "betti1_matrix": betti_weighted_matrices[1],
    })
    print(f"  [{tag}] saved weighted betti curves -> {betti_weighted_out}")


def main() -> None:
    diagrams_path = DATA_DIR / "diagrams_dtm_k5.pkl"
    adv_diagrams_path = DATA_DIR / "adversarial_diagrams_dtm_k5.pkl"

    if not diagrams_path.exists():
        raise FileNotFoundError(
            f"{diagrams_path} not found -- this script expects the DTM diagrams "
            "to already be downloaded there."
        )

    print(f"Loading {diagrams_path} ...")
    diagrams, bundle = load_diagrams(diagrams_path)
    _validate_bundle(bundle, diagrams_path)
    print(f"  {len(diagrams)} diagrams loaded.")

    diagrams, bundle = _filter_broken_diagrams(diagrams, bundle, tag="train_test")
    print(f"  {len(diagrams)} diagrams remain after filtering.")

    print("\nCalibrating birth/persistence ranges from the train_test diagrams ...")
    stats = calibrate(diagrams)
    print(calibrate_report(diagrams))

    betti_weighted_by_dim = build_calibrated_betti(stats, weight_by_persistence=True)

    print("\nComputing train_test weighted betti curves ...")
    compute_weighted_betti_for_split(
        diagrams, bundle, betti_weighted_by_dim,
        DATA_DIR / "betti_weighted_dtm_k5.pkl",
        tag="train_test",
    )

    if adv_diagrams_path.exists():
        print(f"\nLoading {adv_diagrams_path} ...")
        adv_diagrams, adv_bundle = load_diagrams(adv_diagrams_path)
        _validate_bundle(adv_bundle, adv_diagrams_path)
        print(f"  {len(adv_diagrams)} adversarial diagrams loaded.")

        adv_diagrams, adv_bundle = _filter_broken_diagrams(adv_diagrams, adv_bundle, tag="adversarial")
        print(f"  {len(adv_diagrams)} adversarial diagrams remain after filtering.")

        print("Computing adversarial weighted betti curves (reusing train_test calibration) ...")
        compute_weighted_betti_for_split(
            adv_diagrams, adv_bundle, betti_weighted_by_dim,
            DATA_DIR / "adversarial_betti_weighted_dtm_k5.pkl",
            tag="adversarial",
        )
    else:
        print(f"\nNo adversarial diagrams found at {adv_diagrams_path}; skipping adversarial features.")

    print("\nDone.")


if __name__ == "__main__":
    main()
