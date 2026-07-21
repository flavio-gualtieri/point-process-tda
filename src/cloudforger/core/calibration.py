# src/cloudforger/core/calibration.py

from __future__ import annotations

from .diagram import PersistenceDiagram

import numpy as np


def calibrate(
    diagrams: list[PersistenceDiagram],
    percentiles: tuple[float, ...] = (95.0, 99.9),
) -> dict[int, dict[str, dict[float, float]]]:
    """Collect birth and persistence statistics across a list of diagrams.

    Returns a nested dict keyed by dimension, then "birth" / "persistence",
    then percentile value. Use the results to calibrate feature/vectorizer
    ranges (see features.calibrated / vectorizers.calibrated), independent
    of which filtration produced the diagrams.

    Example::

        stats = calibrate(diagrams)
        # stats[1]["persistence"][99.0] -> suggested upper bound for pers_range
    """
    all_dims: set[int] = set()
    for d in diagrams:
        all_dims.update(d.dimensions())

    result: dict[int, dict[str, dict[float, float]]] = {}
    for dim in sorted(all_dims):
        births_list: list[np.ndarray] = []
        pers_list: list[np.ndarray] = []
        for d in diagrams:
            pairs = d.finite_pairs(dim)
            if len(pairs) == 0:
                continue
            births_list.append(pairs[:, 0])
            pers_list.append(pairs[:, 1] - pairs[:, 0])

        if not births_list:
            continue

        births = np.concatenate(births_list)
        perss = np.concatenate(pers_list)

        result[dim] = {
            "birth": {p: float(np.percentile(births, p)) for p in percentiles},
            "persistence": {p: float(np.percentile(perss, p)) for p in percentiles},
        }

    return result


def calibrate_report(
    diagrams: list[PersistenceDiagram],
    percentiles: tuple[float, ...] = (95.0, 99.9),
) -> str:
    """Return a human-readable calibration report and suggested imager kwargs."""
    stats = calibrate(diagrams, percentiles)
    lines: list[str] = [f"Calibration over {len(diagrams)} diagrams\n{'=' * 42}"]
    for dim, axes in stats.items():
        lines.append(f"\nH{dim}")
        for axis, pvals in axes.items():
            row = "  " + axis + ":  " + "  ".join(
                f"p{p:g}={v:.4f}" for p, v in sorted(pvals.items())
            )
            lines.append(row)
        # Suggest imager kwargs based on the upper percentile.
        b_lo = axes["birth"].get(min(percentiles), 0.0)
        b_hi = axes["birth"][max(percentiles)]
        p_hi = axes["persistence"][max(percentiles)]
        lines.append(
            f"  → birth_range=({b_lo:.4f}, {b_hi:.4f}), "
            f"pers_range=(0.0, {p_hi:.4f})"
        )
    return "\n".join(lines)


def axis_bounds(
    stats: dict, dim: int, degenerate_birth_frac: float = 0.25
) -> tuple[float, float]:
    axes = stats.get(dim)
    if axes is None:
        return 1.0, 1.0
    birth_hi = axes["birth"].get(99.9, 1.0)
    persistence_hi = axes["persistence"].get(99.9, 1.0)
    persistence_hi = 1.0 if persistence_hi <= 0 else persistence_hi
    birth_hi = degenerate_birth_frac * persistence_hi if birth_hi <= 0 else birth_hi
    return float(birth_hi), float(persistence_hi)