# src/cloudforger/tda/calibration.py

from __future__ import annotations

from ..core.diagram import PersistenceDiagram

import numpy as np


def calibrate(
    diagrams: list[PersistenceDiagram],
    percentiles: tuple[float, ...] = (95.0, 99.0),
) -> dict[int, dict[str, dict[float, float]]]:
    """Collect birth and persistence statistics across a list of diagrams.

    Returns a nested dict keyed by dimension, then "birth" / "persistence",
    then percentile value.  Use the results to set ``birth_range`` and
    ``pers_range`` on :class:`PersistenceImager`.

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
    percentiles: tuple[float, ...] = (95.0, 99.0),
) -> str:
    """Return a human-readable calibration report and suggested imager kwargs."""
    stats = calibrate(diagrams, percentiles)
    lines: list[str] = [f"Calibration over {len(diagrams)} diagrams\n{'=' * 42}"]
    for dim, axes in stats.items():
        lines.append(f"\nH{dim}")
        for axis, pvals in axes.items():
            row = "  " + axis + ":  " + "  ".join(
                f"p{int(p)}={v:.4f}" for p, v in sorted(pvals.items())
            )
            lines.append(row)
        # Suggest imager kwargs based on 99th percentile.
        b_lo = axes["birth"].get(min(percentiles), 0.0)
        b_hi = axes["birth"][max(percentiles)]
        p_hi = axes["persistence"][max(percentiles)]
        lines.append(
            f"  → birth_range=({b_lo:.4f}, {b_hi:.4f}), "
            f"pers_range=(0.0, {p_hi:.4f})"
        )
    return "\n".join(lines)