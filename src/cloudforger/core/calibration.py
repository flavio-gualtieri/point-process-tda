# src/cloudforger/core/calibration.py

from __future__ import annotations

from .diagram import PersistenceDiagram

import numpy as np


def calibrate(
    diagrams: list[PersistenceDiagram],
    percentiles: tuple[float, ...] = (95.0, 99.9),
    homology_dims: tuple[int, ...] = (0, 1),
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

    result: dict[int, dict[str, dict[float, float]]] = {}
    for dim in homology_dims:
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


def diagram_stats(
        diagrams: list[PersistenceDiagram],
        homology_dim: int,
) -> np.ndarray:
    rows = []
    for d in diagrams:
        pairs = d.finite_pairs(homology_dim)
        if len(pairs) == 0:
            continue
        b, pers = pairs[:, 0],  pairs[:, 1] - pairs[:, 0]
        rows.append((b.min(), b.max(), pers.max(), pers.sum(), len(pairs)))

    return np.asarray(rows) if rows else np.empty((0, 5))


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


""" def axis_bounds(
    stats: dict, dim: int, degenerate_birth_frac: float = 0.25
) -> tuple[float, float]:
    axes = stats.get(dim)
    if axes is None:
        return 1.0, 1.0
    birth_hi = axes["birth"].get(99.9, 1.0)
    persistence_hi = axes["persistence"].get(99.9, 1.0)
    persistence_hi = 1.0 if persistence_hi <= 0 else persistence_hi
    birth_hi = degenerate_birth_frac * persistence_hi if birth_hi <= 0 else birth_hi
    return float(birth_hi), float(persistence_hi) """


def axis_bounds(diagrams, homology_dim, coverage: float = 0.99, pad: float = 1.05):
    """Bounds containing the FULL support of `coverage` of diagrams."""
    s = diagram_stats(diagrams, homology_dim)
    if len(s) == 0:
        return (0.0, 1.0), (0.0, 1.0)
    q = 100.0 * coverage
    birth_lo = float(np.percentile(s[:, 0], 100.0 - q))
    birth_hi = float(pad * np.percentile(s[:, 1], q))
    pers_hi  = float(pad * np.percentile(s[:, 2], q))
    if birth_hi <= birth_lo:          # degenerate axis — don't fabricate one
        raise ValueError(f"H{homology_dim} birth axis is degenerate; use a 1-D vectorizer")
    return (birth_lo, birth_hi), (0.0, pers_hi)

def bifiltration_grid(
    clouds,
    bifiltration,
    resolution: int = 50,
    coverage: float = 0.95,
    n_sample: int | None = 500,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose the shared computation grid for a bifiltration, from TRAIN clouds only.

    Unlike axis_bounds (which sizes a picture to already-computed diagrams),
    this runs BEFORE any persistence computation, because the grid determines
    which module is computed: a grid coarser than the smallest feature scale
    rounds fine structure away entirely, silently.

    Axis 0 (geometric scale) is fixed by the bifiltration's own radius cutoff,
    so the complex and the grid cannot disagree about what the axis means.
    Axis 1 is set by pooled percentiles of the vertex function -- pooled, not
    per-cloud extrema, matching what the coverage sweep showed for images.

    Cheap: touches only the vertex function, never builds a complex.
    """
    if not clouds:
        raise ValueError("bifiltration_grid needs at least one cloud.")
    if n_sample is not None and len(clouds) > n_sample:
        idx = np.random.default_rng(seed).choice(len(clouds), n_sample, replace=False)
        clouds = [clouds[i] for i in idx]

    values = np.concatenate([np.asarray(bifiltration.vertex_function(c)).ravel()
                             for c in clouds])
    q = 100.0 * coverage
    lo, hi = np.percentile(values, [100.0 - q, q])
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError(f"Degenerate vertex-function range: ({lo}, {hi}).")

    thresh = float(bifiltration.params["threshold_radius"])
    return (np.linspace(0.0, thresh, resolution),
            np.linspace(float(lo), float(hi), resolution))


def check_grid_resolves(grid, cluster_scale_min: float) -> None:
    step = float(grid[0][1] - grid[0][0])
    if step > cluster_scale_min:
        raise ValueError(
            f"Grid step {step:.5f} exceeds the smallest cluster scale "
            f"{cluster_scale_min:.5f}; fine structure will be rounded away. "
            f"Need resolution >= {int(np.ceil(grid[0][-1] / cluster_scale_min))}."
        )