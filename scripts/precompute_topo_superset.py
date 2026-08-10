#!/usr/bin/env python3
# scripts/precompute_topo_superset.py
"""Precompute a SUPERSET of topological features for later channel selection.

Pure precomputation -- this script does not decide which channels feed any
downstream model. It exists because the DTM `k` neighbor-count knob drifts
with `parent_intensity` across the new variable-N dataset (nested_thomas):
a fixed integer k means a very different mass fraction of each cloud
depending on N, so this script re-parameterizes DTM by mass fraction
m = k/N (k recomputed per cloud from its own N) and sweeps m across a
log-spaced grid from fine (matches the existing k=5,10,15 regime) to coarse
(unsampled parent-scale territory). It also adds one superlevel-KDE
filtration per cloud (density-peaks merging via a lower-star filtration on
a sparse neighbor-radius Rips graph) as a cheap additional channel to
consider later.

Diagrams are stored RAW (H0 + H1), not persistence images -- sigma/resolution
are downstream, decoupled concerns tuned after m-channel selection.

Before running the full sweep, a handful of clouds are checked at the two
extreme m values (0.01, 0.90) for degenerate (empty or fully collapsed)
diagrams; the script aborts with an explanation if every sampled cloud is
degenerate at an extreme, rather than silently caching garbage.

Usage:
    python scripts/precompute_topo_superset.py configs/runs/nested_thomas_multik_pi_multik_fusion.yaml
    python scripts/precompute_topo_superset.py configs/runs/foo.yaml --thin --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from gudhi import RipsComplex
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.config import load_config
from cloudforger.core.cloud import PointCloud
from cloudforger.core.diagram import PersistenceDiagram
from cloudforger.core.io import dump_pickle, load_pickle
from cloudforger.core.records import diagram_to_record, to_pointcloud
from cloudforger.data_generation.filtration import DTMFiltration
from cloudforger.paths import DEFAULT_DATA_ROOT, DataPaths

# Log-spaced mass-fraction grid: fine end (0.01-0.15) covers the current
# working k=5,10,15 regime; coarse end (0.20-0.90) explores unsampled
# parent-scale territory. See task step 2.
M_VALUES_FULL = [0.01, 0.02, 0.04, 0.07, 0.10, 0.15, 0.20, 0.30, 0.45, 0.65, 0.90]

# Fallback grid if compute/storage forces a cut: keep the fine cluster
# dense, thin the coarse end to ~5 points. Select with --thin.
M_VALUES_THIN = [0.01, 0.02, 0.04, 0.07, 0.10, 0.15, 0.20, 0.35, 0.50, 0.70, 0.90]

MAXDIM = 1  # H0 + H1
DTM_Q = 2.0

# Superlevel-KDE filtration is built on a sparse neighbor-radius Rips graph
# (not the full complex) so it stays cheap regardless of N; k_neighbors sets
# that locality scale the same way DTM's own k does.
KDE_K_NEIGHBORS = 15
KDE_EDGE_MULT = 2.0

SANITY_N_CLOUDS = 5
SANITY_SEED = 0


# ---------------------------------------------------------------------------
# DTM by mass fraction
# ---------------------------------------------------------------------------


def _k_from_m(n_points: int, m: float) -> int:
    return int(np.clip(round(m * n_points), 1, max(1, n_points - 1)))


def _dtm_diagram_for_cloud(cloud: PointCloud, m: float, *, q: float = DTM_Q, maxdim: int = MAXDIM) -> PersistenceDiagram:
    k = _k_from_m(cloud.n_points, m)
    filtration = DTMFiltration(maxdim=maxdim, k=k, q=q, thresh=None)
    diagram = filtration.compute(cloud)
    diagram.filtration_name = "dtm_mfrac"
    diagram.filtration_params = {**diagram.filtration_params, "m": m, "k": k}
    return diagram


def _compute_dtm_for_records(records: list[dict], m: float, maxdim: int, tag: str) -> list[PersistenceDiagram]:
    diagrams = []
    for i, rec in enumerate(records):
        cloud = to_pointcloud(rec)
        diagrams.append(_dtm_diagram_for_cloud(cloud, m, maxdim=maxdim))
        if (i + 1) % 500 == 0 or i + 1 == len(records):
            print(f"  [{tag}] {i + 1}/{len(records)} diagrams computed", flush=True)
    return diagrams


# ---------------------------------------------------------------------------
# Superlevel-KDE filtration (density-peaks merging)
# ---------------------------------------------------------------------------


def _typical_edge_length(points: np.ndarray, k_neighbors: int) -> float:
    n = points.shape[0]
    k = min(k_neighbors, n - 1)
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1)  # column 0 is self (distance 0)
    return float(np.median(dists[:, -1]))


def _kde_superlevel_diagram(
    cloud: PointCloud, *, k_neighbors: int = KDE_K_NEIGHBORS, edge_mult: float = KDE_EDGE_MULT, maxdim: int = MAXDIM
) -> PersistenceDiagram:
    points = np.asarray(cloud.points, dtype=float)
    density = gaussian_kde(points.T)(points.T)
    vertex_values = -density  # superlevel(density) == sublevel(-density) -> lower-star filtration

    edge_length = edge_mult * _typical_edge_length(points, k_neighbors)
    rips = RipsComplex(points=points, max_edge_length=edge_length)
    st = rips.create_simplex_tree(max_dimension=maxdim + 1)

    for simplex, _ in list(st.get_simplices()):
        st.assign_filtration(simplex, float(np.max(vertex_values[simplex])))
    st.make_filtration_non_decreasing()
    st.compute_persistence()

    diagrams: dict[int, np.ndarray] = {}
    for dim in range(maxdim + 1):
        pairs = st.persistence_intervals_in_dimension(dim)
        if pairs.size == 0:
            pairs = np.empty((0, 2), dtype=float)
        diagrams[dim] = np.asarray(pairs, dtype=float)

    return PersistenceDiagram(
        diagrams=diagrams,
        generator_name=cloud.generator_name,
        generator_params=cloud.generator_params,
        seed=cloud.seed,
        filtration_name="kde_superlevel",
        filtration_params={
            "k_neighbors": k_neighbors,
            "edge_mult": edge_mult,
            "max_edge_length": edge_length,
            "maxdim": maxdim,
        },
    )


def _compute_kde_for_records(records: list[dict], maxdim: int, tag: str) -> list[PersistenceDiagram]:
    diagrams = []
    for i, rec in enumerate(records):
        cloud = to_pointcloud(rec)
        diagrams.append(_kde_superlevel_diagram(cloud, maxdim=maxdim))
        if (i + 1) % 500 == 0 or i + 1 == len(records):
            print(f"  [{tag}] {i + 1}/{len(records)} diagrams computed", flush=True)
    return diagrams


# ---------------------------------------------------------------------------
# Degeneracy sanity check (task step 5)
# ---------------------------------------------------------------------------


def _is_degenerate(diagram: PersistenceDiagram, maxdim: int, eps: float = 1e-9) -> bool:
    """True if every homology dim's finite pairs are either absent or have
    ~zero persistence (birth == death), i.e. empty or fully collapsed."""
    for dim in range(maxdim + 1):
        finite = diagram.finite_pairs(dim)
        if finite.size == 0:
            continue
        if np.any((finite[:, 1] - finite[:, 0]) > eps):
            return False
    return True


def _sanity_check_dtm(records: list[dict], m_values: list[float], maxdim: int) -> None:
    rng = np.random.default_rng(SANITY_SEED)
    idx = rng.choice(len(records), size=min(SANITY_N_CLOUDS, len(records)), replace=False)
    extremes = (min(m_values), max(m_values))
    print(f"[sanity] checking m={extremes} on {len(idx)} sampled clouds ...")

    degenerate_count = {m: 0 for m in extremes}
    for i in idx:
        cloud = to_pointcloud(records[int(i)])
        for m in extremes:
            diagram = _dtm_diagram_for_cloud(cloud, m, maxdim=maxdim)
            n0 = diagram.finite_pairs(0).shape[0]
            n1 = diagram.finite_pairs(1).shape[0]
            degenerate = _is_degenerate(diagram, maxdim)
            flag = " DEGENERATE" if degenerate else ""
            print(
                f"  [sanity] cloud#{i} (N={cloud.n_points}) m={m:.2f} k={diagram.filtration_params['k']} "
                f"H0_finite={n0} H1_finite={n1}{flag}"
            )
            if degenerate:
                degenerate_count[m] += 1

    for m in extremes:
        if degenerate_count[m] == len(idx):
            raise RuntimeError(
                f"[sanity] every sampled cloud produced a degenerate (empty/collapsed) DTM diagram at "
                f"m={m:.2f}. Aborting rather than caching garbage -- adjust the m range (task step 2/5), "
                f"or pass --skip-sanity to override."
            )
    print("[sanity] OK -- extreme m values are not degenerate.")


# ---------------------------------------------------------------------------
# Bundle assembly / IO
# ---------------------------------------------------------------------------


def _build_bundle(diagrams: list[PersistenceDiagram], records: list[dict], extra: dict) -> dict:
    label_names = list(records[0]["params"].keys())
    labels = np.array([[r["params"][k] for k in label_names] for r in records], dtype=float)
    bundle = {
        "diagrams": [diagram_to_record(d) for d in diagrams],
        "labels": labels,
        "label_names": label_names,
        "seeds": [r["seed"] for r in records],
        "params": [r["params"] for r in records],
        "n_points": [r["n_points"] for r in records],
        "process": records[0].get("process", ""),
    }
    bundle.update(extra)
    return bundle


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="RunConfig YAML path (used for process name / data_root only)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="path.to.field=value")
    parser.add_argument("--thin", action="store_true", help="use the ~5-point thinned coarse grid instead of the full 11-point m sweep")
    parser.add_argument("--force", action="store_true", help="recompute even if outputs already exist")
    parser.add_argument("--skip-sanity", action="store_true", help="skip the extreme-m (0.01/0.90) degeneracy sanity check")
    parser.add_argument("--no-adversarial", action="store_true", help="skip the adversarial_clouds.pkl split even if present")
    parser.add_argument("--maxdim", type=int, default=MAXDIM, help="max homology dimension (default: 1, i.e. H0+H1)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=args.overrides)
    data_paths = DataPaths(cfg.process.name, root=cfg.data_root or DEFAULT_DATA_ROOT)
    out_dir = data_paths.process_dir / "topo_superset"

    m_values = M_VALUES_THIN if args.thin else M_VALUES_FULL
    print(f"m grid ({'thinned' if args.thin else 'full'}): {m_values}")

    sources = [("train", data_paths.clouds(), False)]
    if not args.no_adversarial and data_paths.clouds(adversarial=True).exists():
        sources.append(("adversarial", data_paths.clouds(adversarial=True), True))

    for tag, clouds_path, is_adv in sources:
        if not clouds_path.exists():
            raise FileNotFoundError(f"{clouds_path} not found -- run scripts/generate.py first.")

        records = load_pickle(clouds_path)
        print(f"[{tag}] {len(records)} clouds from {clouds_path}")

        if not is_adv and not args.skip_sanity:
            _sanity_check_dtm(records, m_values, args.maxdim)

        prefix = "adversarial_" if is_adv else ""

        for m in m_values:
            out_path = out_dir / f"{prefix}dtm_m{m:.2f}_diagrams.pkl"
            if out_path.exists() and not args.force:
                print(f"  [{tag}] {out_path} exists; skipping (pass --force to recompute).")
                continue
            diagrams = _compute_dtm_for_records(records, m, args.maxdim, f"{tag} dtm m={m:.2f}")
            bundle = _build_bundle(diagrams, records, extra={"m": m, "filtration_name": "dtm_mfrac"})
            dump_pickle(out_path, bundle)
            print(f"  [{tag}] saved -> {out_path}")

        kde_out_path = out_dir / f"{prefix}kde_superlevel_diagrams.pkl"
        if kde_out_path.exists() and not args.force:
            print(f"  [{tag}] {kde_out_path} exists; skipping (pass --force to recompute).")
        else:
            diagrams = _compute_kde_for_records(records, args.maxdim, f"{tag} kde_superlevel")
            bundle = _build_bundle(diagrams, records, extra={"filtration_name": "kde_superlevel"})
            dump_pickle(kde_out_path, bundle)
            print(f"  [{tag}] saved -> {kde_out_path}")

    print(f"\nDone. Cached superset under {out_dir}")
    print("Channel selection (which m / whether to use kde_superlevel) happens in a later step.")


if __name__ == "__main__":
    main()
