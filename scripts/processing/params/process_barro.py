# scripts/processing/params/process_barro.py
#
# End-to-end BCI/Barro Colorado Island processing:
#   1. Extract raw tree coordinates from .rdata censuses (via R) and
#      normalize them into point-cloud pickles.
#   2. Compute persistence diagrams from those clouds.
#   3. Compute betti_0 curves from those diagrams.

from __future__ import annotations

import argparse
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from cloudforger.core.betti import BettiCurveFeature
from cloudforger.core.cloud import PointCloud
from cloudforger.core.region import Box
from cloudforger.core.diagram import PersistenceDiagram
from cloudforger.tda.features.betti_curve import BettiCurve
from cloudforger.tda.filtration.rips import RipsFiltration

# Nominal BCI plot window in metres (x by y).
DEFAULT_WIN = (1000.0, 500.0)
TRAIN_MAX_N = 9110  # densest training cloud, for the OOD warning

# Mean point count across data/params/2d/thomas/clouds.pkl, the training set
# this real-data crop is meant to be density-comparable to.
TRAIN_TARGET_N = 900

R_EXTRACT = r"""
args <- commandArgs(trailingOnly = TRUE)
infile <- args[1]; outfile <- args[2]; status_keep <- args[3]
e <- new.env(); load(infile, envir = e)
d <- get(ls(e)[1], envir = e)
if (!all(c("gx", "gy", "status") %in% names(d)))
    stop("expected columns gx, gy, status not found in ", infile)
keep <- !is.na(d$gx) & !is.na(d$gy)
if (nzchar(status_keep) && status_keep != "ALL")
    keep <- keep & !is.na(d$status) & d$status == status_keep
write.table(d[keep, c("gx", "gy")], outfile, sep = ",",
            row.names = FALSE, col.names = FALSE)
"""


# ---------------------------------------------------------------------------
# Step 1: extract + normalize raw censuses into point-cloud pickles
# ---------------------------------------------------------------------------

def check_rscript() -> str:
    rscript = shutil.which("Rscript")
    if rscript is None:
        sys.exit(
            "Error: 'Rscript' not found on PATH.\n"
            "These .rdata files need R (pyreadr cannot read them).\n"
            "  Debian/Ubuntu: sudo apt-get install r-base-core\n"
            "  macOS:         brew install r"
        )
    return rscript


def extract_raw_points(rscript: str, infile: Path, status_keep: str) -> np.ndarray:
    """Return raw (N, 2) coordinates in metres for the kept trees."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        csv_path = Path(tmp.name)
    try:
        proc = subprocess.run(
            [rscript, "-e", R_EXTRACT, str(infile), str(csv_path), status_keep],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"R failed on {infile.name}:\n{proc.stderr.strip()}")
        pts = np.loadtxt(csv_path, delimiter=",", ndmin=2)
        if pts.size == 0:
            pts = pts.reshape(0, 2)
        return np.ascontiguousarray(pts, dtype=np.float64)
    finally:
        csv_path.unlink(missing_ok=True)


def normalize(pts: np.ndarray, mode: str, win: tuple[float, float]):
    """Map raw metric coordinates into the training domain.

    Returns (scaled_points, region_low, region_high) with region arrays as
    float64 to match the training schema exactly.
    """
    wx, wy = win
    if mode == "none":
        low, high = (0.0, 0.0), (wx, wy)
        out = pts.copy()
    elif mode == "unit":  # anisotropic -> fills [0,1]^2, region == training
        out = pts / np.array([wx, wy], dtype=np.float64)
        low, high = (0.0, 0.0), (1.0, 1.0)
    elif mode == "isotropic":  # single scale -> preserves geometry
        s = 1.0 / max(wx, wy)
        out = pts * s
        low, high = (0.0, 0.0), (wx * s, wy * s)
    else:
        raise ValueError(f"unknown normalize mode: {mode}")
    return (np.ascontiguousarray(out, dtype=np.float64),
            np.asarray(low, dtype=np.float64),
            np.asarray(high, dtype=np.float64))


def build_cloud(pts: np.ndarray, low: np.ndarray, high: np.ndarray) -> dict:
    """Assemble one cloud dict in the exact training schema (labels = None)."""
    return {
        "points": pts,                       # (N, 2) float64, points as rows
        "params": None,                      # no labels on real data
        "process": None,                     # unknown / to be inferred
        "seed": None,                        # not applicable to observed data
        "n_points": int(pts.shape[0]),
        "dimension": 2,
        "region": {"low": low, "high": high},
    }


def extract_clouds(
    data_dir: Path,
    out_dir: Path | None,
    glob: str,
    status: str,
    normalize_mode: str,
    win: tuple[float, float],
    suffix: str,
    combined: Path | None,
) -> None:
    out_dir = out_dir or data_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rscript = check_rscript()

    files = sorted(data_dir.glob(glob))
    if not files:
        sys.exit(f"No files matching '{glob}' in {data_dir}")

    print(f"normalize={normalize_mode}  window={win} m  status={status}\n")

    all_clouds = []
    for f in files:
        raw = extract_raw_points(rscript, f, status)
        pts, low, high = normalize(raw, normalize_mode, win)
        cloud = build_cloud(pts, low, high)
        all_clouds.append(cloud)

        flag = "  <-- OOD (>%d)" % TRAIN_MAX_N if cloud["n_points"] > TRAIN_MAX_N else ""
        print(f"{f.name:22s}  N={cloud['n_points']:>7d}  "
              f"region [{low[0]:.2f},{low[1]:.2f}]-[{high[0]:.2f},{high[1]:.2f}]  "
              f"x[{pts[:,0].min():.3f},{pts[:,0].max():.3f}] "
              f"y[{pts[:,1].min():.3f},{pts[:,1].max():.3f}]{flag}")

        if combined is None:
            # Same container type as training: a LIST of cloud dicts.
            out_path = out_dir / (f.stem + suffix)
            with open(out_path, "wb") as fh:
                pickle.dump([cloud], fh, protocol=pickle.HIGHEST_PROTOCOL)

    if combined is not None:
        with open(combined, "wb") as fh:
            pickle.dump(all_clouds, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\nWrote {len(all_clouds)} clouds -> {combined}")
    else:
        print(f"\nWrote {len(files)} files (each a 1-element list) to {out_dir}")

    if any(c["n_points"] > TRAIN_MAX_N for c in all_clouds):
        print(f"\nWARNING: point counts far exceed the training max "
              f"({TRAIN_MAX_N}). Predictions are extrapolation; consider "
              f"tiling the plot or subsampling before inference.")


# ---------------------------------------------------------------------------
# Step 2 & 3: persistence diagrams and betti_0 curves from clouds
# ---------------------------------------------------------------------------

def load_cloud(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    # extract_clouds() writes one cloud per file as a 1-element list,
    # matching the training container schema.
    if isinstance(data, list):
        data = data[0]
    return data


def to_pointcloud(cloud: Any) -> PointCloud:
    if not isinstance(cloud, dict):
        return cloud

    region = None
    reg = cloud.get("region")
    if reg is not None:
        try:
            region = Box(
                low=np.asarray(reg["low"], dtype=float),
                high=np.asarray(reg["high"], dtype=float),
            )
        except Exception:
            region = None

    return PointCloud(
        points=np.asarray(cloud["points"]),
        generator_name=cloud.get("process") or "",
        generator_params=dict(cloud.get("params") or {}),
        seed=cloud.get("seed"),
        region=region,
    )


def sample_unit_box_window(
        point_cloud: PointCloud,
        target_n: int = TRAIN_TARGET_N,
        rng: np.random.Generator | None = None,
) -> PointCloud:
    """Crop a random square window sized to hold ~target_n points, rescaled to (0,1)^d.

    The raw cloud is far denser than the training clouds it's compared against
    (e.g. the BCI census has ~235k points vs. a training mean of ~900), so a
    naive clip to the existing (0,1) region is a no-op and blows up Rips.
    Instead we pick a window whose area matches the training density, crop it,
    and rescale it onto the unit box.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    region = point_cloud.region
    if region is None:
        low = point_cloud.points.min(axis=0)
        high = point_cloud.points.max(axis=0)
        region = Box(low=low, high=high)

    extent = region.high - region.low
    density = point_cloud.n_points / region.volume
    window = float(np.sqrt(target_n / density))
    window = min(window, float(extent.min()))

    slack = extent - window
    corner = region.low + rng.uniform(0.0, 1.0, size=extent.shape) * slack

    mask = np.all(
        (point_cloud.points >= corner) & (point_cloud.points <= corner + window),
        axis=1,
    )
    rescaled = (point_cloud.points[mask] - corner) / window

    return PointCloud(
        points=rescaled,
        generator_name=point_cloud.generator_name,
        generator_params=point_cloud.generator_params,
        seed=point_cloud.seed,
        region=Box(low=np.zeros(point_cloud.dimension), high=np.ones(point_cloud.dimension)),
    )


def diagram_to_record(diagram: PersistenceDiagram) -> dict[str, Any]:
    return {
        "diagrams": {
            int(dim): np.asarray(pairs)
            for dim, pairs in diagram.diagrams.items()
        },
        "params": dict(diagram.generator_params),
        "seed": diagram.seed,
        "process": diagram.generator_name,
        "filtration": diagram.filtration_name,
        "filtration_params": dict(diagram.filtration_params),
    }


def compute_diagrams(
        cloud_path: Path,
        out_path: Path,
        filtration: RipsFiltration, process: str,
        diagram_format: str = "dict"
) -> None:
    cloud = load_cloud(cloud_path)
    if isinstance(cloud, dict) and not cloud.get("process"):
        cloud = {**cloud, "process": process}

    print(f"  Computing persistence diagram from {cloud_path} ...")
    point_cloud = to_pointcloud(cloud)
    point_cloud = sample_unit_box_window(point_cloud)
    computed = filtration.compute(point_cloud)
    print(f"\r    Done", end="")
    print()

    diagram = diagram_to_record(computed) if diagram_format == "dict" else computed

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "diagrams": diagram,
                "labels": None,
                "label_names": None,
                "params": None,
                "seeds": None,
                "process": process,
                "filtration_params": filtration.params,
            },
            f,
        )

    print(f"  Saved diagram → {out_path}")
    return computed


def betti_features_to_record(curve: BettiCurveFeature) -> dict[str, Any]:
    return {
        "curves": {
            int(dim): np.asarray(values)
            for dim, values in curve.curves.items()
        },
        "vector": curve.vector(),
        "params": dict(curve.generator_params),
        "seed": curve.seed,
        "process": curve.generator_name,
        "filtration": curve.filtration_name,
        "filtration_params": dict(curve.filtration_params),
        "feature": curve.feature_name,
        "feature_params": dict(curve.feature_params),
    }


def compute_curves(
        diagram: PersistenceDiagram,
        out_path: Path,
        process: str,
        curve_format: str = "dict"
) -> None:
    betti = BettiCurve(
        homology_dims=0,
    )
    print(f"  Computing betti_0 curve ...")

    computed = betti.compute(diagram)
    print(f"\r    Done", end="")
    print()

    curve = betti_features_to_record(computed) if curve_format == "dict" else computed

    betti0_curve = computed.curves[0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "curve": curve,
                "betti0_curve": betti0_curve,
                "params": dict(computed.generator_params),
                "seed": computed.seed,
                "process": process,
                "betti_params": betti.params,
            },
            f,
        )

    print(f"  Saved betti_0 curve → {out_path}")


def diagrams_and_curves(data_dir: Path, n_censuses: int = 8) -> None:
    for census in range(1, n_censuses + 1):
        cloud_path = data_dir / f"bci.tree{census}_cloud.pkl"
        diagram_path = data_dir / f"bci.tree{census}_diagram.pkl"
        betti0_path = data_dir / f"bci.tree{census}_betti0.pkl"
        filtration = RipsFiltration(maxdim=1, thresh=None)
        diagram = compute_diagrams(
            cloud_path=cloud_path,
            out_path=diagram_path,
            filtration=filtration,
            process="barro",
        )
        compute_curves(
            diagram=diagram,
            out_path=betti0_path,
            process="barro",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data/barro"))
    p.add_argument("--out-dir", type=Path, default=None,
                   help="default: same as --data-dir")
    p.add_argument("--glob", default="bci.tree*.rdata")
    p.add_argument("--status", default="A",
                   help="status to keep, or ALL (default: A = alive)")
    p.add_argument("--normalize", choices=["isotropic", "unit", "none"],
                   default="isotropic",
                   help="coordinate mapping into the training domain "
                        "(default: isotropic; see module docstring)")
    p.add_argument("--win-x", type=float, default=DEFAULT_WIN[0],
                   help="plot width in metres (default: 1000)")
    p.add_argument("--win-y", type=float, default=DEFAULT_WIN[1],
                   help="plot height in metres (default: 500)")
    p.add_argument("--suffix", default="_cloud.pkl",
                   help="per-census output suffix (default: _cloud.pkl)")
    p.add_argument("--combined", type=Path, default=None,
                   help="if set, write ALL censuses as one list to this path "
                        "instead of one file per census")
    p.add_argument("--skip-diagrams", action="store_true",
                   help="only run cloud extraction, skip diagrams/betti curves")
    args = p.parse_args()

    out_dir = args.out_dir or args.data_dir
    win = (args.win_x, args.win_y)

    # Step 1: rdata -> cloud pickles
    extract_clouds(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        glob=args.glob,
        status=args.status,
        normalize_mode=args.normalize,
        win=win,
        suffix=args.suffix,
        combined=args.combined,
    )

    if args.skip_diagrams:
        return

    # Steps 2 & 3: cloud pickles -> persistence diagrams -> betti_0 curves
    diagrams_and_curves(out_dir)


if __name__ == "__main__":
    main()
