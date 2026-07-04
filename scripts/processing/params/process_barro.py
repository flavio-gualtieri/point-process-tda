# scripts/processing/params/process_barro.py

from __future__ import annotations

import argparse
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# Nominal BCI plot window in metres (x by y).
DEFAULT_WIN = (1000.0, 500.0)
TRAIN_MAX_N = 9110  # densest training cloud, for the OOD warning

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
    args = p.parse_args()

    out_dir = args.out_dir or args.data_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rscript = check_rscript()
    win = (args.win_x, args.win_y)

    files = sorted(args.data_dir.glob(args.glob))
    if not files:
        sys.exit(f"No files matching '{args.glob}' in {args.data_dir}")

    print(f"normalize={args.normalize}  window={win} m  status={args.status}\n")

    all_clouds = []
    for f in files:
        raw = extract_raw_points(rscript, f, args.status)
        pts, low, high = normalize(raw, args.normalize, win)
        cloud = build_cloud(pts, low, high)
        all_clouds.append(cloud)

        flag = "  <-- OOD (>%d)" % TRAIN_MAX_N if cloud["n_points"] > TRAIN_MAX_N else ""
        print(f"{f.name:22s}  N={cloud['n_points']:>7d}  "
              f"region [{low[0]:.2f},{low[1]:.2f}]-[{high[0]:.2f},{high[1]:.2f}]  "
              f"x[{pts[:,0].min():.3f},{pts[:,0].max():.3f}] "
              f"y[{pts[:,1].min():.3f},{pts[:,1].max():.3f}]{flag}")

        if args.combined is None:
            # Same container type as training: a LIST of cloud dicts.
            out_path = out_dir / (f.stem + args.suffix)
            with open(out_path, "wb") as fh:
                pickle.dump([cloud], fh, protocol=pickle.HIGHEST_PROTOCOL)

    if args.combined is not None:
        with open(args.combined, "wb") as fh:
            pickle.dump(all_clouds, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\nWrote {len(all_clouds)} clouds -> {args.combined}")
    else:
        print(f"\nWrote {len(files)} files (each a 1-element list) to {out_dir}")

    if any(c["n_points"] > TRAIN_MAX_N for c in all_clouds):
        print(f"\nWARNING: point counts far exceed the training max "
              f"({TRAIN_MAX_N}). Predictions are extrapolation; consider "
              f"tiling the plot or subsampling before inference.")


if __name__ == "__main__":
    main()