#!/usr/bin/env python3
# scripts/run_pi_calibration_sweep.py
"""Reproduce the PI (pi_multik) persistence-image calibration sweeps for the
paper writeup: one axis at a time (coverage, pad, sigma_pixels, resolution),
DTM k=5/10/15, H0+H1, holding every other calibration knob at the base
config's default -- ablation style, matching this repo's diagnostic-notebook
convention (see notebooks/pi_sigma_diagnostics.ipynb / memory
feedback-pi-sigma-methodology).

This is a thin driver over scripts/train.py: pi_multik fits its persistence-
image calibration (resolution/sigma_pixels/pd_calibration_coverage/pad) per
seed, on that seed's train split only (src/cloudforger/experiments/pi_multik/
pi_multik.py) -- so there is nothing to precompute here. Each sweep point is
just one train.py invocation with a `--set method.params.<key>=<value>`
override and a `--run-tag` so it lands in its own results/ subdirectory
instead of overwriting the base run:

    results/<process>/dtm_k5-k10-k15.../pi_multik/_runs/cal_<axis><value>/seed_<seed>/

Every run also appends a provenance row to results/experiments.jsonl (see
scripts/train.py's docstring), and results/**/*.pt|*.png stay untracked
(.gitignore) -- only results.json/results.jsonl are meant to be committed
alongside the writeup.

Usage:
    # full sweep (both processes x 4 axes x 3 seeds) -- see --dry-run first
    python scripts/run_pi_calibration_sweep.py --dry-run
    python scripts/run_pi_calibration_sweep.py

    # one process, one axis (e.g. re-running just the coverage axis after a fix)
    python scripts/run_pi_calibration_sweep.py --process nested_thomas --axes coverage

    # single SLURM-array-friendly point (one process/axis/value/seed)
    python scripts/run_pi_calibration_sweep.py --process thomas --axes pad \\
        --pad 1.10 --seeds 9371

Reading results back out afterwards (scripts/evaluate.py's method@run_tag
syntax, one call per axis):
    python scripts/evaluate.py configs/runs/nested_thomas/pi_multik.yaml \\
        --methods pi_multik@cal_coverage0.9 pi_multik@cal_coverage0.95 \\
                  pi_multik@cal_coverage0.99 pi_multik@cal_coverage1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = ROOT / "scripts" / "train.py"

# process -> base RunConfig. Both already have DTM k=5,10,15 / H0+H1
# persistence_image + pi_multik configured, diagrams already cached
# (data/<process>/dtm_k{5,10,15}/diagrams.pkl), so this script only ever
# touches method.params via --set -- no featurization needed first.
BASE_CONFIGS = {
    "nested_thomas": ROOT / "configs/runs/nested_thomas/pi_multik.yaml",
    "thomas": ROOT / "configs/runs/thomas/thomas_pi_multik_k5k10k15.yaml",
}

# axis -> (method.params key, default grid). Matches the calibration knobs
# build_calibrated_imager exposes (src/cloudforger/vectorization/
# persistence_images/calibrated.py): coverage is axis_bounds' quantile,
# pad its padding factor, sigma_pixels/resolution the imager's own params.
AXES: dict[str, tuple[str, list[float]]] = {
    "coverage": ("method.params.pd_calibration_coverage", [0.90, 0.95, 0.99, 1.00]),
    "pad": ("method.params.pad", [1.00, 1.05, 1.10]),
    "sigma_pixels": ("method.params.sigma_pixels", [0.5, 1.0, 2.0]),
    "resolution": ("method.params.resolution", [32, 64, 128]),
}

DEFAULT_SEEDS = [9371, 9372, 9373]


def run_tag(axis: str, value: float) -> str:
    return f"cal_{axis}{value:g}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--process", choices=[*BASE_CONFIGS, "both"], default="both",
        help="which base config(s) to sweep (default: both)",
    )
    parser.add_argument(
        "--axes", nargs="+", choices=list(AXES), default=list(AXES),
        help="which calibration axis/axes to sweep (default: all four)",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help=f"seeds per sweep point (default: {DEFAULT_SEEDS})")
    for axis, (_, default_grid) in AXES.items():
        parser.add_argument(f"--{axis.replace('_', '-')}", type=float, nargs="+", default=default_grid, help=f"{axis} grid (default: {default_grid})")
    parser.add_argument("--force", action="store_true", help="retrain even if a sweep point's results.pt already exists (passed through to train.py)")
    parser.add_argument("--dry-run", action="store_true", help="print the train.py commands without running them")
    args = parser.parse_args(argv)

    grids = {axis: getattr(args, axis) for axis in AXES}
    processes = list(BASE_CONFIGS) if args.process == "both" else [args.process]

    commands: list[list[str]] = []
    for process in processes:
        config = BASE_CONFIGS[process]
        for axis in args.axes:
            param_path, _ = AXES[axis]
            for value in grids[axis]:
                tag = run_tag(axis, value)
                for seed in args.seeds:
                    cmd = [
                        sys.executable, str(TRAIN_PY), str(config),
                        "--seed", str(seed),
                        "--set", f"{param_path}={value:g}",
                        "--run-tag", tag,
                    ]
                    if args.force:
                        cmd.append("--force")
                    commands.append(cmd)

    print(f"{len(commands)} train.py run(s) queued "
          f"({len(processes)} process(es) x {len(args.axes)} axis/axes x seeds={args.seeds}).\n")

    for cmd in commands:
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=False, cwd=ROOT)

    if args.dry_run:
        print("\n(dry run -- nothing executed; drop --dry-run to actually train.)")


if __name__ == "__main__":
    main()
