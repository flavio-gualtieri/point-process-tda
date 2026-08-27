#!/bin/bash
# Submits the full PI calibration sweep: 4 parameters (coverage q, padding
# factor pad, sigma_pixels, resolution) x 2 processes (thomas,
# nested_thomas) x DTM k=5,10,15 x 3 seeds -- see each
# slurm/pi_multik_calib_<param>_<process>.sh for the per-sweep grid and
# results-path layout. NOT run automatically by anything -- run it yourself
# from the point-process-tda repo root:
#   bash slurm/submit_pi_multik_calib_sweep.sh
# or submit the sbatch files individually.

set -euo pipefail
cd "$(dirname "$0")/.."

for param in q pad sigma resolution; do
  for process in thomas nested_thomas; do
    sbatch "slurm/pi_multik_calib_${param}_${process}.sh"
  done
done
