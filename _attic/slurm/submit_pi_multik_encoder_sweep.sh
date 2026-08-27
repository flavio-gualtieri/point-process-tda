#!/bin/bash
# Submits all 6 encoder/fusion combinations of the pi_multik sweep plus the
# vihrs comparison baseline (nested_thomas, dtm k=5,10,15 for pi_multik;
# 5 seeds each via job arrays -- 35 GPU tasks total). NOT run automatically
# by anything -- run it yourself from the point-process-tda repo root:
#   bash slurm/submit_pi_multik_encoder_sweep.sh
# or submit the sbatch files individually.

set -euo pipefail
cd "$(dirname "$0")/.."

sbatch slurm/pi_multik_sweep_shared_concat.sh
sbatch slurm/pi_multik_sweep_shared_conv_avg.sh
sbatch slurm/pi_multik_sweep_shared_conv_flatten.sh
sbatch slurm/pi_multik_sweep_independent_concat.sh
sbatch slurm/pi_multik_sweep_independent_conv_avg.sh
sbatch slurm/pi_multik_sweep_independent_conv_flatten.sh
sbatch slurm/vihrs_nested_thomas.sh
