#!/bin/bash
# Two follow-ups to the original encoder/fusion sweep:
#   1. Completes the plain baseline pi_multik run (9 remaining seeds).
#   2. Reruns all 6 encoder/fusion combos with include_entropy=true (the
#      persistence-entropy auxiliary variables), 5 seeds each -- 30 GPU
#      tasks. Run-tags get an "_aux" suffix vs. the original sweep's tags.
# 40 GPU tasks total. NOT run automatically by anything -- run it yourself
# from the point-process-tda repo root:
#   bash slurm/submit_pi_multik_followups.sh
# or submit the sbatch files individually.

set -euo pipefail
cd "$(dirname "$0")/.."

sbatch slurm/pi_multik_baseline_complete.sh

sbatch slurm/pi_multik_sweep_shared_concat_aux.sh
sbatch slurm/pi_multik_sweep_shared_conv_avg_aux.sh
sbatch slurm/pi_multik_sweep_shared_conv_flatten_aux.sh
sbatch slurm/pi_multik_sweep_independent_concat_aux.sh
sbatch slurm/pi_multik_sweep_independent_conv_avg_aux.sh
sbatch slurm/pi_multik_sweep_independent_conv_flatten_aux.sh
