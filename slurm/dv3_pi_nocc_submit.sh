#!/bin/bash
# slurm/dv3_pi_nocc_submit.sh -- submit the no-CoordConv persistence-image
# cells (slurm/dv3_pi_nocc_train.sh): parameter estimation first (12 tasks,
# one GPU each, all 10 seeds at once), classification after it (3 tasks).
# Not an sbatch script: run it on the login node from the repo root.
#
#   bash slurm/dv3_pi_nocc_submit.sh            # everything, DTM k=5
#   bash slurm/dv3_pi_nocc_submit.sh params     # parameter estimation only
#   DTM_K=10 bash slurm/dv3_pi_nocc_submit.sh   # another filtration, once its diagrams exist
#
# Classification waits for every parameter task to END (afterany), so it never
# takes one of the 12 GPUs (the sae per-user cap) the parameter tasks use, and
# a failed seed does not block it (re-submit that task with --array=<id>).

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

K="${DTM_K:-5}"
mode="${1:-all}"
case "$mode" in all|params) ;; *) echo "usage: bash $0 [all|params]" >&2; exit 2 ;; esac

params=$(DTM_K="$K" PAR=10 sbatch --parsable --array=0-11 --export=ALL slurm/dv3_pi_nocc_train.sh)
echo "parameter estimation: ${params} (tasks 0-11: h0_nocc_norm, h1_nocc_norm, h0_nocc x 4 families)" >&2
if [[ "$mode" == all ]]; then
  clf=$(DTM_K="$K" PAR=5 sbatch --parsable --array=12-14 --time=03:00:00 --dependency="afterany:${params}" \
                              --export=ALL slurm/dv3_pi_nocc_train.sh)
  echo "classification:       ${clf} (tasks 12-14, after ${params} ends)" >&2
fi
echo "watch:  squeue -u \$USER -n dv3_pi_nocc" >&2
