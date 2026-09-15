#!/bin/bash
# slurm/dv3_pi_nocc_submit.sh -- submit the no-CoordConv persistence-image
# cells (slurm/dv3_pi_nocc_train.sh): parameter estimation first (16 tasks,
# one GPU each, all 10 seeds at once), classification after it (4 tasks).
# Cells already on disk are skipped seed by seed, so a full re-submit only
# runs what is missing; `h01` submits just the H0+H1 cell.
# Not an sbatch script: run it on the login node from the repo root.
#
#   bash slurm/dv3_pi_nocc_submit.sh            # everything, DTM k=5
#   bash slurm/dv3_pi_nocc_submit.sh params     # parameter estimation only
#   bash slurm/dv3_pi_nocc_submit.sh h01        # only the H0+H1 cell (tasks 12-15, 19)
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
case "$mode" in
  all)    P_ARRAY=0-15;  C_ARRAY=16-19 ;;
  params) P_ARRAY=0-15;  C_ARRAY= ;;
  h01)    P_ARRAY=12-15; C_ARRAY=19 ;;
  *) echo "usage: bash $0 [all|params|h01]" >&2; exit 2 ;;
esac

params=$(DTM_K="$K" PAR=10 sbatch --parsable --array="$P_ARRAY" --export=ALL slurm/dv3_pi_nocc_train.sh)
echo "parameter estimation: ${params} (tasks ${P_ARRAY})" >&2
if [[ -n "$C_ARRAY" ]]; then
  clf=$(DTM_K="$K" PAR=5 sbatch --parsable --array="$C_ARRAY" --time=03:00:00 --dependency="afterany:${params}" \
                              --export=ALL slurm/dv3_pi_nocc_train.sh)
  echo "classification:       ${clf} (tasks ${C_ARRAY}, after ${params} ends)" >&2
fi
echo "watch:  squeue -u \$USER -n dv3_pi_nocc" >&2
