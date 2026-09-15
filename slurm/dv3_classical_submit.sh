#!/bin/bash
# slurm/dv3_classical_submit.sh -- submit the DV3 classical-baseline batch with
# PARAMETER ESTIMATION (inference) on the critical path and classification
# strictly after it. Not an sbatch script: run it on the login node, from the
# repo root. It only calls sbatch.
#
#   bash slurm/dv3_classical_submit.sh              # inference now, classification queued behind it
#   bash slurm/dv3_classical_submit.sh params       # inference only
#   bash slurm/dv3_classical_submit.sh classify [JOBID ...]
#                                                   # classification only, optionally after JOBIDs end
#
# The chain (--parsable ids, all dependencies set at submit time):
#
#   1  prep   tasks 0-11  CPU  params feature caches (4 families x 3 passes, 8 CPUs each)
#   2  train  tasks 0-71  GPU  inference, afterok:1  (18 arms x 4 families x 10 seeds)
#   3  prep   task  12    CPU  _classify bundle + caches, afterany:1 (16 CPUs;
#                              never competes with 1 for CPU slots)
#   4  train  tasks 72-89 GPU  classification, afterok:3 AND afterany:2 -- starts
#                              only once every inference task has ended, so it
#                              never holds one of the 12 GPUs inference could use.
#                              afterany, not afterok, on 2: a failed inference
#                              seed must not block classification (re-submit
#                              the failed inference tasks separately).
#
# If a prep task fails, its dependants stay pending with reason
# DependencyNeverSatisfied: fix, re-run the prep task, then `scancel` them and
# re-submit that part (both stages are resumable).

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

PREP=slurm/dv3_classical_prep.sh
TRAIN=slurm/dv3_classical_train.sh
mode="${1:-all}"

submit_params() {
  local prep train
  prep=$(sbatch --parsable --array=0-11 "$PREP")
  train=$(sbatch --parsable --array=0-71 --dependency="afterok:${prep}" "$TRAIN")
  echo "inference:       prep ${prep} (tasks 0-11)  ->  train ${train} (tasks 0-71)" >&2
  PARAMS_PREP=$prep PARAMS_TRAIN=$train
}

submit_classify() {  # submit_classify <prep-after-dep> <train-after-dep>  (either may be empty)
  local prep train prep_dep=() train_dep="afterok"
  [[ -n "$1" ]] && prep_dep=(--dependency="$1")
  prep=$(sbatch --parsable --array=12 --cpus-per-task=16 --mem=64G "${prep_dep[@]}" "$PREP")
  train_dep="afterok:${prep}${2:+,$2}"
  train=$(sbatch --parsable --array=72-89 --dependency="$train_dep" "$TRAIN")
  echo "classification:  prep ${prep} (task 12)  ->  train ${train} (tasks 72-89, --dependency=${train_dep})" >&2
}

case "$mode" in
  all)
    submit_params
    submit_classify "afterany:${PARAMS_PREP}" "afterany:${PARAMS_TRAIN}"
    ;;
  params)
    submit_params
    ;;
  classify)
    shift
    after=""
    for j in "$@"; do after+="${after:+:}${j}"; done
    submit_classify "" "${after:+afterany:${after}}"
    ;;
  *)
    echo "usage: bash $0 [all|params|classify [JOBID ...]]" >&2; exit 2
    ;;
esac

echo >&2
echo "watch:   squeue -u \$USER -n dv3_cl_prep,dv3_cl_train" >&2
echo "then:    python scripts/dv3_matrix.py      # refresh docs/status/experiment_matrix.xlsx" >&2
