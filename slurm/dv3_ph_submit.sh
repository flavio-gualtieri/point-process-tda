#!/bin/bash
# slurm/dv3_ph_submit.sh -- submit the DV3 persistence-feature cells for ONE
# DTM filtration: the diagrams they still need, then the PERSISTENCE-IMAGE
# cells first, then Betti curves and landscapes. Not an sbatch script: run it
# on the login node from the repo root; it only calls sbatch.
#
#   bash slurm/dv3_ph_submit.sh               # DTM k=5, every cell, persistence images first
#   bash slurm/dv3_ph_submit.sh pi            # persistence-image cells only
#   bash slurm/dv3_ph_submit.sh params        # parameter estimation only (PI first), no classification
#   DTM_K=10 bash slurm/dv3_ph_submit.sh      # k=10 (also computes its A/B/C diagrams)
#
# Chain (all dependencies set at submit time; the diagram stage is skipped
# when its bundles are already merged):
#   1 dtm        slurm/dv3_ph_dtm_compute.sh  CPU 120x16  diagrams for $SETS (k=5: train only)
#   2 merge      slurm/dv3_ph_dtm_merge.sh    CPU         afterok:1
#   3 PI params  slurm/dv3_ph_train.sh 0-7    GPU x8      afterok:2, all 10 seeds in one wave (PAR=10)
#   4 bundle     slurm/dv3_ph_bundle.sh       CPU         afterok:2   (_classify diagram bundles)
#   5 PI clf     slurm/dv3_ph_train.sh 24-25  GPU x2      afterok:4, PAR=5, --time=10:00:00
#   6 params     slurm/dv3_ph_train.sh 8-23   GPU         afterany:3 AND after:5 (Betti, landscape)
#   7 clf        slurm/dv3_ph_train.sh 26-29  GPU         afterok:4 AND afterany:6
# The 10 PI tasks fit inside the 12-GPU sae cap together. Steps 6-7 can only
# START once every PI parameter task has ended and the PI classification
# tasks have started (after: = "began execution"), so they never take a GPU
# the PI cells are waiting for. afterany, not afterok: a failed seed never
# blocks the next stage (re-submit that task with --array=<id>).

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

K="${DTM_K:-5}"
mode="${1:-all}"
case "$mode" in all|pi|params) ;; *) echo "usage: bash $0 [all|pi|params]" >&2; exit 2 ;; esac
pi_clf=""
if [[ "$K" == 5 ]]; then SETS_K="train"; else SETS_K="train A B C"; fi
declare -A N_GROUPS=([train]=5 [A]=5 [B]=4 [C]=5)

missing_sets=()
for s in $SETS_K; do
  for f in poisson thomas nested matern2 lgcp; do
    [[ "$s" == B && "$f" == poisson ]] && continue
    [[ -f "data/dv3/${s}/${f}/dtm_k${K}/diagrams.pkl" ]] || { missing_sets+=("$s"); break; }
  done
done

dep=""
if (( ${#missing_sets[@]} )); then
  n=0; for s in $SETS_K; do n=$(( n + N_GROUPS[$s] )); done
  # SETS may hold spaces, so it travels in the environment (--export=ALL), not in --export=VAR=...
  comp=$(DTM_K="$K" SETS="$SETS_K" sbatch --parsable --export=ALL slurm/dv3_ph_dtm_compute.sh)
  merge=$(DTM_K="$K" SETS="$SETS_K" sbatch --parsable --array=0-$(( n - 1 )) --dependency=afterok:"$comp" \
                 --export=ALL slurm/dv3_ph_dtm_merge.sh)
  echo "diagrams dtm_k${K} on [${SETS_K}]: compute ${comp} (0-119) -> merge ${merge} (0-$(( n - 1 )))" >&2
  dep="afterok:${merge}"
else
  echo "diagrams dtm_k${K}: all merged already, no diagram jobs" >&2
fi

train() {  # train <array> <dependency> [extra sbatch args] -> job id
  local array="$1" dependency="$2"; shift 2
  sbatch --parsable --array="$array" ${dependency:+--dependency=$dependency} "$@" slurm/dv3_ph_train.sh
}

pi_params=$(DTM_K="$K" PAR=10 train 0-7 "$dep" --export=ALL)
echo "PI parameter estimation: ${pi_params} (tasks 0-7, 10 seeds per GPU)" >&2

if [[ "$mode" != params ]]; then
  bundle=$(DTM_K="$K" sbatch --parsable ${dep:+--dependency=$dep} --export=ALL slurm/dv3_ph_bundle.sh)
  pi_clf=$(DTM_K="$K" PAR=5 train 24-25 "afterok:${bundle}" --time=10:00:00 --export=ALL)
  echo "PI classification:       bundle ${bundle} -> ${pi_clf} (tasks 24-25)" >&2
fi

if [[ "$mode" != pi ]]; then
  params=$(DTM_K="$K" train 8-23 "afterany:${pi_params}${pi_clf:+,after:${pi_clf}}" --export=ALL)
  echo "Betti/landscape params:  ${params} (tasks 8-23, after the PI cells)" >&2
  if [[ "$mode" != params ]]; then
    clf=$(DTM_K="$K" train 26-29 "afterok:${bundle},afterany:${params}" --time=10:00:00 --export=ALL)
    echo "Betti/landscape classify: ${clf} (tasks 26-29, after ${params} ends)" >&2
  fi
fi
echo >&2
echo "watch:  squeue -u \$USER -n dv3_ph_dtm,dv3_ph_dtm_merge,dv3_ph_bundle,dv3_ph_train" >&2
