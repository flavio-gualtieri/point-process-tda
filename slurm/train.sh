#!/bin/bash

#SBATCH -J train
# The "compute" partition has no GPUs (GRES=null), and the "gpu" partition only admits the
# pilot_gpu/its-research accounts, so --gres=gpu:1 there fails at submit time with "Requested node
# configuration is not available" / "Invalid account or account/partition combination". Our GPU
# access is the "sae" partition via the pilot_sae_gpu account (a100-80gb/h100/h200/l40s nodes);
# it is not the default account, so -A has to be named explicitly.
#SBATCH -A pilot_sae_gpu
#SBATCH -p sae
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err

# One array task per (task, features, variant) of the full matrix, via scripts/train.py; each task
# loops over all 10 seeds in-process, over both feature arms x {5-way classification, parameters
# for each family}:
#   PH         6 filtrations x {H0, H1, H0+H1}            = 18 per task
#   classical  4 curve specs (see CURVES below)          =  4 per task
# = 132 array tasks x 10 seeds = 1320 runs.
#
#   bash slurm/train.sh                     # print the run list and the --array range, submit nothing
#   sbatch --array=0-131%20 slurm/train.sh  # %20 caps how many run at once
#
# Any axis can be narrowed at submit time instead of edited, and setting FILTRATIONS or CURVES to
# the empty string drops that arm entirely:
#   TASKS="classify" FILTRATIONS="dtm_k10" CURVES="" SEEDS="1 2 3" bash slurm/train.sh
#
# Resumable: a seed whose run.json exists is skipped, so a re-submit only fills the gaps (the
# features are still built once for whatever seeds remain).
#
# Sizing: images are held in memory, ~1.6 GB per 2-D (filtration, homology dim) channel over the
# 100k classification patterns and a fifth of that per family, plus ~1 min each to rasterize.
# Rips/alpha H0 are 1-D and negligible. GPU: scripts/train.py uses CUDA when it sees it.

set -euo pipefail

# TASKS, not GROUPS: bash keeps a special GROUPS variable (the caller's group ids) and would
# silently ignore the assignment.
TASKS="${TASKS:-classify params:poisson params:thomas params:nested params:matern2 params:lgcp}"
FILTRATIONS="${FILTRATIONS-rips alpha_diameter dtm_k5 dtm_k10 dtm_k15 dtm_k20}"   # PH arm
DIMS="${DIMS:-0 1 0,1}"
# Classical arm: one self-contained spec per entry, NAME@grid per function (unqualified names take
# scripts/train.py's --grid default, sqrtn_u2). Per-function grids exist because F and G are distance
# CDFs: on the fixed r axis they sit saturated at 1.0 over 62%/66% of their 512 samples, against
# 24%/30% on the sqrt(n) axis, while L is only comparable to the literature on the fixed one.
# L alone is the VIHRS feature set.
CURVES="${CURVES-L@fixed,F,G,J L,F,G,J L@fixed,F@fixed,G@fixed,J@fixed L@fixed}"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"

runs=()
for entry in $TASKS; do
  task="${entry%%:*}"
  family="${entry#*:}"
  [ "$family" = "$entry" ] && family=""          # "classify" carries no family
  head="--task $task ${family:+--family $family}"
  for filtration in $FILTRATIONS; do
    for dims in $DIMS; do
      runs+=("$head --filtration $filtration --dims $dims")
    done
  done
  for curves in $CURVES; do
    runs+=("$head --curves $curves")
  done
done

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  printf '%s\n' "${runs[@]}"
  echo "${#runs[@]} array tasks x $(echo $SEEDS | wc -w) seeds -> sbatch --array=0-$(( ${#runs[@]} - 1 ))%20 slurm/train.sh"
  exit 0
fi

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

run="${runs[$SLURM_ARRAY_TASK_ID]}"
echo "Host: $(hostname)  Job ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}  ${run}  seeds: $SEEDS"

# One process for all the seeds, not one per seed: scripts/train.py --seed accepts a list and builds
# the features once, so an array task rasterizes its images once instead of ten times.
# shellcheck disable=SC2086
python -u scripts/train.py $run --seed "$(echo $SEEDS | tr ' ' ',')"
