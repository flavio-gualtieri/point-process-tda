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
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err

# One array task per (task, features, variant, seed chunk) of the experiment matrix, via
# scripts/train.py. Use slurm/matrix.sh to submit the whole thing; this script is the worker and is
# also what you run bare to inspect the run list.
#
#   bash slurm/train.sh                     # print the run list and the --array range, submit nothing
#   bash slurm/matrix.sh                    # print every sbatch the matrix needs
#   bash slurm/matrix.sh submit             # actually submit them
#
# Axes, all overridable at submit time so nothing here has to be edited; setting FILTRATIONS,
# CURVES or VARIANTS to the empty string drops that arm entirely:
#
#   TASKS        classify + one params run per family
#   FILTRATIONS  the PH arm's filtrations (one array entry each; "a,b,c" is the multi-k arm)
#   DIMS         homology dimensions per filtration
#   VARIANTS     how the PH arm's diagrams become a vector: image (rasterize) or perslay (learn it)
#   CURVES       the classical arm, one self-contained spec per entry
#   SEEDS        seeds per run
#   SEED_CHUNK   seeds per array task (see "Walltime" below)
#
#   TASKS="classify" FILTRATIONS="dtm_k10" CURVES="" SEEDS="1 2 3" bash slurm/train.sh
#
# Both PH variants are one axis rather than two submissions: they write to h<dims> and
# perslay_h<dims> respectively, so a filtration's two vectorizations sit side by side under the
# same `features` directory and scripts/regimes.py compares them by --reference like any two runs.
#
# Resumable, and split-aware: scripts/train.py skips a seed whose run.json records the CURRENT
# cloudforger.simulation.split and RETRAINS one that records an older split, printing "[stale]".
# That matters here -- the results/ layout does not mention the split, so widening the test block
# leaves every path identical and only changes what predictions.npz means. A re-submit after a
# split change therefore refills the matrix on its own; no --force and no manual clean-out.
#
# Memory, at the 50k-theta bank (100k patterns per family, so 500k for classification). Every
# pattern is held, not just the split being trained on, and build() peaks at ~3x one channel while
# it stacks and sqrt-transforms:
#
#              per (filtration, dim) channel      classify peak, 1 filtration x H0+H1
#   image      7.6 GB classify / 1.5 GB params    ~31 GB
#   perslay    5.7 GB classify / 1.1 GB params    ~17 GB   (no sqrt copy)
#   classical  1.0 GB per curve                   ~4 GB
#
# so --mem=64G covers every classification run above and is 4x what a params run needs; matrix.sh
# drops it to 16G for the params arrays. The multi-k arm multiplies the image figures by the number
# of filtrations (~92 GB for three at H0+H1) and needs its own --mem if you add it to FILTRATIONS.
#
# Walltime: the bank is 5x its previous size, so an epoch is ~5x longer (370k training patterns for
# classification against 70k before). scripts/train.py has no mid-run checkpoint, so a timeout
# loses the whole array task -- which is why SEED_CHUNK defaults to 2 rather than putting all ten
# seeds in one process. The features are rebuilt once per chunk, a few minutes against hours of
# training, so the old reason to keep the seeds together no longer pays for the risk.

set -euo pipefail

# TASKS, not GROUPS: bash keeps a special GROUPS variable (the caller's group ids) and would
# silently ignore the assignment.
TASKS="${TASKS:-classify params:poisson params:thomas params:nested params:matern2 params:lgcp}"
FILTRATIONS="${FILTRATIONS-rips alpha_diameter dtm_k5 dtm_k10 dtm_k15 dtm_k20}"   # PH arm
DIMS="${DIMS:-0 1 0,1}"
VARIANTS="${VARIANTS-image perslay}"           # PH arm: rasterized, learned
# Classical arm: one self-contained spec per entry, NAME@grid per function (unqualified names take
# scripts/train.py's --grid default, sqrtn_u2). Per-function grids exist because F and G are distance
# CDFs: on the fixed r axis they sit saturated at 1.0 over 62%/66% of their 512 samples, against
# 24%/30% on the sqrt(n) axis, while L is only comparable to the literature on the fixed one.
# L alone is the VIHRS feature set.
CURVES="${CURVES-L@fixed,F,G,J L,F,G,J L@fixed,F@fixed,G@fixed,J@fixed L@fixed}"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
SEED_CHUNK="${SEED_CHUNK:-2}"

# Seeds split into chunks of SEED_CHUNK, each its own array task.
chunks=()
chunk=""
count=0
for seed in $SEEDS; do
  chunk="${chunk:+$chunk,}$seed"
  count=$(( count + 1 ))
  if [ "$count" -eq "$SEED_CHUNK" ]; then chunks+=("$chunk"); chunk=""; count=0; fi
done
# `[ -n "$chunk" ] && chunks+=(...)` would return 1 on an exact division and set -e would kill the
# script there -- which the default SEEDS/SEED_CHUNK do.
if [ -n "$chunk" ]; then chunks+=("$chunk"); fi

runs=()
for entry in $TASKS; do
  task="${entry%%:*}"
  family="${entry#*:}"
  [ "$family" = "$entry" ] && family=""          # "classify" carries no family
  head="--task $task ${family:+--family $family}"
  for filtration in $FILTRATIONS; do
    for dims in $DIMS; do
      for variant in $VARIANTS; do
        [ "$variant" = "perslay" ] && flag="--perslay" || flag=""
        for c in "${chunks[@]}"; do
          runs+=("$head --filtration $filtration --dims $dims $flag --seed $c")
        done
      done
    done
  done
  for curves in $CURVES; do
    for c in "${chunks[@]}"; do
      runs+=("$head --curves $curves --seed $c")
    done
  done
done

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  printf '%s\n' "${runs[@]}"
  echo "${#runs[@]} array tasks (${#chunks[@]} seed chunk(s) of up to $SEED_CHUNK)" \
       "-> sbatch --array=0-$(( ${#runs[@]} - 1 ))%20 slurm/train.sh"
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
echo "Host: $(hostname)  Job ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}  ${run}"

# shellcheck disable=SC2086
python -u scripts/train.py $run
