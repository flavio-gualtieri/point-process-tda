#!/bin/bash

#SBATCH -J train
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err

# One array task per (filtration, dims, seed) of one task, via scripts/train.py.
#
#   bash slurm/train.sh classify           # print the run list and the --array range, submit nothing
#   sbatch --array=0-89 slurm/train.sh classify
#   sbatch --array=0-89 slurm/train.sh params nested
#
# The matrix is the environment, so a subset is a submit-time choice, not an edit:
#   FILTRATIONS="dtm_k10" DIMS="0,1" SEEDS="1 2 3" bash slurm/train.sh classify
#
# Sizing: images are held in memory, ~1.6 GB per (filtration, homology dim) at resolution 64 over
# 100k patterns (less where H0 is 1-D), plus ~1 min each to rasterize. The multi-k arm
# (dtm_k5,dtm_k10,dtm_k15 with dims 0,1) is the big one at ~10 GB.
#
# GPU: if this cluster has a GPU partition, set PARTITION/GRES below -- scripts/train.py picks CUDA
# up on its own (--device overrides). On CPU, keep --cpus-per-task for torch's thread pool.

set -euo pipefail

TASK="${1:?usage: train.sh classify | params <family>}"
FAMILY="${2:-}"

FILTRATIONS="${FILTRATIONS:-rips alpha dtm_k5 dtm_k10 dtm_k15 dtm_k20 dtm_k5,dtm_k10,dtm_k15}"
DIMS="${DIMS:-0 1 0,1}"
SEEDS="${SEEDS:-1 2 3 4 5}"

runs=()
for filtration in $FILTRATIONS; do
  for dims in $DIMS; do
    for seed in $SEEDS; do
      runs+=("--filtration $filtration --dims $dims --seed $seed")
    done
  done
done

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  printf '%s\n' "${runs[@]}"
  echo "${#runs[@]} runs -> sbatch --array=0-$(( ${#runs[@]} - 1 )) slurm/train.sh $TASK $FAMILY"
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
echo "Host: $(hostname)  Job ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}  ${TASK} ${FAMILY} ${run}"

# shellcheck disable=SC2086
python -u scripts/train.py --task "$TASK" ${FAMILY:+--family "$FAMILY"} $run
