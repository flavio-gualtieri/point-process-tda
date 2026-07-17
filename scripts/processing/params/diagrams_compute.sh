#!/bin/bash
#SBATCH -J diagrams_compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=128G
#SBATCH -t 03:00:00
#SBATCH --array=0-59
#SBATCH --output=logs/diagrams_compute_%A_%a.out
#SBATCH --error=logs/diagrams_compute_%A_%a.err

# STAGE 2 of 3 -- array job, one task per group.
# --array=0-39  MUST match  --n-groups 40  (last index = N-1).
# Each task reads its pre-split chunk and computes that group's diagrams into
# data/params/2d/thomas/_chunks/diagrams_dtm_k10.group<NNNN>.pkl
# Resumable: re-submitting only recomputes groups whose partial is missing.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

# Keep diagram computation single-threaded (one core per task).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

python scripts/processing/params/diagrams_slurm.py \
  --config configs/params/processing/new_features_k10.yaml \
  --process thomas \
  --dimension 2 \
  --n-groups 60 \
  --group-id "${SLURM_ARRAY_TASK_ID}" \
  --splits train_test adversarial \
  --keep-cloud-chunks