#!/bin/bash
#SBATCH -J diagrams_compute_k5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=128G
#SBATCH -t 03:00:00
#SBATCH --array=0-59
#SBATCH --output=logs/diagrams_compute_k5_%A_%a.out
#SBATCH --error=logs/diagrams_compute_k5_%A_%a.err

# STAGE 2 of 3 -- k=5 variant. Reuses the clouds.group<NNNN>.pkl /
# adversarial_clouds.group<NNNN>.pkl chunks already on disk under
# data/params/2d/thomas/_chunks/ (they don't depend on filtration k,
# only on --n-groups, which is unchanged), so no split rerun is needed.
# Writes diagrams.group<NNNN>.pkl / adversarial_diagrams.group<NNNN>.pkl.
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
  --config configs/params/processing/new_features.yaml \
  --process thomas \
  --dimension 2 \
  --n-groups 60 \
  --group-id "${SLURM_ARRAY_TASK_ID}" \
  --splits train_test adversarial \
  --keep-cloud-chunks
