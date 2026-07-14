#!/bin/bash

#SBATCH -J compare
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH -t 00:30:00
#SBATCH --output=logs/compare_%j.out
#SBATCH --error=logs/compare_%j.err

# Submit from the point-process-tda repo root: sbatch dtm_experiment/run_compare.sh
# Runs dtm_experiment/compare.py, which loads results.pt for each
# feature/seed under dtm_experiment/results, prints loss summaries and
# paired comparisons, and writes training_curves.png / loss_distribution.png
# to dtm_experiment/results/summary. Requires train.py to have already
# produced results for at least one feature.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unset}"

python dtm_experiment/compare.py
