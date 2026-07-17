#!/bin/bash

#SBATCH -J compute_features
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH -t 04:00:00
#SBATCH --output=logs/compute_features_%j.out
#SBATCH --error=logs/compute_features_%j.err

# Submit from the point-process-tda repo root: sbatch dtm_experiment/run_compute_features.sh
# Runs dtm_experiment/compute_features.py, which reads
# data/params/2d/thomas/diagrams_dtm_k5.pkl (+ adversarial_diagrams_dtm_k5.pkl,
# if present) and writes betti_dtm_k5.pkl / images_dtm_k5.pkl (+ adversarial_*)
# alongside them. Single process, no CLI args, not resumable -- reruns
# recompute both splits from scratch.

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

python dtm_experiment/compute_features.py
