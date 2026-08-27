#!/bin/bash

#SBATCH -J mincontrast_thomas
#SBATCH -t 08:00:00
#SBATCH --mem-per-cpu=11G
#SBATCH --array=0-4
#SBATCH --output=logs/mincontrast_thomas_%A_%a.out
#SBATCH --error=logs/mincontrast_thomas_%A_%a.err

# "min contrast on K" classical baseline, thomas -- CPU only (scipy.optimize
# Nelder-Mead, no torch/GPU use anywhere in mincontrast.py). Fits every
# cloud in that seed's TEST partition (scripts/train.py's
# run_classical_baseline, dispatched via CLASSICAL_BASELINE_NAMES), 10
# multistart restarts each (method.params.n_starts in
# configs/runs/thomas/mincontrast.yaml).
#
# No GPU directives (-p/-A/--gres) -- mirrors slurm/run_cpu.sh's plain CPU
# convention, not the GPU-job template every pi_multik-family script uses.
#
# Time limit is a rough guess (~700-1000 test clouds x 10 multistart fits
# each, no reference runtime measured yet) -- adjust after watching the
# first array task; each fit() call recomputes the O(n^2) empirical_K
# matrix from scratch per start (fit_multistart's existing, pre-this-batch
# behavior, not specific to this script).
#
# Results land under results/thomas/raw/mincontrast/seed_<seed>/.
#
# 5 seeds via --array=0-4, matching mincontrast.yaml's 5-seed list.
#
# ASSUMES data/thomas/clouds.pkl already exists.
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/mincontrast_thomas.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

SEEDS=(9371 9372 9373 9374 9375)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unset}  Array task: ${SLURM_ARRAY_TASK_ID:-unset}  Seed: $SEED"

python -u scripts/train.py configs/runs/thomas/mincontrast.yaml \
  --seed "$SEED"
