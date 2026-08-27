#!/bin/bash

#SBATCH -J mincontrast_nested_thomas
#SBATCH -t 16:00:00
#SBATCH --mem-per-cpu=11G
#SBATCH --array=0-4
#SBATCH --output=logs/mincontrast_nested_thomas_%A_%a.out
#SBATCH --error=logs/mincontrast_nested_thomas_%A_%a.err

# "min contrast on K" classical baseline, nested Thomas -- CPU only (scipy.
# optimize Nelder-Mead, no torch/GPU use anywhere in mincontrast_nested.py).
# Fits every cloud in that seed's TEST partition (scripts/train.py's
# run_classical_baseline, dispatched via CLASSICAL_BASELINE_NAMES), 10
# multistart restarts each (method.params.n_starts in
# configs/runs/nested_thomas/mincontrast.yaml), against the closed-form K
# of the writeup's \eqref{eq:g-nested} -- fills Table~\ref{tab:classical-
# comparison}'s footnote $^d$ gap. NEW baseline, not yet validated
# end-to-end against ground truth -- see mincontrast_nested.py's module
# docstring; spot-check the first array task's recovered parameters
# before trusting the numbers.
#
# No GPU directives (-p/-A/--gres) -- mirrors slurm/mincontrast_thomas.sh's
# plain CPU convention, not the GPU-job template every pi_multik-family
# script uses.
#
# Time limit is a rough guess, generously above mincontrast_thomas.sh's
# 08:00:00: same per-fit O(n^2) empirical_K cost, but 4 free parameters
# (kappa, mu1, sigma1, sigma2) instead of 2, so Nelder-Mead's simplex needs
# more iterations/evals per fit (mincontrast_nested.fit's maxiter/maxfev
# are 4x mincontrast.fit's) -- adjust after watching the first array task.
#
# Results land under results/nested_thomas/raw/mincontrast_nested/seed_<seed>/.
#
# 5 seeds via --array=0-4, matching mincontrast.yaml's 5-seed list (same
# seeds as thomas's mincontrast_thomas.sh, for a like-for-like row in
# Table~\ref{tab:classical-comparison}).
#
# ASSUMES data/nested_thomas/clouds.pkl already exists (it does).
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/mincontrast_nested_thomas.sh

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

python -u scripts/train.py configs/runs/nested_thomas/mincontrast.yaml \
  --seed "$SEED"
