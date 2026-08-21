#!/bin/bash

#SBATCH -J mincontrast_g_thomas
#SBATCH -t 08:00:00
#SBATCH --mem-per-cpu=11G
#SBATCH --array=0-4
#SBATCH --output=logs/mincontrast_g_thomas_%A_%a.out
#SBATCH --error=logs/mincontrast_g_thomas_%A_%a.err

# "min contrast on G" classical baseline, thomas -- CPU only. Sibling of
# mincontrast_thomas.sh (K-based); see mincontrast_g.py's module docstring
# for the pair-correlation contrast this fits instead. Same
# run_classical_baseline dispatch, same test-partition/multistart
# convention (method.params.n_starts in
# configs/runs/thomas/mincontrast_g.yaml).
#
# empirical_g is new, previously-unvalidated code as of this batch --
# spot-checked standalone (pure numpy, no scipy) against empirical_K
# (internal K/g consistency), a homogeneous Poisson process (g ~= 1), and
# simulated Thomas clusters (tracks g_thomas's closed form within ~3-6%
# at the r values that matter for the fit) before this script was written;
# NOT yet checked end-to-end through fit_multistart's actual optimizer
# (scipy unavailable on the login node -- see
# [[project_python_env_mismatch]]), so watch the first array task's
# recovered parameters against mincontrast_thomas.sh's (K-based) ones on
# the same seed as a sanity cross-check.
#
# Time limit is a rough guess, same caveat as mincontrast_thomas.sh --
# empirical_g's per-t kernel evaluation is a similar O(n^2)-per-fit cost
# to empirical_K's per-t masking, so expect comparable-or-somewhat-slower
# runtime, not a different order of magnitude.
#
# Results land under results/thomas/raw/mincontrast_g/seed_<seed>/.
#
# 5 seeds via --array=0-4, matching mincontrast_g.yaml's 5-seed list.
#
# ASSUMES data/thomas/clouds.pkl already exists.
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/mincontrast_g_thomas.sh

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

python -u scripts/train.py configs/runs/thomas/mincontrast_g.yaml \
  --seed "$SEED"
