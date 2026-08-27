#!/bin/bash

#SBATCH -J extra_seeds_cpu_thomas
#SBATCH -t 08:00:00
#SBATCH --mem-per-cpu=11G
#SBATCH --array=0-9
#SBATCH --output=logs/extra_seeds_cpu_thomas_%A_%a.out
#SBATCH --error=logs/extra_seeds_cpu_thomas_%A_%a.err

# Tops mincontrast / mincontrast_g (currently n=5) up to n=10, by running
# the 5 seeds (9376-9380) each is missing -- same commands as
# mincontrast_thomas.sh / mincontrast_g_thomas.sh, other half of the seed
# list. CPU only, no GPU directives -- mirrors those scripts' convention.
# No nested-Thomas counterpart: nested Thomas has no closed-form K or g.
#
# 2 arms x 5 seeds = 10 tasks, --array=0-9. ARM = task_id / 5 selects K vs
# g (case statement below); SEED = SEEDS[task_id % 5].
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/extra_seeds_cpu_thomas.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

SEEDS=(9376 9377 9378 9379 9380)
ARM=$(( SLURM_ARRAY_TASK_ID / 5 ))
SIDX=$(( SLURM_ARRAY_TASK_ID % 5 ))
SEED="${SEEDS[$SIDX]}"

case "$ARM" in
  0) CMD=(python -u scripts/train.py configs/runs/thomas/mincontrast.yaml --seed "$SEED") ;;
  1) CMD=(python -u scripts/train.py configs/runs/thomas/mincontrast_g.yaml --seed "$SEED") ;;
  *) echo "bad ARM index: $ARM" >&2; exit 1 ;;
esac

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unset}  Array task: ${SLURM_ARRAY_TASK_ID:-unset}  Arm: $ARM  Seed: $SEED"
echo "Command: ${CMD[*]}"

"${CMD[@]}"
