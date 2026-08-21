#!/bin/bash

#SBATCH -J error_floor_estimate_thomas
#SBATCH -t 08:00:00
#SBATCH --mem-per-cpu=11G
#SBATCH --array=0-9
#SBATCH --output=logs/error_floor_estimate_thomas_%A_%a.out
#SBATCH --error=logs/error_floor_estimate_thomas_%A_%a.err

# Computes the irreducible error floor on the thomas error-floor dataset
# (scripts/estimate_error_floor.py) via mincontrast (min-contrast on K --
# the estimator picked for this, per the same reasoning as
# mincontrast_thomas.sh: fits per-cloud, no trained checkpoint needed,
# nested_thomas has no equivalent so this is thomas-only).
#
# CPU-only. Sharded across 10 array tasks, 5 of the 50 theta groups (1000
# of the 10,000 clouds) each -- the full 10,000-cloud fit in one job would
# be ~2-3x the total work of mincontrast_thomas.sh's entire 5-seed batch
# combined; this splits it the same way that batch is split across seeds.
# Each task writes its own shard JSON
# (data/error_floor/thomas/error_floor_mincontrast_groups<a>-<b>.json);
# run scripts/merge_error_floor_shards.py AFTER all 10 tasks finish to get
# the actual floor_L number -- no single shard's floor_L is the answer.
#
# Time limit is a rough guess, same caveat as mincontrast_thomas.sh (no
# reference runtime measured yet -- adjust after watching task 0).
#
# ASSUMES slurm/error_floor_generate_thomas.sh has already completed
# (data/error_floor/thomas/clouds.pkl exists).
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/error_floor_estimate_thomas.sh
# then, once all 10 tasks finish:
#   python scripts/merge_error_floor_shards.py \
#     data/error_floor/thomas/error_floor_mincontrast_groups*.json \
#     --out data/error_floor/thomas/error_floor_mincontrast.json

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

GROUPS_PER_SHARD=5
GROUP_START=$(( SLURM_ARRAY_TASK_ID * GROUPS_PER_SHARD ))
GROUP_END=$(( GROUP_START + GROUPS_PER_SHARD ))

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unset}  Array task: ${SLURM_ARRAY_TASK_ID:-unset}  Groups: [$GROUP_START, $GROUP_END)"

python -u scripts/estimate_error_floor.py data/error_floor/thomas/clouds.pkl \
  --reps 200 \
  --estimator mincontrast \
  --group-start "$GROUP_START" \
  --group-end "$GROUP_END"
