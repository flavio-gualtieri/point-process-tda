#!/bin/bash

#SBATCH -J featurize
#SBATCH -p computeshort
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH --array=0-29
#SBATCH --output=logs/featurize_%A_%a.out
#SBATCH --error=logs/featurize_%A_%a.err

# One array task per (family, filtration) pair: 5 families x 6 filtrations = 30 tasks, each running
# that pair's 40 shards on an 8-worker pool. Total work is ~5.6 CPU-h, so a task is ~10 CPU-min.
#
# Small tasks backfill into gaps rather than waiting for a whole free node, and 8 CPUs / 16G / 1 h
# fits `computeshort`, which shares `compute`'s nodes but has a far shorter queue.
#
# Memory: a worker holds one family's points (~100 MB) plus ripser/alpha O(n^2) scratch for n < 1000
# (~10s of MB), so ~0.5 G per worker; 16 G for 8 workers is already margin.
#
# Merge is a separate job because it must not start until every shard of every pair exists:
#   sbatch slurm/featurize_array.sh
#   sbatch --dependency=afterok:<array job id> slurm/featurize_merge.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

read -r FAMILY TAG < <(python -c "
from cloudforger.featurization.filtrations import tag
from cloudforger.featurization.sweep import Config, families
pairs = [(f, tag(s)) for f in families() for s in Config.load().filtrations]
print(*pairs[$SLURM_ARRAY_TASK_ID])
")

echo "Host: $(hostname)  Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  ${FAMILY}/${TAG}  CPUs ${SLURM_CPUS_PER_TASK}"

python -u scripts/featurize.py run --family "$FAMILY" --tag "$TAG" --jobs "$SLURM_CPUS_PER_TASK"
