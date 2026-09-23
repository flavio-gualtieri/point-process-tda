#!/bin/bash

#SBATCH -J featurize
#SBATCH -p computeshort
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH --array=0-29
#SBATCH --output=logs/featurize_%A_%a.out
#SBATCH --error=logs/featurize_%A_%a.err

# Persistence diagrams for the cloud bank: every pattern in data/bank/<family>/points.npz, for
# every filtration in configs/featurization/config.yaml (rips, alpha, dtm k = 5, 10, 15, 20).
#
# One array task per (family, filtration) pair: 5 families x 6 filtrations = 30 tasks, each running
# that pair's 200 shards (100000 patterns / shard_size 500) on an 8-worker pool.
#
# Small tasks backfill into gaps rather than waiting for a whole free node, and 8 CPUs / 24G / 1 h
# fits `computeshort`, which shares `compute`'s nodes but has a far shorter queue.
#
# Sizing (measured on bank patterns, 150 per family per filtration): alpha 2 ms, rips 31-37 ms and
# dtm 49-67 ms per pattern, so a pair is 0.05 CPU-h (alpha) to 1.9 CPU-h (nested dtm) and the whole
# sweep ~35-40 CPU-h. The worst task is then ~15 min of wall on 8 workers; the hour is ~4x margin.
#
# Memory: a worker holds the family's whole point set (0.54 G -- run_shard reloads points.npz per
# shard, and npz cannot be memory-mapped) plus ripser's O(n^2) scratch, which for the largest
# pattern in the bank (n = 1968, up from n < 1000 in the old simulation set) peaks at 0.8 G.
# So ~1.4 G per worker in the worst case, and 24 G for 8 workers is ~2x margin.
#
# Disk: ~3.4 MB per shard x 6000 shards, freed by the merge, and ~0.65 G per merged pair; leave
# ~45 G free under data/featurization.
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

python -u slurm/featurize_preflight.py

read -r FAMILY TAG < <(python -c "
from cloudforger.featurization.filtrations import tag
from cloudforger.featurization.sweep import Config, families
pairs = [(f, tag(s)) for f in families() for s in Config.load().filtrations]
print(*pairs[$SLURM_ARRAY_TASK_ID])
")

echo "Host: $(hostname)  Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  ${FAMILY}/${TAG}  CPUs ${SLURM_CPUS_PER_TASK}"

python -u scripts/featurize.py run --family "$FAMILY" --tag "$TAG" --jobs "$SLURM_CPUS_PER_TASK"
