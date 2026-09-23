#!/bin/bash

#SBATCH -J featurize
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH --output=logs/featurize_%j.out
#SBATCH --error=logs/featurize_%j.err

# Persistence diagrams for every pattern in the cloud bank, data/bank/<family>/points.npz, for
# every filtration in configs/featurization/config.yaml (rips, alpha, dtm k = 5, 10, 15, 20).
#
#   run    all (family, filtration, shard) tasks on a 48-worker pool -> data/featurization/shards/
#   merge  -> data/featurization/<family>/<tag>/diagrams.npz, then deletes data/featurization/shards/
#
# Sizing: 5 families x 100000 patterns x 6 filtrations = 3M diagrams, ~35-40 CPU-h in total
# (measured on bank patterns: alpha 2 ms, rips 31-37 ms, dtm 49-67 ms each), so one 48-CPU node
# needs ~50 min plus ~10 min of merge; 3 h is margin. 48 CPUs fits any node in `compute`.
#
# Memory: a worker holds one family's whole point set (0.54 G -- run_shard reloads points.npz per
# shard, and npz cannot be memory-mapped) plus ripser's O(n^2) scratch, 0.8 G for the bank's
# largest pattern (n = 1968). 96 G over 48 workers is 2 G each, above that 1.4 G worst case. The
# merge then runs alone and peaks at ~2 G, one (family, tag) pair at a time.
#
# Disk: ~3.4 MB per shard x 6000 shards, freed by the merge, and ~0.65 G per merged pair; leave
# ~45 G free under data/featurization.
#
# Resumable: finished shards and merged diagrams are skipped. If the job dies, the shards stay on
# disk and merge does not run; just re-submit. Merge only runs after every shard succeeded, and
# refuses to write (keeping the shards) if any shard is missing or out of order. That same skipping
# is why the preflight runs first: it refuses to start on diagrams left from an earlier bank.
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/featurize.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

echo "Host: $(hostname)  Job ${SLURM_JOB_ID}  CPUs ${SLURM_CPUS_PER_TASK}"

python -u slurm/featurize_preflight.py
python -u scripts/featurize.py run --jobs "$SLURM_CPUS_PER_TASK"
python -u scripts/featurize.py merge

if [ -d data/featurization/shards ]; then
  echo "shards left over in data/featurization/shards" >&2
  exit 1
fi
ls data/featurization/*/*/diagrams.npz
