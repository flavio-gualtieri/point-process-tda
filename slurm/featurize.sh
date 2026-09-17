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

# Persistence diagrams for every pattern in data/simulation/<family>/, for every filtration in
# configs/featurization/config.yaml (rips, alpha, dtm k = 5, 10, 15, 20).
#
#   run    all (family, filtration, shard) tasks on a 48-worker pool -> data/featurization/shards/
#   merge  -> data/featurization/<family>/<tag>/diagrams.npz, then deletes data/featurization/shards/
#
# Sizing: ~5.6 CPU-h in total (measured, all n < 1000, ripser/alpha use O(n^2) memory), so one
# 48-CPU node needs ~10-20 min; 3 h is margin. 48 CPUs fits any node in `compute`.
#
# Resumable: finished shards and merged diagrams are skipped. If the job dies, the shards stay on
# disk and merge does not run; just re-submit. Merge only runs after every shard succeeded, and
# refuses to write (keeping the shards) if any shard is missing or out of order.
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

python -u scripts/featurize.py run --jobs "$SLURM_CPUS_PER_TASK"
python -u scripts/featurize.py merge

if [ -d data/featurization/shards ]; then
  echo "shards left over in data/featurization/shards" >&2
  exit 1
fi
ls data/featurization/*/*/diagrams.npz
