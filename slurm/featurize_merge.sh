#!/bin/bash

#SBATCH -J featurize_merge
#SBATCH -p computeshort
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --output=logs/featurize_merge_%j.out
#SBATCH --error=logs/featurize_merge_%j.err

# Stitches the shards written by slurm/featurize_array.sh into
# data/featurization/<family>/<tag>/diagrams.npz, then deletes data/featurization/shards/.
#
# Single-threaded and I/O bound: merge() concatenates one (family, tag) pair at a time, so peak
# memory is one pair's 200 shards plus the concatenated copy, ~2 G for the bank's 100000 patterns
# per family -- not all 30 pairs. Runs as one job rather than per-pair because it also prunes the
# shared shards/ parent directories, which would race across concurrent tasks.
#
#   sbatch --dependency=afterok:<array job id> slurm/featurize_merge.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

echo "Host: $(hostname)  Job ${SLURM_JOB_ID}"

python -u scripts/featurize.py merge

if [ -d data/featurization/shards ]; then
  echo "shards left over in data/featurization/shards" >&2
  exit 1
fi
ls data/featurization/*/*/diagrams.npz
