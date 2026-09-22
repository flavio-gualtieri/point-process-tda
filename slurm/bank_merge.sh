#!/bin/bash

#SBATCH -J bank_merge
#SBATCH -p computeshort
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --output=logs/bank_merge_%j.out
#SBATCH --error=logs/bank_merge_%j.err

# Stitches one family's shards into data/bank/<family>/{points.npz, manifest.csv} and checks they
# are complete and consistent. A separate job because it must not start until every shard exists:
#
#   sbatch --dependency=afterok:<array job id> slurm/bank_merge.sh
#
# Single-threaded and I/O bound; peak memory is the family's whole point set (~0.6 G) plus the
# concatenation, so 32 G is ample. The shards are left in place: they are the resumable unit, and
# merge is cheap to repeat.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

FAMILY="${FAMILY:-lgcp}"

echo "Host: $(hostname)  Job ${SLURM_JOB_ID}  ${FAMILY}"

python -u scripts/simulate.py merge --family "$FAMILY"
ls -l "data/bank/${FAMILY}"
