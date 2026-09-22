#!/bin/bash

#SBATCH -J bank
#SBATCH -p computeshort
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH --array=0-99
#SBATCH --output=logs/bank_%A_%a.out
#SBATCH --error=logs/bank_%A_%a.err

# One array task per shard of one family's bank: ceil(thetas / shard_size) = 50000 / 500 = 100.
# FAMILY selects which; only LGCP is worth a cluster (the other four are ~2 CPU-min in total).
#
#   sbatch slurm/bank.sh                                   # lgcp
#   FAMILY=nested sbatch --array=0-99 slurm/bank.sh        # any other family
#   sbatch --dependency=afterok:<array job id> slurm/bank_merge.sh
#
# Sizing is set by LGCP's grid size M, which grid_size picks per theta and which drives both cost
# and memory -- the circulant embedding works on (2M)^2 arrays and lgcp() holds several at once.
# Measured: M = 128 costs 3 ms and 0.07 G per pattern, M = 4096 costs 2.0 s and 1.94 G. Only 1.7%
# of thetas reach 4096, but a 500-theta shard will contain some, so 8 G is the worst case with
# margin. A shard is ~4 min, so the hour is ~15x headroom.
#
# One CPU per task rather than a pool: tasks this small backfill into gaps, and each carries only
# one worker's peak instead of eight. Resumable -- run_shard skips a shard that already exists,
# so a re-submit only fills what is missing.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

FAMILY="${FAMILY:-lgcp}"

echo "Host: $(hostname)  Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  ${FAMILY} shard ${SLURM_ARRAY_TASK_ID}"

python -u scripts/simulate.py run --family "$FAMILY" --shard "$SLURM_ARRAY_TASK_ID" --jobs 1
