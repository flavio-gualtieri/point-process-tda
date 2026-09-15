#!/bin/bash

#SBATCH -J diagrams_merge_lgcp
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/diagrams_merge_lgcp_%A_%a.out
#SBATCH --error=logs/diagrams_merge_lgcp_%A_%a.err

# STAGE 2 of 2 -- stitch slurm/diagrams_compute_lgcp.sh's shards into
#   data/{lgcp,lgcp_strauss}/<filtration>/{,adversarial_}diagrams.pkl
#
#   task = filtration-major, same order as the compute job:
#     0 dtm_k5/lgcp   1 dtm_k5/lgcp_strauss   2 dtm_k10/lgcp   3 dtm_k10/lgcp_strauss
#     4 dtm_k15/lgcp  5 dtm_k15/lgcp_strauss  6 rips/lgcp      7 rips/lgcp_strauss
#
# So `sbatch --array=0-1 slurm/diagrams_merge_lgcp.sh` merges just the k5
# bundles, which is all the current experiments need.
#
# merge validates the stitched count against clouds.pkl and REFUSES to write a
# truncated bundle, naming the missing shard indices -- if it fails, rerun
# those compute shards (typically OOM; see the compute script's header).
# 48 GB (not 24) because LGCP diagrams are larger than any merged before.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

FILTRATIONS=(dtm_k5 dtm_k10 dtm_k15 rips)
PROCESSES=(lgcp lgcp_strauss)
NSHARDS=120   # MUST match slurm/diagrams_compute_lgcp.sh
NPROC=${#PROCESSES[@]}

t="${SLURM_ARRAY_TASK_ID:?run this as an array job (sbatch slurm/diagrams_merge_lgcp.sh)}"
if (( t >= ${#FILTRATIONS[@]} * NPROC )); then
  echo "task $t out of range -- nothing to do"; exit 0
fi
filt="${FILTRATIONS[$(( t / NPROC ))]}"
proc="${PROCESSES[$(( t % NPROC ))]}"

echo "Host: $(hostname)"
echo "Job ${SLURM_JOB_ID:-unset}  Array task ${t}  ->  merge process=${proc}  filtration=${filt}  (${NSHARDS} shards)"

python -u scripts/processing/diagrams_shard.py merge \
  --process "$proc" \
  --filtration "$filt" \
  --n-shards "$NSHARDS"
