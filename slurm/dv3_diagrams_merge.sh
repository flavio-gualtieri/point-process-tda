#!/bin/bash

#SBATCH -J dv3_diagrams_merge
#SBATCH -p compute
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH --array=0-55
#SBATCH --output=logs/dv3_diagrams_merge_%A_%a.out
#SBATCH --error=logs/dv3_diagrams_merge_%A_%a.err

# STAGE 2 of 2 -- stitch the chunks from slurm/dv3_diagrams_compute.sh into
#   data/dv3/<set>/<family>/<tag>/diagrams.pkl
# (the diagrams_shard.py / featurize.py bundle, plus case_ids and splits).
#
# One task per (set, family, filtration): A 5 families + B 4 + C 5 = 14,
# x 4 filtrations = 56 tasks (--array=0-55). List the index -> group map:
#   python scripts/processing/dv3_diagrams.py groups
# Each task checks the chunks tile [0, n) exactly with matching seeds and
# refuses to write a partial bundle, naming where the gap starts. On success
# it deletes that group's _chunks/ (edit in --keep-chunks to keep them).
# Already-merged groups are skipped.
#
# Submit after the compute array has finished OK (or chain it):
#   sbatch --dependency=afterok:<compute_job_id> slurm/dv3_diagrams_merge.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

SETS=(A B C)   # MUST match slurm/dv3_diagrams_compute.sh

t="${SLURM_ARRAY_TASK_ID:?run this as an array job (sbatch slurm/dv3_diagrams_merge.sh)}"
echo "Host: $(hostname)  Job ${SLURM_JOB_ID}  merge group ${t}"

python -u scripts/processing/dv3_diagrams.py merge \
  --group-index "$t" \
  --sets "${SETS[@]}"
