#!/bin/bash

#SBATCH -J dv3_dtm5_merge
#SBATCH -p compute,computeshort
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH --array=0-13
#SBATCH --output=logs/dv3_dtm5_merge_%A_%a.out
#SBATCH --error=logs/dv3_dtm5_merge_%A_%a.err

# Stitch dtm_k5 chunks into data/dv3/<set>/<family>/dtm_k5/diagrams.pkl.
# One task per (set, family): A 5 + B 4 + C 5 = 14 (--array=0-13). Map:
#   python scripts/processing/dv3_diagrams.py groups --filtrations dtm_k5
# Refuses to write a partial bundle; deletes that group's _chunks/ on success.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

SETS=(A B C)
t="${SLURM_ARRAY_TASK_ID:?run this as an array job}"
echo "Host: $(hostname)  Job ${SLURM_JOB_ID}  merge group ${t}"

python -u scripts/processing/dv3_diagrams.py merge \
  --group-index "$t" \
  --sets "${SETS[@]}" \
  --filtrations dtm_k5
