#!/bin/bash

#SBATCH -J dv3_ph_dtm_merge
#SBATCH -p compute,computeshort
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH --array=0-4
#SBATCH --output=logs/dv3_ph_dtm_merge_%A_%a.out
#SBATCH --error=logs/dv3_ph_dtm_merge_%A_%a.err

# Stitch slurm/dv3_ph_dtm_compute.sh's chunks into
# data/dv3/<set>/<family>/dtm_k<K>/diagrams.pkl. One task per (set, family):
# train alone = 5 groups (--array=0-4, the default); SETS="train A B C" = 19
# (--array=0-18). Same DTM_K / SETS env as the compute job. Map:
#   python scripts/processing/dv3_diagrams.py groups --sets train --filtrations dtm_k5
# Refuses to write a partial bundle; deletes that group's _chunks/ on success.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

DTM_K="${DTM_K:-5}"
read -r -a SET_LIST <<< "${SETS:-train}"
t="${SLURM_ARRAY_TASK_ID:?run this as an array job}"
echo "Host: $(hostname)  Job ${SLURM_JOB_ID}  merge group ${t}  (dtm_k${DTM_K} on ${SET_LIST[*]})"

python -u scripts/processing/dv3_diagrams.py merge \
  --group-index "$t" \
  --sets "${SET_LIST[@]}" \
  --filtrations "dtm_k${DTM_K}"
