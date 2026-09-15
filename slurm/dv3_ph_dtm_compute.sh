#!/bin/bash

#SBATCH -J dv3_ph_dtm
#SBATCH -p compute,computeshort
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH --array=0-119
#SBATCH --output=logs/dv3_ph_dtm_%A_%a.out
#SBATCH --error=logs/dv3_ph_dtm_%A_%a.err

# DTM diagrams for the persistence experiments -- by default the TRAIN set at
# k=5, which slurm/dv3_dtm5_compute.sh (A/B/C only) never computed and every
# PH model trains on. Same driver and shape as that script.
#
#   env DTM_K  (default 5)       -> filtration dtm_k<DTM_K>
#   env SETS   (default "train") -> e.g. SETS="train A B C" for k=10/15, whose
#                                   A/B/C bundles are not merged either
#
# Normally submitted by slurm/dv3_ph_submit.sh. Cost (plan, 2026-09-15):
# train dtm_k5 = 192 CPU-h, 2,499 chunks, ~6 min modelled per task at
# 120 x 16 CPUs; the largest cloud (train/lgcp, n=1477) needs ~36 GB, inside
# the 96 GB task. Resumable: finished chunks are skipped, so re-submit the
# same array (keep NJOBS=120 and the same DTM_K/SETS). MEM_SCALE=2 for caution.
# Output: data/dv3/<set>/<family>/dtm_k<K>/diagrams.pkl after the merge
# (slurm/dv3_ph_dtm_merge.sh).

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

NJOBS=120                  # MUST equal the --array span (0-119)
DTM_K="${DTM_K:-5}"
read -r -a SET_LIST <<< "${SETS:-train}"
MEM_SCALE="${MEM_SCALE:-1.0}"

t="${SLURM_ARRAY_TASK_ID:?run this as an array job}"
echo "Host: $(hostname)  partition ${SLURM_JOB_PARTITION}  job ${SLURM_JOB_ID}  task ${t}/${NJOBS}  CPUs ${SLURM_CPUS_PER_TASK}  mem ${SLURM_MEM_PER_NODE} MB  ->  dtm_k${DTM_K} on ${SET_LIST[*]}"

python -u scripts/processing/dv3_diagrams.py run \
  --job-index "$t" \
  --n-jobs "$NJOBS" \
  --sets "${SET_LIST[@]}" \
  --filtrations "dtm_k${DTM_K}" \
  --mem-scale "$MEM_SCALE"
