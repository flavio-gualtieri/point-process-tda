#!/bin/bash

#SBATCH -J dv3_dtm5
#SBATCH -p compute,computeshort
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH --array=0-119
#SBATCH --output=logs/dv3_dtm5_%A_%a.out
#SBATCH --error=logs/dv3_dtm5_%A_%a.err

# DTM k=5 ONLY, DV3 sets A/B/C -- the fast-turnaround variant of
# slurm/dv3_diagrams_compute.sh (same driver, --filtrations dtm_k5).
#
# Shaped to START as soon as possible on a busy cluster:
#   * 16 CPUs / 96G per task fits any node in the partition and slots into
#     small gaps as other jobs end;
#   * a <= 1 h limit makes every task eligible for BOTH `compute` and
#     `computeshort` (separate 2,000-CPU per-user QOS caps; Slurm starts
#     each task in whichever partition can take it first) and makes it an
#     easy backfill candidate.
# 120 tasks x 16 CPUs; ~250 CPU-h of dtm_k5 remained on 2026-09-15 (the
# earlier 20x96 run finished ~25%); simulated ~14 min per task, <= ~30 min
# if the cost model is 2x optimistic.
#
# Resumable and safe to re-submit: finished chunks (and merged groups) are
# skipped. Keep NJOBS=120 when re-submitting a subset (--array=5,17); if
# you change NJOBS, re-submit the full array.
#
# Submit from the repo root:
#   jid=$(sbatch --parsable slurm/dv3_dtm5_compute.sh)
#   sbatch --dependency=afterok:${jid} slurm/dv3_dtm5_merge.sh
# Output: data/dv3/<set>/<family>/dtm_k5/diagrams.pkl

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
SETS=(A B C)
MEM_SCALE="${MEM_SCALE:-1.0}"

t="${SLURM_ARRAY_TASK_ID:?run this as an array job}"
echo "Host: $(hostname)  partition ${SLURM_JOB_PARTITION}  job ${SLURM_JOB_ID}  task ${t}/${NJOBS}  CPUs ${SLURM_CPUS_PER_TASK}  mem ${SLURM_MEM_PER_NODE} MB"

python -u scripts/processing/dv3_diagrams.py run \
  --job-index "$t" \
  --n-jobs "$NJOBS" \
  --sets "${SETS[@]}" \
  --filtrations dtm_k5 \
  --mem-scale "$MEM_SCALE"
