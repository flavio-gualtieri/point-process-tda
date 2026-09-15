#!/bin/bash

#SBATCH -J dv3_diagrams
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH -t 04:00:00
#SBATCH --array=0-39
#SBATCH --output=logs/dv3_diagrams_%A_%a.out
#SBATCH --error=logs/dv3_diagrams_%A_%a.err

# STAGE 1 of 2 -- persistence diagrams for the DV3 evaluation sets
#   sets        A, B, C   (data/dv3/<set>/<family>/clouds.pkl; 14 set/family pairs, 115,800 clouds)
#   filtrations rips, dtm_k5, dtm_k10, dtm_k15   (maxdim 1, DTM q=2 -- same as diagrams_shard.py)
#
# 40 array tasks x 48 CPUs = 1,920 CPUs (same total as the original 20x96
# layout; QOS cap on `compute` is 2,000 cpu). Deliberately kept BELOW the
# 96-cpu/376G per-job QOS cap: 48 cpus fits any node in the partition (the
# ~220 48-core ddy/sdx nodes, not just the 12 384-core ehc ones), which
# backfills far sooner when the cluster is busy -- switched from 96/task
# 2026-09-16 because all ~220 compute-partition nodes were fully booked and
# the 96-cpu array was stuck behind a next-morning estimated start. --mem
# keeps the same ~3.75G/core ratio as the 376G/96-cpu layout.
#
# Work: 56 (set, family, filtration) groups cut into ~16,100 small
# contiguous chunks (<= 40 clouds / ~5 CPU-min for DTM), dealt to the 40
# tasks by estimated cost (~25 CPU-h each, ~1,000 CPU-h total). Inside a
# task, scripts/processing/dv3_diagrams.py runs one FRESH python process per
# chunk, up to 48 at once, and only launches a chunk if its estimated peak
# memory fits in 88% of --mem. Why that is OOM-safe:
#   * GUDHI's DTM leak (~80 MB/diagram) resets every chunk (<= 40 diagrams);
#   * the real DTM cost is the full 2-skeleton, ~n^3 memory: n=1320 (largest
#     A cloud) is ~36 GB for ONE diagram, n~330 is < 1 GB -- the scheduler
#     sizes each chunk by its largest cloud and admits it only if it fits;
#   * each worker reports its true peak RSS; the driver raises its memory
#     scale if the model was optimistic, and a worker killed by the OOM
#     killer is re-queued with 2x the memory estimate (3 attempts).
# If a task still logs "[fail] ... likely OOM", re-submit with a more
# cautious model:  MEM_SCALE=2 sbatch --array=<ids> slurm/dv3_diagrams_compute.sh
#
# Expected wall ~60-100 min per task (scheduler simulation at 48 cpus/180G);
# the 4 h limit is margin. Resumable: finished chunks are kept on disk and
# skipped regardless of which task index owns them now, so re-submitting is
# always safe. IMPORTANT: NJOBS determines chunk ownership -- if you change
# it (as happened here, 20->40), submit the FULL new array (0-39), not just
# the indices that failed under the old NJOBS, since the chunk-to-task
# mapping is recomputed from scratch; already-done chunks are still skipped
# either way. Only re-submit a subset of indices (--array=3,7) when NJOBS is
# UNCHANGED from the run that produced the failures.
#
# Dry run first (reads only cloud sizes, a few seconds, fine on the login node):
#   python scripts/processing/dv3_diagrams.py plan --n-jobs 40 --cpus 48 --mem-gb 180
#
# Submit from the point-process-tda repo root:
#   jid=$(sbatch --parsable slurm/dv3_diagrams_compute.sh)
#   sbatch --dependency=afterok:${jid} slurm/dv3_diagrams_merge.sh
# Outputs: data/dv3/<set>/<family>/<tag>/diagrams.pkl

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

# One thread per worker process: 96 single-threaded workers per task.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

NJOBS=40                   # MUST equal the --array span (0-39), also on re-submits
SETS=(A B C)
MEM_SCALE="${MEM_SCALE:-1.0}"

t="${SLURM_ARRAY_TASK_ID:?run this as an array job (sbatch slurm/dv3_diagrams_compute.sh)}"

echo "Host: $(hostname)"
echo "Job ${SLURM_JOB_ID}  array task ${t}/${NJOBS}  CPUs ${SLURM_CPUS_PER_TASK}  mem ${SLURM_MEM_PER_NODE} MB  sets ${SETS[*]}"

python -u scripts/processing/dv3_diagrams.py run \
  --job-index "$t" \
  --n-jobs "$NJOBS" \
  --sets "${SETS[@]}" \
  --mem-scale "$MEM_SCALE"
