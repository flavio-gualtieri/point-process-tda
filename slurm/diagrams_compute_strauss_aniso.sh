#!/bin/bash

#SBATCH -J diagrams_compute_sa
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=24G
#SBATCH -t 02:00:00
#SBATCH --array=0-639%600
#SBATCH --output=logs/diagrams_compute_sa_%A_%a.out
#SBATCH --error=logs/diagrams_compute_sa_%A_%a.err

# STAGE 1 of 2 -- shard-parallel persistence-diagram computation for the
# NEW strauss / aniso_thomas clouds. Identical mechanism to
# slurm/diagrams_compute.sh (thomas/nested_thomas/matern_cluster); only the
# PROCESSES list and array span differ.
#
#   4 filtrations  (rips, DTM k=5, DTM k=10, DTM k=15)
# x 2 processes    (strauss, aniso_thomas)
# x NSHARDS shards each
#   = 640 array tasks, 1 CPU per task, up to 600 running at once (%600).
#
# Each task computes its shard's diagrams (train/test AND adversarial
# splits) into  data/<process>/<tag>/_shards/  and nothing else.
# STAGE 2 (slurm/diagrams_merge_strauss_aniso.sh) stitches the shards into
# the canonical data/<process>/<tag>/{,adversarial_}diagrams.pkl.
#
# Resumable: a shard whose output already exists is skipped, so a plain
# re-submit only recomputes what is missing. If specific tasks fail
# (timeout / OOM), rerun just those with more memory:
#   sbatch --array=<ids> --mem-per-cpu=48G -t 03:00:00 slurm/diagrams_compute_strauss_aniso.sh
#
# GUDHI's DTM backend leaks ~80 MB per diagram, unbounded, so NSHARDS is
# large (~100 diagrams/task) and the leak resets when each task's process
# exits. DTM tasks ~30 min (the thresh=None 2-skeleton is ~18 s/diagram);
# rips tasks are seconds. strauss clouds are small (<=~880 pts) so it
# should stay well under 24 GB; aniso_thomas (<=~1170 pts) is in the same
# class as matern_cluster -- expect a small fraction of DTM tasks to OOM
# at 24 GB, resubmit those at 48-96 GB.
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/diagrams_compute_strauss_aniso.sh
# then, once every task has finished OK:
#   sbatch slurm/diagrams_merge_strauss_aniso.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

# One thread per task -- GUDHI's DTM backend is internally multi-threaded.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

PROCESSES=(strauss aniso_thomas)
FILTRATIONS=(rips dtm_k5 dtm_k10 dtm_k15)
NSHARDS=80

NFILT=${#FILTRATIONS[@]}                   # 4
PER_PROC=$(( NFILT * NSHARDS ))            # 320
NTASKS=$(( ${#PROCESSES[@]} * PER_PROC ))  # 640  -- must equal the --array span (0-639)

t="${SLURM_ARRAY_TASK_ID:?run this as an array job (sbatch slurm/diagrams_compute_strauss_aniso.sh)}"
if (( t >= NTASKS )); then
  echo "task $t >= $NTASKS -- nothing to do (check --array span vs NSHARDS)"; exit 0
fi

proc="${PROCESSES[$(( t / PER_PROC ))]}"
rem=$(( t % PER_PROC ))
filt="${FILTRATIONS[$(( rem / NSHARDS ))]}"
shard=$(( rem % NSHARDS ))

echo "Host: $(hostname)"
echo "Job ${SLURM_JOB_ID:-unset}  Array task ${t}  ->  process=${proc}  filtration=${filt}  shard ${shard}/${NSHARDS}"

python -u scripts/processing/diagrams_shard.py compute \
  --process "$proc" \
  --filtration "$filt" \
  --n-shards "$NSHARDS" \
  --shard-index "$shard"
