#!/bin/bash

#SBATCH -J diagrams_compute
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=24G
#SBATCH -t 02:00:00
#SBATCH --array=0-959%600
#SBATCH --output=logs/diagrams_compute_%A_%a.out
#SBATCH --error=logs/diagrams_compute_%A_%a.err

# STAGE 1 of 2 -- shard-parallel persistence-diagram computation for the
# regenerated thomas / nested_thomas / matern_cluster clouds.
#
#   4 filtrations  (rips, DTM k=5, DTM k=10, DTM k=15)
# x 3 processes    (thomas, nested_thomas, matern_cluster)
# x NSHARDS shards each
#   = 960 array tasks, 1 CPU per task, up to 600 running at once (%600).
#
# Each task computes its shard's diagrams (train/test AND adversarial
# splits) into  data/<process>/<tag>/_shards/  and nothing else --
# calibration/imaging is per-seed inside the experiments, not here.
# STAGE 2 (slurm/diagrams_merge.sh) stitches the shards back into the
# canonical data/<process>/<tag>/{,adversarial_}diagrams.pkl.
#
# Resumable: a shard whose output already exists is skipped, so a plain
# re-submit only recomputes what is missing. If specific tasks fail
# (timeout / OOM), rerun just those:
#   sbatch --array=<comma,separated,ids>%600 slurm/diagrams_compute.sh
#
# GUDHI's DTM backend leaks ~80 MB per diagram, unbounded, and there is
# no in-process fix (a forked worker pool deadlocks; spawn re-imports the
# world every recycle). So NSHARDS is large -- ~100 diagrams per task --
# and the leak simply resets when each task's process exits. ~100 * ~3 s
# ~= 30 min/task for DTM (the thresh=None 2-skeleton is ~18 s/diagram --
# slower than first guessed); rips tasks are seconds. With %600 the 960
# tasks are essentially one wave -> ~35-45 min wall. 24 GB: at 16 GB the
# first run OOM'd ~1.3% of tasks (leak ~80 MB/diag + big-nested_thomas
# transients). If a task still OOMs/times out, re-submit just those ids
# with a higher --mem-per-cpu / -t on the sbatch command line.
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/diagrams_compute.sh
# then, once every task has finished OK:
#   sbatch slurm/diagrams_merge.sh
# (or chain: sbatch --dependency=afterok:<this_job_id> slurm/diagrams_merge.sh)

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

PROCESSES=(thomas nested_thomas matern_cluster)
FILTRATIONS=(rips dtm_k5 dtm_k10 dtm_k15)
NSHARDS=80

NFILT=${#FILTRATIONS[@]}                 # 4
PER_PROC=$(( NFILT * NSHARDS ))          # 320
NTASKS=$(( ${#PROCESSES[@]} * PER_PROC ))  # 960  -- must equal the --array span (0-959)

t="${SLURM_ARRAY_TASK_ID:?run this as an array job (sbatch slurm/diagrams_compute.sh)}"
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
