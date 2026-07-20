#!/bin/bash

#SBATCH -J nested_thomas_diagrams_compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH -t 00:30:00
#SBATCH --array=0-9
#SBATCH --output=logs/nested_thomas_diagrams_compute_%A_%a.out
#SBATCH --error=logs/nested_thomas_diagrams_compute_%A_%a.err

# Submit from the point-process-tda repo root, AFTER
# nested_thomas_generate.sh (stage 1) has finished:
#   sbatch scripts/processing/params/nested_thomas_diagrams_compute.sh
#
# STAGE 2a of 3 for the nested_thomas dataset -- DTM persistence diagrams,
# 10-way SLURM array (not diagrams_compute_k5.sh's 60-way -- that was sized
# for the full ~40,000-cloud flat-Thomas dataset; ~1,000 clouds here doesn't
# need that much fan-out, but this is still the heaviest stage -- persistent
# homology computation scales much worse than linearly with point count, see
# the sizing comment in configs/params/processing/nested_thomas_cloudgen.yaml
# -- so it's still worth parallelizing rather than running serially).
#
# No separate "split" pre-stage (unlike diagrams_split.sh before
# diagrams_compute_k5.sh): that exists to avoid each of 60 array tasks
# loading the full ~40,000-cloud clouds.pkl redundantly. At ~1,000 clouds
# the file is small enough that each of these 10 tasks loading and slicing
# it independently (diagrams_slurm.py's own fallback path when no chunk
# exists yet) is not worth the extra stage.
#
# Each array task computes ~1/10 of the clouds and writes a partial to
# data/params/2d/nested_thomas/_chunks/. 30 min/task is a safety-margin
# guess, not a measured number -- if any task times out or OOMs, check
# logs/nested_thomas_diagrams_compute_<jobid>_<taskid>.out; that task is
# resumable (re-submit just needs the same --array range; diagrams_slurm.py
# skips groups whose partial already exists).
#
# Next: scripts/processing/params/nested_thomas_diagrams_merge.sh, after
# every array task here has finished successfully.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

# Keep diagram computation single-threaded (one core per task), matching
# diagrams_compute_k5.sh's own convention.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unset}  Array task: ${SLURM_ARRAY_TASK_ID:-unset}"

python scripts/processing/params/diagrams_slurm.py \
  --config configs/params/processing/nested_thomas_diagrams.yaml \
  --process nested_thomas \
  --dimension 2 \
  --n-groups 10 \
  --group-id "${SLURM_ARRAY_TASK_ID}" \
  --splits train_test adversarial \
  --mode compute
