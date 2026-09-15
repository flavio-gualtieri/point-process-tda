#!/bin/bash

#SBATCH -J diagrams_merge_sa
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/diagrams_merge_sa_%A_%a.out
#SBATCH --error=logs/diagrams_merge_sa_%A_%a.err

# STAGE 2 of 2 for strauss / aniso_thomas -- stitch the shard fragments
# from slurm/diagrams_compute_strauss_aniso.sh back into the canonical
# per-(process, filtration) bundles:
#   data/<process>/<tag>/diagrams.pkl
#   data/<process>/<tag>/adversarial_diagrams.pkl
# in exactly the shape scripts/featurize.py's own diagram stage writes.
#
#   2 processes x 4 filtrations = 8 independent merge tasks (--array=0-7).
# Each validates the stitched count against the current clouds.pkl and
# refuses to write a truncated bundle; it exits non-zero naming the missing
# shard indices if any compute task did not finish. Re-run those compute
# indices, then re-submit this.
#
# NSHARDS MUST match slurm/diagrams_compute_strauss_aniso.sh.
#
# On success each task deletes its own data/<process>/<tag>/_shards/
# fragments (pass --keep-shards in the python call below to keep them).
#
# Submit from the point-process-tda repo root, after
# diagrams_compute_strauss_aniso.sh has fully finished:
#   sbatch slurm/diagrams_merge_strauss_aniso.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

PROCESSES=(strauss aniso_thomas)
FILTRATIONS=(rips dtm_k5 dtm_k10 dtm_k15)
NSHARDS=80   # MUST match slurm/diagrams_compute_strauss_aniso.sh

NFILT=${#FILTRATIONS[@]}   # 4
t="${SLURM_ARRAY_TASK_ID:?run this as an array job (sbatch slurm/diagrams_merge_strauss_aniso.sh)}"

proc="${PROCESSES[$(( t / NFILT ))]}"
filt="${FILTRATIONS[$(( t % NFILT ))]}"

echo "Host: $(hostname)"
echo "Job ${SLURM_JOB_ID:-unset}  Array task ${t}  ->  merge process=${proc}  filtration=${filt}  (${NSHARDS} shards)"

python -u scripts/processing/diagrams_shard.py merge \
  --process "$proc" \
  --filtration "$filt" \
  --n-shards "$NSHARDS"
