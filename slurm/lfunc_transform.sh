#!/bin/bash

#SBATCH -J lfunc_transform
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 4
#SBATCH --cpus-per-gpu=4
#SBATCH -t 01:00:00
#SBATCH --mem-per-cpu=14G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/lfunc_transform_%j.out
#SBATCH --error=logs/lfunc_transform_%j.err

# PREREQUISITE for the L-reparameterization 2x2. Builds
#   data/<process>/l_dtm_k5/{,adversarial_}diagrams.pkl
# from the existing dtm_k5 bundles by applying (b, d) -> (L(b), L(d)), with L
# the cloud's own empirical Ripley L read from the cached
# {,adversarial_}clouds.lfunc*_cache.npz the vihrs baseline already wrote.
#
# NO persistence is recomputed -- this is a per-point interpolation over
# diagrams that already exist, so it runs in seconds to minutes, not hours.
# (The from-scratch path exists too: filtration REGISTRY names l_dtm / l_rips,
# for data generated in future. Same maths, shared code.)
#
# Processes: thomas, nested_thomas, classification. Seed joins verified: 0
# missing seeds on all six bundles, including classification's globally-unique
# seeds against its differently-named lfunc_classify_cache.npz.
#
# The GPU request is vestigial -- this job is pure numpy and uses no GPU, but
# the sae partition's pilot_sae_gpu account expects a GPU allocation and this
# is a two-minute job.
#
# Run BEFORE slurm/lfunc_{thomas,nested_thomas,classification}.sh:
#   sbatch slurm/lfunc_transform.sh
# or chain them:
#   tid=$(sbatch --parsable slurm/lfunc_transform.sh)
#   sbatch --dependency=afterok:$tid --array=0-29 slurm/lfunc_thomas.sh
#
# Idempotent: existing l_dtm_k5 bundles are skipped unless --force is added
# to the command below.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

echo "Host: $(hostname)"
echo "Job ${SLURM_JOB_ID:-unset}  ->  l_dtm_k5 transform for thomas / nested_thomas / classification"

python -u scripts/processing/transform_diagrams_lfunc.py \
    --tag dtm_k5 \
    --process thomas \
    --process nested_thomas \
    --process classification

echo
echo "=== resulting bundles ==="
for p in thomas nested_thomas classification; do
  for f in diagrams.pkl adversarial_diagrams.pkl; do
    path="data/$p/l_dtm_k5/$f"
    if [ -f "$path" ]; then echo "  OK  $path ($(du -h "$path" | cut -f1))"; else echo "  MISSING $path"; fi
  done
done
