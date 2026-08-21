#!/bin/bash

#SBATCH -J vec_multik_silhouette_calib_G_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-8
#SBATCH --output=logs/vec_multik_silhouette_calib_G_thomas_%A_%a.out
#SBATCH --error=logs/vec_multik_silhouette_calib_G_thomas_%A_%a.err

# SIL-1 (silhouette, p=1.0, native encoder) grid-resolution calibration
# sweep, thomas, FUSED DTM k=5,10,15 (shared-weight encoder + flat concat
# fusion, vec_multik's defaults). 3 values of G (method.params.G, the
# silhouette sample-grid resolution -- see vectorized_multik.py's
# build_silhouette_tensor). 3 seeds (9371, 9372, 9373).
# Base config is configs/runs/thomas/vec_multik_silhouette_native.yaml
# (G=128 baseline, p=1.0, homology_dims=[0,1], pd_calibration_coverage=0.95
# already set there -- untouched here), overridden per task via --set
# (config.py's apply_overrides).
#
# Each value gets its own --run-tag, so results land under
# results/thomas/dtm_k5+10+15/vec_multik_silhouette_native/_runs/calib_G<value>/seed_<seed>/
# -- separate from the plain (untagged) G=128 baseline and from every other
# sweep value, never overwriting.
#
# 3 values x 3 seeds = 9 tasks, flattened into one array
# (SLURM_ARRAY_TASK_ID = value_index * 3 + seed_index), same pattern as
# pi_multik_calib_q_thomas.sh.
#
# ASSUMES data/thomas/clouds.pkl and data/thomas/dtm_k{5,10,15}/diagrams.pkl
# already exist. If they don't, run once, serially, BEFORE submitting this
# array job (concurrent array tasks writing the same clouds.pkl/
# diagrams.pkl would race):
#   python scripts/generate.py configs/runs/thomas/vec_multik_silhouette_native.yaml
#   python scripts/featurize.py configs/runs/thomas/vec_multik_silhouette_native.yaml
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/vec_multik_silhouette_calib_G_thomas.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

CONFIG=configs/runs/thomas/vec_multik_silhouette_native.yaml
VALUES=(64 128 256)
TAGS=(calib_G064 calib_G128 calib_G256)
SEEDS=(9371 9372 9373)
N_SEEDS=${#SEEDS[@]}

VALUE_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))
VALUE="${VALUES[$VALUE_IDX]}"
TAG="${TAGS[$VALUE_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unset}  Array task: ${SLURM_ARRAY_TASK_ID:-unset}  G=$VALUE  Tag: $TAG  Seed: $SEED"
echo "Assigned GPU: ${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python -u scripts/train.py "$CONFIG" \
  --seed "$SEED" \
  --set method.params.G="$VALUE" \
  --run-tag "$TAG"
