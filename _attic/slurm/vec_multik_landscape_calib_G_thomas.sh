#!/bin/bash

#SBATCH -J vec_multik_landscape_calib_G_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-8
#SBATCH --output=logs/vec_multik_landscape_calib_G_thomas_%A_%a.out
#SBATCH --error=logs/vec_multik_landscape_calib_G_thomas_%A_%a.err

# Landscape grid-resolution sweep, thomas, FUSED DTM k=5,10,15 (H0+H1,
# native CNN encoder over the (K, G) raster -- vec_multik_landscape_native.yaml's
# defaults), 3 values of the calibrated grid size G (method.params.G). K is
# LEFT UNSET in every arm, so each still gets its own data-derived K from
# the coverage rule (calibrated.py's choose_K) recomputed on that arm's own
# G-point grid -- this sweep isolates grid resolution, not K.
# 3 seeds (9371, 9372, 9373).
#
# Base config is the existing configs/runs/thomas/vec_multik_landscape_native.yaml
# (G=128 baseline, homology_dims=[0,1], pd_calibration_coverage=0.95),
# overridden per task via --set (config.py's apply_overrides).
#
# Each value gets its own --run-tag, so results land under
# results/thomas/dtm_k5+10+15/vec_multik_landscape_native/_runs/calib_G<value>/seed_<seed>/
# -- separate from the plain (untagged) baseline results and from every
# other sweep value, never overwriting.
#
# 3 values x 3 seeds = 9 tasks, flattened into one array
# (SLURM_ARRAY_TASK_ID = value_index * 3 + seed_index), same pattern as
# pi_multik_calib_resolution_thomas.sh/betti_multik_dims_thomas.sh.
#
# ASSUMES data/thomas/clouds.pkl and data/thomas/dtm_k{5,10,15}/diagrams.pkl
# already exist (they do).
# Submit from the point-process-tda repo root:
#   sbatch slurm/vec_multik_landscape_calib_G_thomas.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

CONFIG=configs/runs/thomas/vec_multik_landscape_native.yaml
VALUES=(64 128 256)
TAGS=(calib_G64 calib_G128 calib_G256)
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
