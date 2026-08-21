#!/bin/bash

#SBATCH -J vec_multik_landscape_calib_kthr_nested_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-8
#SBATCH --output=logs/vec_multik_landscape_calib_kthr_nested_thomas_%A_%a.out
#SBATCH --error=logs/vec_multik_landscape_calib_kthr_nested_thomas_%A_%a.err

# nested_thomas sibling of vec_multik_landscape_calib_kthr_thomas.sh -- see
# that script for the full rationale. K-rule relative-threshold sweep
# (method.params.K_rel_threshold, calibrated.py's choose_K), FUSED DTM
# k=5,10,15 (H0+H1, native CNN encoder), 3 values, K left unset (data-
# derived) in every arm. 3 seeds (9371, 9372, 9373).
#
# Base config is the existing configs/runs/nested_thomas/vec_multik_landscape_native.yaml,
# overridden per task via --set.
#
# Each value gets its own --run-tag, so results land under
# results/nested_thomas/dtm_k5+10+15/vec_multik_landscape_native/_runs/calib_kthr<value>/seed_<seed>/
# -- separate from the plain (untagged) baseline results and from every
# other sweep value, never overwriting.
#
# 3 values x 3 seeds = 9 tasks, flattened into one array
# (SLURM_ARRAY_TASK_ID = value_index * 3 + seed_index).
#
# ASSUMES data/nested_thomas/clouds.pkl and data/nested_thomas/dtm_k{5,10,15}/diagrams.pkl
# already exist (they do).
# Submit from the point-process-tda repo root:
#   sbatch slurm/vec_multik_landscape_calib_kthr_nested_thomas.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

CONFIG=configs/runs/nested_thomas/vec_multik_landscape_native.yaml
VALUES=(0.005 0.01 0.02)
TAGS=(calib_kthr0p5 calib_kthr1 calib_kthr2)
SEEDS=(9371 9372 9373)
N_SEEDS=${#SEEDS[@]}

VALUE_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))
VALUE="${VALUES[$VALUE_IDX]}"
TAG="${TAGS[$VALUE_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unset}  Array task: ${SLURM_ARRAY_TASK_ID:-unset}  K_rel_threshold=$VALUE  Tag: $TAG  Seed: $SEED"
echo "Assigned GPU: ${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python -u scripts/train.py "$CONFIG" \
  --seed "$SEED" \
  --set method.params.K_rel_threshold="$VALUE" \
  --run-tag "$TAG"
