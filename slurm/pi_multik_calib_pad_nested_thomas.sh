#!/bin/bash

#SBATCH -J pi_multik_calib_pad_nested_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-8
#SBATCH --output=logs/pi_multik_calib_pad_nested_thomas_%A_%a.out
#SBATCH --error=logs/pi_multik_calib_pad_nested_thomas_%A_%a.err

# PI calibration padding-factor sweep, nested_thomas, FUSED DTM k=5,10,15 (one
# 3-channel multi-scale model per grid point -- shared-weight encoder + flat
# concat fusion, the pi_multik defaults -- NOT three independent single-k
# models), 3 values of padding factor alpha (method.params.pad, threaded
# through build_calibrated_imager -> axis_bounds -- newly wired for this
# sweep, see vectorization/persistence_images/calibrated.py; default 1.05
# unchanged for every config that doesn't set it). 3 seeds (9371, 9372, 9373).
# Base config is the existing configs/runs/nested_thomas/pi_multik.yaml (resolution=64, sigma_pixels=0.75,
# pd_calibration_coverage=0.95, pad=1.05 baseline, no encoder_mode/fusion_mode
# override -> PIMultiKExperiment._build_model's defaults, shared encoder +
# concat fusion -- see its header/docstring), overridden per task via --set
# (config.py's apply_overrides).
#
# Each value gets its own --run-tag, so results land under
# results/nested_thomas/dtm_k5+10+15/pi_multik/_runs/calib_pad<value>/seed_<seed>/
# -- separate from the plain (untagged) baseline results and from every
# other sweep value, never overwriting.
#
# 3 values x 3 seeds = 9 tasks, flattened into one
# array (SLURM_ARRAY_TASK_ID = value_index * 3 + seed_index), same
# pattern as betti_multik_dims_thomas.sh / betti_cnn_sweep_thomas.sh.
#
# ASSUMES data/nested_thomas/clouds.pkl and data/nested_thomas/dtm_k{5,10,15}/diagrams.pkl
# already exist (they do).
# Submit from the point-process-tda repo root:
#   sbatch slurm/pi_multik_calib_pad_nested_thomas.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

CONFIG=configs/runs/nested_thomas/pi_multik.yaml
VALUES=(1.00 1.05 1.10)
TAGS=(calib_pad100 calib_pad105 calib_pad110)
SEEDS=(9371 9372 9373)
N_SEEDS=${#SEEDS[@]}

VALUE_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))
VALUE="${VALUES[$VALUE_IDX]}"
TAG="${TAGS[$VALUE_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unset}  Array task: ${SLURM_ARRAY_TASK_ID:-unset}  pad=$VALUE  Tag: $TAG  Seed: $SEED"
echo "Assigned GPU: ${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python -u scripts/train.py "$CONFIG" \
  --seed "$SEED" \
  --set method.params.pad="$VALUE" \
  --run-tag "$TAG"
