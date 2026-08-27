#!/bin/bash

#SBATCH -J vec_multik_landscape_calib_kthr_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-8
#SBATCH --output=logs/vec_multik_landscape_calib_kthr_thomas_%A_%a.out
#SBATCH --error=logs/vec_multik_landscape_calib_kthr_thomas_%A_%a.err

# K-rule relative-threshold sweep, thomas, FUSED DTM k=5,10,15 (H0+H1,
# native CNN encoder), 3 values of the coverage rule's sup-norm relative
# threshold (method.params.K_rel_threshold -- calibrated.py's choose_K:
# smallest K s.t. ||lambda_{K+1}||_inf <= K_rel_threshold * ||lambda_1||_inf
# for >= K_coverage of training diagrams, capped at K_cap). K itself is
# LEFT UNSET (data-derived) in every arm -- this sweep is over the rule's
# threshold, not a fixed K. K_rel_threshold defaulted to 0.01 (1%) and was
# hardcoded until this sweep's --set exposed it (see calibrated.py's
# choose_K/build_calibrated_landscape and vectorized_multik.py's
# build_landscape_tensor/_build_tensor).
# 3 seeds (9371, 9372, 9373).
#
# Base config is the existing configs/runs/thomas/vec_multik_landscape_native.yaml
# (G=128). Its pd_calibration_coverage: 0.95 is the persistence-diagram
# grid-bounds coverage q (axis_bounds_1d), a separate knob from K's own
# K_coverage (defaults to 0.99, untouched by this sweep -- see
# calibrated.py's build_calibrated_landscape). Overridden per task via --set.
#
# Each value gets its own --run-tag, so results land under
# results/thomas/dtm_k5+10+15/vec_multik_landscape_native/_runs/calib_kthr<value>/seed_<seed>/
# -- separate from the plain (untagged) baseline results and from every
# other sweep value, never overwriting.
#
# 3 values x 3 seeds = 9 tasks, flattened into one array
# (SLURM_ARRAY_TASK_ID = value_index * 3 + seed_index), same pattern as
# vec_multik_landscape_calib_G_thomas.sh.
#
# ASSUMES data/thomas/clouds.pkl and data/thomas/dtm_k{5,10,15}/diagrams.pkl
# already exist (they do).
# Submit from the point-process-tda repo root:
#   sbatch slurm/vec_multik_landscape_calib_kthr_thomas.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

CONFIG=configs/runs/thomas/vec_multik_landscape_native.yaml
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
