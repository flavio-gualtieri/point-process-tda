#!/bin/bash

#SBATCH -J betti_multik_calib_q_nested_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-8
#SBATCH --output=logs/betti_multik_calib_q_nested_thomas_%A_%a.out
#SBATCH --error=logs/betti_multik_calib_q_nested_thomas_%A_%a.err

# nested_thomas sibling of betti_multik_calib_q_thomas.sh -- see that
# file's header for the full grid/results-path/range_pad explanation.
# BC (betti_multik, H0+H1, include_euler=false), FUSED DTM k=5,10,15,
# 3 values of q (method.params.pd_calibration_coverage) x 3 seeds
# (9371, 9372, 9373), range_pad fixed at 1.05 on every task.
# Base config: configs/runs/nested_thomas/betti_multik.yaml.
#
# ASSUMES data/nested_thomas/clouds.pkl and
# data/nested_thomas/dtm_k{5,10,15}/diagrams.pkl already exist. If they
# don't, run once, serially, BEFORE submitting this array job:
#   python scripts/generate.py configs/runs/nested_thomas/betti_multik.yaml
#   python scripts/featurize.py configs/runs/nested_thomas/betti_multik.yaml
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/betti_multik_calib_q_nested_thomas.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

CONFIG=configs/runs/nested_thomas/betti_multik.yaml
VALUES=(0.95 0.99 1.00)
TAGS=(calib_q095 calib_q099 calib_q100)
SEEDS=(9371 9372 9373)
N_SEEDS=${#SEEDS[@]}

VALUE_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))
VALUE="${VALUES[$VALUE_IDX]}"
TAG="${TAGS[$VALUE_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unset}  Array task: ${SLURM_ARRAY_TASK_ID:-unset}  pd_calibration_coverage=$VALUE  Tag: $TAG  Seed: $SEED"
echo "Assigned GPU: ${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python -u scripts/train.py "$CONFIG" \
  --seed "$SEED" \
  --set method.params.pd_calibration_coverage="$VALUE" \
  --set method.params.range_pad=1.05 \
  --run-tag "$TAG"
