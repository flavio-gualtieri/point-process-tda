#!/bin/bash

#SBATCH -J betti_multik_fairq_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-5
#SBATCH --output=logs/betti_multik_fairq_thomas_%A_%a.out
#SBATCH --error=logs/betti_multik_fairq_thomas_%A_%a.err

# Inherited-q fairness check, BC arm (betti_multik, H0+H1,
# include_euler=false), thomas, FUSED DTM k=5,10,15. Sibling scripts:
# vec_multik_landscape_fairq_thomas.sh (LS arm),
# vec_multik_silhouette_fairq_thomas.sh (SIL-1 arm) -- see the LS script's
# header for the full fairness-check rationale (3 vectorizations x 2 q
# values = 6 configuration points, each across these same 3 seeds).
# range_pad is set explicitly to 1.05 on every task, same reasoning as
# betti_multik_calib_G_thomas.sh (its own default, 1.1, diverges from the
# pad this check inherits from the pi_multik calibration sweep).
#
# 2 values of q (method.params.pd_calibration_coverage: 0.95, 0.99) x 3
# seeds (9371, 9372, 9373) = 6 tasks, flattened into one array
# (SLURM_ARRAY_TASK_ID = value_index * 3 + seed_index). Base config:
# configs/runs/thomas/thomas_betti_multik_k5k10k15.yaml (grid_size=128,
# homology_dims=[0,1] already set there, untouched).
#
# Each value gets its own --run-tag, so results land under
# results/thomas/dtm_k5+10+15/betti_multik/_runs/fairq<value>/seed_<seed>/
# -- separate from the plain (untagged) q=0.95 baseline and from the other
# sweep value, never overwriting.
#
# ASSUMES data/thomas/clouds.pkl and data/thomas/dtm_k{5,10,15}/diagrams.pkl
# already exist. If they don't, run once, serially, BEFORE submitting this
# array job:
#   python scripts/generate.py configs/runs/thomas/thomas_betti_multik_k5k10k15.yaml
#   python scripts/featurize.py configs/runs/thomas/thomas_betti_multik_k5k10k15.yaml
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/betti_multik_fairq_thomas.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

CONFIG=configs/runs/thomas/thomas_betti_multik_k5k10k15.yaml
VALUES=(0.95 0.99)
TAGS=(fairq095 fairq099)
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
