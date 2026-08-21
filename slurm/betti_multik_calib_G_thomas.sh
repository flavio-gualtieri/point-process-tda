#!/bin/bash

#SBATCH -J betti_multik_calib_G_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-8
#SBATCH --output=logs/betti_multik_calib_G_thomas_%A_%a.out
#SBATCH --error=logs/betti_multik_calib_G_thomas_%A_%a.err

# BC (betti_multik, H0+H1, include_euler=false) grid-resolution calibration
# sweep, thomas, FUSED DTM k=5,10,15. 3 values of G (method.params.grid_size
# -- the Betti-curve sample-grid resolution, betti_multik's "G" equivalent
# -- see betti_multik.py's build_betti_tensor). 3 seeds (9371, 9372, 9373).
# Base config is configs/runs/thomas/thomas_betti_multik_k5k10k15.yaml
# (grid_size=128 baseline, homology_dims=[0,1], pd_calibration_coverage=0.95
# already set there), overridden per task via --set. range_pad is also set
# explicitly to 1.05 on every task -- betti_multik.py's own default
# (range_pad=1.1) differs from the pi_multik calibration sweep's chosen pad
# (1.05), which this sweep inherits per the writeup's calibration note.
#
# Each value gets its own --run-tag, so results land under
# results/thomas/dtm_k5+10+15/betti_multik/_runs/calib_G<value>/seed_<seed>/
# -- separate from the plain (untagged) grid_size=128 baseline and from
# every other sweep value, never overwriting.
#
# 3 values x 3 seeds = 9 tasks, flattened into one array
# (SLURM_ARRAY_TASK_ID = value_index * 3 + seed_index), same pattern as
# pi_multik_calib_q_thomas.sh.
#
# ASSUMES data/thomas/clouds.pkl and data/thomas/dtm_k{5,10,15}/diagrams.pkl
# already exist. If they don't, run once, serially, BEFORE submitting this
# array job:
#   python scripts/generate.py configs/runs/thomas/thomas_betti_multik_k5k10k15.yaml
#   python scripts/featurize.py configs/runs/thomas/thomas_betti_multik_k5k10k15.yaml
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/betti_multik_calib_G_thomas.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

CONFIG=configs/runs/thomas/thomas_betti_multik_k5k10k15.yaml
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
echo "Job ID: ${SLURM_JOB_ID:-unset}  Array task: ${SLURM_ARRAY_TASK_ID:-unset}  grid_size=$VALUE  Tag: $TAG  Seed: $SEED"
echo "Assigned GPU: ${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python -u scripts/train.py "$CONFIG" \
  --seed "$SEED" \
  --set method.params.grid_size="$VALUE" \
  --set method.params.range_pad=1.05 \
  --run-tag "$TAG"
