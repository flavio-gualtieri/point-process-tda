#!/bin/bash

#SBATCH -J pi_multik_confirm_shared_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --output=logs/pi_multik_confirm_shared_thomas_%A_%a.out
#SBATCH --error=logs/pi_multik_confirm_shared_thomas_%A_%a.err

# PI confirmation pass, combo 1/2: encoder_mode=shared, fusion_mode=concat
# (the pi_multik baseline architecture -- see
# src/cloudforger/experiments/pi_multik/pi_multik.py's PIMultiK docstring),
# AT THE SELECTED CALIBRATION closing the greedy calibration-sweep loop:
# resolution=64 (config default, untouched), sigma_pixels=0.5 (overridden
# below -- the base config's own baseline is 0.75; 0.5 is the pi_multik
# calibration sweep's chosen value), pad=1.05 (pi_multik.py's own default,
# untouched), pd_calibration_coverage=0.95 (config default, untouched).
# thomas, FUSED DTM k=5,10,15
# (configs/runs/thomas/thomas_pi_multik_k5k10k15.yaml).
#
# Sibling: pi_multik_confirm_independent_thomas.sh (encoder_mode=independent).
# Together these 2 configs x {thomas, nested_thomas} are the confirmation
# pass's 4 runs.
#
# 5 seeds via --array=0-4 indexing SEEDS below (first 5 of the config's
# 10-seed list) -- same convention as pi_multik_sweep_shared_concat.sh.
# --run-tag confirm_shared_concat (distinct from that other script's
# encsweep_shared_concat tag -- different sigma_pixels, must not share a
# results path) puts results under
# results/thomas/dtm_k5+10+15/pi_multik/_runs/confirm_shared_concat/seed_<seed>/.
#
# ASSUMES data/thomas/clouds.pkl and data/thomas/dtm_k{5,10,15}/diagrams.pkl
# already exist. If they don't, run once, serially, BEFORE submitting this
# array job:
#   python scripts/generate.py configs/runs/thomas/thomas_pi_multik_k5k10k15.yaml
#   python scripts/featurize.py configs/runs/thomas/thomas_pi_multik_k5k10k15.yaml
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/pi_multik_confirm_shared_thomas.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

SEEDS=(9371 9372 9373 9374 9375)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unset}  Array task: ${SLURM_ARRAY_TASK_ID:-unset}  Seed: $SEED"
echo "Assigned GPU: ${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python -u scripts/train.py configs/runs/thomas/thomas_pi_multik_k5k10k15.yaml \
  --seed "$SEED" \
  --run-tag confirm_shared_concat \
  --set method.params.sigma_pixels=0.5 \
  --set method.params.encoder_mode=shared \
  --set method.params.fusion_mode=concat
