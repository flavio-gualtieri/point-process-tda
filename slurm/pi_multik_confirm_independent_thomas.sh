#!/bin/bash

#SBATCH -J pi_multik_confirm_independent_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --output=logs/pi_multik_confirm_independent_thomas_%A_%a.out
#SBATCH --error=logs/pi_multik_confirm_independent_thomas_%A_%a.err

# PI confirmation pass, combo 2/2: encoder_mode=independent,
# fusion_mode=concat (n_k separate per-k CoordConvPIEncoder instances,
# still flat-concat fused -- see EncoderBank's docstring), AT THE SELECTED
# CALIBRATION -- see pi_multik_confirm_shared_thomas.sh's header for the
# full resolution/sigma_pixels/pad/coverage rationale (identical here,
# only encoder_mode differs). thomas, FUSED DTM k=5,10,15
# (configs/runs/thomas/thomas_pi_multik_k5k10k15.yaml).
#
# 5 seeds via --array=0-4 (first 5 of the config's 10-seed list).
# --run-tag confirm_independent_concat puts results under
# results/thomas/dtm_k5+10+15/pi_multik/_runs/confirm_independent_concat/seed_<seed>/.
#
# ASSUMES data/thomas/clouds.pkl and data/thomas/dtm_k{5,10,15}/diagrams.pkl
# already exist (shared with pi_multik_confirm_shared_thomas.sh -- if that
# script's array already ran, they do).
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/pi_multik_confirm_independent_thomas.sh

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
  --run-tag confirm_independent_concat \
  --set method.params.sigma_pixels=0.5 \
  --set method.params.encoder_mode=independent \
  --set method.params.fusion_mode=concat
