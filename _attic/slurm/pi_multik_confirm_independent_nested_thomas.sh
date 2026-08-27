#!/bin/bash

#SBATCH -J pi_multik_confirm_independent_nested_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --output=logs/pi_multik_confirm_independent_nested_thomas_%A_%a.out
#SBATCH --error=logs/pi_multik_confirm_independent_nested_thomas_%A_%a.err

# nested_thomas sibling of pi_multik_confirm_independent_thomas.sh -- see
# that file's header, and pi_multik_confirm_shared_nested_thomas.sh's
# header, for the full rationale. encoder_mode=independent,
# fusion_mode=concat, AT THE SELECTED CALIBRATION (sigma_pixels=0.5
# override). FUSED DTM k=5,10,15 (configs/runs/nested_thomas/pi_multik.yaml).
#
# 5 seeds via --array=0-4 (first 5 of the config's 10-seed list).
# --run-tag confirm_independent_concat puts results under
# results/nested_thomas/dtm_k5+10+15/pi_multik/_runs/confirm_independent_concat/seed_<seed>/.
#
# ASSUMES data/nested_thomas/clouds.pkl and
# data/nested_thomas/dtm_k{5,10,15}/diagrams.pkl already exist (shared with
# pi_multik_confirm_shared_nested_thomas.sh -- if that script's array
# already ran, they do).
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/pi_multik_confirm_independent_nested_thomas.sh

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

python -u scripts/train.py configs/runs/nested_thomas/pi_multik.yaml \
  --seed "$SEED" \
  --run-tag confirm_independent_concat \
  --set method.params.sigma_pixels=0.5 \
  --set method.params.encoder_mode=independent \
  --set method.params.fusion_mode=concat
