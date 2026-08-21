#!/bin/bash

#SBATCH -J extra_seeds_gpu_nested_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-44
#SBATCH --output=logs/extra_seeds_gpu_nested_thomas_%A_%a.out
#SBATCH --error=logs/extra_seeds_gpu_nested_thomas_%A_%a.err

# nested-Thomas counterpart of extra_seeds_gpu_thomas.sh -- see that
# script's header for the full rationale. Same 9 arms (vihrs, logn_only,
# and the 7 pi_multik run-tag ablations), same missing seeds (9376-9380),
# just against configs/runs/nested_thomas/. No mincontrast counterpart:
# nested Thomas has no closed-form K or g to fit against.
#
# 9 arms x 5 seeds = 45 tasks, --array=0-44.
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/extra_seeds_gpu_nested_thomas.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

SEEDS=(9376 9377 9378 9379 9380)
ARM=$(( SLURM_ARRAY_TASK_ID / 5 ))
SIDX=$(( SLURM_ARRAY_TASK_ID % 5 ))
SEED="${SEEDS[$SIDX]}"

PI_CFG=configs/runs/nested_thomas/pi_multik.yaml

case "$ARM" in
  0) CMD=(python -u scripts/train.py configs/runs/nested_thomas/nested_thomas_vihrs.yaml --seed "$SEED") ;;
  1) CMD=(python -u scripts/train.py configs/runs/nested_thomas/logn_only.yaml --seed "$SEED") ;;
  2) CMD=(python -u scripts/train.py "$PI_CFG" --seed "$SEED" \
          --run-tag h0only --set method.params.homology_dims=[0]) ;;
  3) CMD=(python -u scripts/train.py "$PI_CFG" --seed "$SEED" \
          --run-tag h1only --set method.params.homology_dims=[1]) ;;
  4) CMD=(python -u scripts/train.py "$PI_CFG" --seed "$SEED" \
          --run-tag coordconv_removed --set method.params.coordconv=false) ;;
  5) CMD=(python -u scripts/train.py "$PI_CFG" --seed "$SEED" \
          --run-tag logn_removed --set method.params.include_log_n=false) ;;
  6) CMD=(python -u scripts/train.py "$PI_CFG" --seed "$SEED" \
          --run-tag confirm_shared_concat \
          --set method.params.sigma_pixels=0.5 \
          --set method.params.encoder_mode=shared \
          --set method.params.fusion_mode=concat) ;;
  7) CMD=(python -u scripts/train.py "$PI_CFG" --seed "$SEED" \
          --run-tag confirm_independent_concat \
          --set method.params.sigma_pixels=0.5 \
          --set method.params.encoder_mode=independent \
          --set method.params.fusion_mode=concat) ;;
  8) CMD=(python -u scripts/train.py "$PI_CFG" --seed "$SEED" \
          --run-tag shuffled_labels --set method.params.shuffle_labels=true) ;;
  *) echo "bad ARM index: $ARM" >&2; exit 1 ;;
esac

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unset}  Array task: ${SLURM_ARRAY_TASK_ID:-unset}  Arm: $ARM  Seed: $SEED"
echo "Assigned GPU: ${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Command: ${CMD[*]}"

"${CMD[@]}"
