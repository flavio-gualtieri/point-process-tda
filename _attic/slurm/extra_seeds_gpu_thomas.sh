#!/bin/bash

#SBATCH -J extra_seeds_gpu_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-44
#SBATCH --output=logs/extra_seeds_gpu_thomas_%A_%a.out
#SBATCH --error=logs/extra_seeds_gpu_thomas_%A_%a.err

# Tops every GPU-trained thomas arm currently sitting at n=5 up to n=10,
# by running the 5 seeds (9376-9380) each arm is missing -- same commands
# as their existing n=5 sbatch scripts, just the other half of the seed
# list. Covers: vihrs baseline, logn_only control, and the 7 pi_multik
# run-tag ablation arms (h0only/h1only/coordconv_removed/logn_removed/
# confirm_shared_concat/confirm_independent_concat/shuffled_labels) --
# i.e. everything in Table tab:classical-comparison / tab:branches that
# isn't already at n=10. mincontrast/mincontrast_g are CPU-only jobs and
# are topped up separately by extra_seeds_cpu_thomas.sh.
#
# 9 arms x 5 seeds = 45 tasks, --array=0-44. ARM = task_id / 5 selects the
# arm (case statement below); SEED = SEEDS[task_id % 5].
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/extra_seeds_gpu_thomas.sh

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

PI_CFG=configs/runs/thomas/thomas_pi_multik_k5k10k15.yaml

case "$ARM" in
  0) CMD=(python -u scripts/train.py configs/runs/thomas/thomas_vihrs.yaml --seed "$SEED") ;;
  1) CMD=(python -u scripts/train.py configs/runs/thomas/logn_only.yaml --seed "$SEED") ;;
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
