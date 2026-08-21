#!/bin/bash

#SBATCH -J pi_multik_singlescale_nested_thomas_k15
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-9
#SBATCH --output=logs/pi_multik_singlescale_nested_thomas_k15_%A_%a.out
#SBATCH --error=logs/pi_multik_singlescale_nested_thomas_k15_%A_%a.err

# Single-scale reference run: nested_thomas, pi_multik (shared encoder +
# flat concat -- the defaults, no encoder_mode/fusion_mode/fusion_pool
# overrides), dtm k=15 ONLY (configs/runs/nested_thomas/pi_multik_k15.yaml --
# filtration: has just this one entry, so k_values auto-derives to [15],
# n_k=1). All 10 configured seeds via --array=0-9. No --run-tag needed --
# results land under results/nested_thomas/dtm_k15/pi_multik/seed_<seed>/,
# separate from the k5,10,15 fused run's dtm_k5+10+15/pi_multik/.
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/pi_multik_singlescale_nested_thomas_k15.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

SEEDS=(9371 9372 9373 9374 9375 9376 9377 9378 9379 9380)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unset}  Array task: ${SLURM_ARRAY_TASK_ID:-unset}  Seed: $SEED"
echo "Assigned GPU: ${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python -u scripts/train.py configs/runs/nested_thomas/pi_multik_k15.yaml --seed "$SEED"
