#!/bin/bash

#SBATCH -J vec_multik_silhouette_mlp_p1_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --output=logs/vec_multik_silhouette_mlp_p1_thomas_%A_%a.out
#SBATCH --error=logs/vec_multik_silhouette_mlp_p1_thomas_%A_%a.err

# [SIL-1 | H0+H1 | DTM{5,10,15} | 5 | mlp]: architecture-agnostic control
# for silhouettes, p=1 (the config's own default -- no --set p= needed).
# Sibling of vec_multik_silhouette_thomas.sh (encoder_path=native).
# configs/runs/thomas/vec_multik_silhouette_native.yaml, encoder_path
# overridden -> results/thomas/dtm_k5+10+15/vec_multik_silhouette_mlp_p1/seed_<seed>/.
#
# 5 seeds via --array=0-4 (first 5 of the config's 10-seed list).
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/vec_multik_silhouette_mlp_p1_thomas.sh

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

python -u scripts/train.py configs/runs/thomas/vec_multik_silhouette_native.yaml \
  --seed "$SEED" \
  --set method.params.encoder_path=mlp
