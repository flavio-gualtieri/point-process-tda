#!/bin/bash

#SBATCH -J dtm_train
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/dtm_train_%j.out
#SBATCH --error=logs/dtm_train_%j.err

# Submit from the point-process-tda repo root: sbatch dtm_experiment/run_train_gpu.sh
# Runs dtm_experiment/train.py (3 seeds x {betti_cnn_0, betti_cnn_1, pi_0, pi_1,
# vihrs}) on a GPU. Resumable: train.py skips any seed/method whose
# results.pt already exists, so a re-submit after hitting the walltime just
# picks up where it left off.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unset}"
echo "Assigned GPU: ${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python dtm_experiment/train.py
