#!/bin/bash

#SBATCH -J pi_multik_gpu
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/pi_multik_gpu_%j.out
#SBATCH --error=logs/pi_multik_gpu_%j.err

# Submit from the point-process-tda repo root: sbatch slurm/nested_thomas_pi_multik_k5k10k15.sh
# Runs scripts/train.py on configs/runs/nested_thomas/nested_thomas_pi_multik_k5k10k15.yaml.

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

python -u scripts/train.py configs/runs/nested_thomas/pi_multik.yaml
