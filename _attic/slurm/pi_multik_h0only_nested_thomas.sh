#!/bin/bash

#SBATCH -J pi_multik_h0only_nested_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --output=logs/pi_multik_h0only_nested_thomas_%A_%a.out
#SBATCH --error=logs/pi_multik_h0only_nested_thomas_%A_%a.err

# [PI | H0 | DTM{5,10,15} | 5]: nested_thomas counterpart of
# pi_multik_h0only_thomas.sh -- see that file's header for the full
# rationale. configs/runs/nested_thomas/pi_multik.yaml, AT THE NEWLY
# ESTABLISHED CALIBRATION already baked into that config.
#
# --run-tag h0only ->
# results/nested_thomas/dtm_k5+10+15/pi_multik/_runs/h0only/seed_<seed>/.
#
# 5 seeds via --array=0-4 (first 5 of the config's 10-seed list).
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/pi_multik_h0only_nested_thomas.sh

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
  --run-tag h0only \
  --set method.params.homology_dims=[0]
