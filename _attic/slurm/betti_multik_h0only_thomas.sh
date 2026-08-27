#!/bin/bash

#SBATCH -J betti_multik_h0only_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --output=logs/betti_multik_h0only_thomas_%A_%a.out
#SBATCH --error=logs/betti_multik_h0only_thomas_%A_%a.err

# [BC | H0 | DTM{5,10,15} | 5]: homology-dimension ablation (one of the
# three summaries kept -- PI's own H0/H1 arms already exist, see
# pi_multik_h0only_thomas.sh). configs/runs/thomas/
# thomas_betti_multik_k5k10k15.yaml already defaults to include_euler=false
# (BC), homology_dims overridden to [0] alone (beta_0 only, C=1).
#
# BettiMultiKExperiment.subdir already branches on homology_dims -> no
# --run-tag needed: results/thomas/dtm_k5+10+15/betti_multik_h0/seed_<seed>/.
#
# 5 seeds via --array=0-4 (first 5 of the config's 10-seed list).
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/betti_multik_h0only_thomas.sh

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

python -u scripts/train.py configs/runs/thomas/thomas_betti_multik_k5k10k15.yaml \
  --seed "$SEED" \
  --set method.params.homology_dims=[0]
