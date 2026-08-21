#!/bin/bash

#SBATCH -J logn_only_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:29:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --output=logs/logn_only_thomas_%A_%a.out
#SBATCH --error=logs/logn_only_thomas_%A_%a.err

# [none (log N only) | -- | -- | 5]: the intensity-only baseline -- the
# cheapest and most-asked-for control. method: logn_only (new -- see
# src/cloudforger/experiments/logn_only.py's module docstring), reads
# data/thomas/clouds.pkl directly, no diagrams/persistence images/
# filtration of any kind: an MLP regresses targets from log N(x) alone.
# Paired with slurm/pi_multik_logn_removed_thomas.sh (the complementary
# "topology minus log N" arm) -- together they bound how much of pi_multik's
# performance is attributable to topology vs. just counting points.
#
# No filtration config -> results land under
# results/thomas/raw/logn_only/seed_<seed>/ (not nested under
# dtm_k5+10+15/ -- see configs/runs/thomas/logn_only.yaml's header). No
# --run-tag needed.
#
# 5 seeds via --array=0-4 (first 5 of the config's 10-seed list). 29-minute
# time limit (vs. every other pi_multik-family script's 59) -- this model
# is a 1-input MLP, converges in a couple hundred epochs at most; kept on a
# GPU node for consistency with every other arm in this batch, but it does
# not need one.
#
# ASSUMES data/thomas/clouds.pkl and data/thomas/adversarial_clouds.pkl (if
# use_adversarial) already exist.
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/logn_only_thomas.sh

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

python -u scripts/train.py configs/runs/thomas/logn_only.yaml \
  --seed "$SEED"
