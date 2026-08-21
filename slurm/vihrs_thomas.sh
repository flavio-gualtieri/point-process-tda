#!/bin/bash

#SBATCH -J vihrs_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --output=logs/vihrs_thomas_%A_%a.out
#SBATCH --error=logs/vihrs_thomas_%A_%a.err

# vihrs baseline (L(r)-r + n(x) 1D-CNN, Vihrs 2022), thomas -- counterpart
# of vihrs_nested_thomas.sh (see that script's header). No filtration/
# features needed: vihrs reads clouds.pkl directly.
# configs/runs/thomas/thomas_vihrs.yaml already exists (checkpoint_best:
# true, same as nested_thomas's), just had no slurm script before this
# batch. Results land under
# results/thomas/raw/vihrs_checkpointed/seed_<seed>/.
#
# 5 seeds via --array=0-4, matching vihrs_nested_thomas.sh's seed count.
#
# ASSUMES data/thomas/clouds.pkl already exists.
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/vihrs_thomas.sh

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

python -u scripts/train.py configs/runs/thomas/thomas_vihrs.yaml \
  --seed "$SEED"
