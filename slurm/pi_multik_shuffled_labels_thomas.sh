#!/bin/bash

#SBATCH -J pi_multik_shuffled_labels_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --output=logs/pi_multik_shuffled_labels_thomas_%A_%a.out
#SBATCH --error=logs/pi_multik_shuffled_labels_thomas_%A_%a.err

# [PI | H0+H1 | DTM{5,10,15} | 5 | shuffled labels]: leakage sanity check --
# should return L~=1 (the "predicts nothing beyond the design-distribution
# mean" reference point, see the writeup's loss-definition section).
# method.params.shuffle_labels=true permutes targets against every other
# per-cloud field with a seed-keyed RNG, BEFORE the train/val/test split
# (see src/cloudforger/experiments/pi_multik/pi_multik.py's run(),
# shuffle_labels flag added for this arm -- default false leaves every
# other run unchanged). If this arm's test loss comes in well below ~1, that
# is evidence of a leakage bug elsewhere in the pipeline, not of a model
# that generalizes from noise. thomas, FUSED DTM k=5,10,15
# (configs/runs/thomas/thomas_pi_multik_k5k10k15.yaml), AT THE NEWLY
# ESTABLISHED CALIBRATION already baked into that config.
#
# --run-tag shuffled_labels ->
# results/thomas/dtm_k5+10+15/pi_multik/_runs/shuffled_labels/seed_<seed>/.
#
# 5 seeds via --array=0-4 (first 5 of the config's 10-seed list).
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/pi_multik_shuffled_labels_thomas.sh

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

python -u scripts/train.py configs/runs/thomas/thomas_pi_multik_k5k10k15.yaml \
  --seed "$SEED" \
  --run-tag shuffled_labels \
  --set method.params.shuffle_labels=true
