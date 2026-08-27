#!/bin/bash

#SBATCH -J vec_multik_silhouette_normonly_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --output=logs/vec_multik_silhouette_normonly_thomas_%A_%a.out
#SBATCH --error=logs/vec_multik_silhouette_normonly_thomas_%A_%a.err

# [SIL-1 | H0+H1 | DTM{5,10,15} | 5 | normalized channel only, numerator
# dropped]: silhouette channel ablation. build_silhouette_tensor always
# stacked TWO channels per dim (normalized phi^(p) + its unnormalized
# numerator, on the untested hypothesis that total feature weight matters
# for intensity-like parameters -- see that function's docstring), giving
# silhouette 2*C input channels against every other vectorization's plain
# C. --set method.params.include_unnormalized=false (added for this arm)
# drops the numerator channel, so this arm is both the hypothesis test AND
# the channel-count-matched comparison. p=1 (this config's own default,
# unchanged -- SIL-1 specifically, not the other p arms).
# configs/runs/thomas/vec_multik_silhouette_native.yaml, encoder_path=
# native (unchanged, this config's own default).
#
# VectorizedMultiKExperiment.subdir already appends "_normonly" when
# include_unnormalized=false -> no --run-tag needed:
# results/thomas/dtm_k5+10+15/vec_multik_silhouette_native_p1_normonly/seed_<seed>/.
#
# 5 seeds via --array=0-4 (first 5 of the config's 10-seed list).
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/vec_multik_silhouette_normonly_thomas.sh

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
  --set method.params.include_unnormalized=false
