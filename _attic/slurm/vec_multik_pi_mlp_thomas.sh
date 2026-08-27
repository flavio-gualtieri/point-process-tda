#!/bin/bash

#SBATCH -J vec_multik_pi_mlp_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --output=logs/vec_multik_pi_mlp_thomas_%A_%a.out
#SBATCH --error=logs/vec_multik_pi_mlp_thomas_%A_%a.err

# [PI | H0+H1 | DTM{5,10,15} | 5 | encoder_path = mlp] -- the
# architecture-agnostic control: same calibrated persistence images as the
# headline pi_multik run (configs/runs/thomas/thomas_pi_multik_k5k10k15.yaml),
# but flattened through the shared FlattenMLPEncoder primary path instead of
# CoordConvPIEncoder (method: vec_multik, vectorization: persistence_image,
# encoder_path: mlp -- see configs/runs/thomas/vec_multik_pi_mlp.yaml's
# header). If the pi_multik vs vec_multik-mlp gap is small, the summary (not
# the CNN) is doing the work. Own subdir (vec_multik_persistence_image_mlp,
# from VectorizedMultiKExperiment.subdir), so no --run-tag is needed --
# results/thomas/dtm_k5+10+15/vec_multik_persistence_image_mlp/seed_<seed>/.
#
# 5 seeds via --array=0-4 (first 5 of the config's 10-seed list).
#
# ASSUMES data/thomas/clouds.pkl and data/thomas/dtm_k{5,10,15}/diagrams.pkl
# already exist (shared with every other thomas pi_multik config).
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/vec_multik_pi_mlp_thomas.sh

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

python -u scripts/train.py configs/runs/thomas/vec_multik_pi_mlp.yaml \
  --seed "$SEED"
