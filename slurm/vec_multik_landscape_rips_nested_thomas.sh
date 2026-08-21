#!/bin/bash

#SBATCH -J vec_multik_landscape_rips_nested_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-9
#SBATCH --output=logs/vec_multik_landscape_rips_nested_thomas_%A_%a.out
#SBATCH --error=logs/vec_multik_landscape_rips_nested_thomas_%A_%a.err

# Trains vec_multik (vectorization=landscape, encoder_path=native) on rips
# alone, all 10 configured seeds, one per array task -- nested_thomas
# counterpart of vec_multik_landscape_rips_thomas.sh; see that script's
# header (and configs/runs/nested_thomas/vec_multik_landscape_rips.yaml's)
# for why this trains where pi_multik_rips_nested_thomas.sh is expected to
# fail.
#
# ASSUMES data/nested_thomas/clouds.pkl and
# data/nested_thomas/rips/diagrams.pkl already exist (they do --
# pi_multik_rips_nested_thomas.sh's featurize step and/or
# betti_cnn_rips_h0.yaml's earlier run already produced rips/diagrams.pkl;
# clouds.pkl is shared across every nested_thomas config). If somehow
# missing, run once, serially, BEFORE submitting this array job (concurrent
# array tasks writing the same diagrams.pkl would race):
#   python scripts/generate.py  configs/runs/nested_thomas/vec_multik_landscape_rips.yaml
#   python scripts/featurize.py configs/runs/nested_thomas/vec_multik_landscape_rips.yaml
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/vec_multik_landscape_rips_nested_thomas.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

SEEDS=(9371 9372 9373 9374 9375 9376 9377 9378 9379 9380)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unset}  Array task: ${SLURM_ARRAY_TASK_ID:-unset}  Seed: $SEED"
echo "Assigned GPU: ${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python -u scripts/train.py configs/runs/nested_thomas/vec_multik_landscape_rips.yaml \
  --seed "$SEED"
