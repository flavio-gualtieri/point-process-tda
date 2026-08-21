#!/bin/bash

#SBATCH -J betti_multik_nested_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-9
#SBATCH --output=logs/betti_multik_nested_thomas_%A_%a.out
#SBATCH --error=logs/betti_multik_nested_thomas_%A_%a.err

# Trains betti_multik (experiments/pi_multik/betti_multik.py) on
# configs/runs/nested_thomas/betti_multik.yaml, all 10 configured seeds,
# one per array task -- same array-per-seed pattern as
# pi_multik_baseline_complete.sh / betti_multik_thomas.sh (see that file's
# header for why).
#
# REPLACEMENT RUN: the config now sets method.params.include_euler: false
# (previously implicit true -- beta_0/beta_1 stacked, no derived
# chi = beta_0 - beta_1 channel). The old include_euler:true results were
# deleted from results/nested_thomas/dtm_k5+10+15/betti_multik/ and their
# 10 rows pruned from results/experiments.jsonl, so is_done() sees this as
# not-yet-run and every seed retrains fresh into that same path.
#
# ASSUMES data/nested_thomas/clouds.pkl and
# data/nested_thomas/dtm_k{5,10,15}/diagrams.pkl already exist
# (betti_multik.yaml reads the identical process/design/filtration grid as
# pi_multik.yaml, so whichever of the two you ran generate.py/featurize.py
# against already produced them for both). If they don't exist yet, run
# once, serially, BEFORE submitting this array job (concurrent array tasks
# writing the same clouds.pkl/diagrams.pkl would race):
#   python scripts/generate.py configs/runs/nested_thomas/betti_multik.yaml
#   python scripts/featurize.py configs/runs/nested_thomas/betti_multik.yaml
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/betti_multik_nested_thomas.sh

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

python -u scripts/train.py configs/runs/nested_thomas/betti_multik.yaml \
  --seed "$SEED"
