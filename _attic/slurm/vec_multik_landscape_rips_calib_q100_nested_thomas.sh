#!/bin/bash

#SBATCH -J vec_multik_landscape_rips_calib_q100_nested_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-9
#SBATCH --output=logs/vec_multik_landscape_rips_calib_q100_nested_thomas_%A_%a.out
#SBATCH --error=logs/vec_multik_landscape_rips_calib_q100_nested_thomas_%A_%a.err

# Rips-alone arm, nested_thomas, at coverage q=1.00 -- nested_thomas
# counterpart of vec_multik_landscape_rips_calib_q100_thomas.sh; see that
# script's header (and
# configs/runs/nested_thomas/vec_multik_landscape_rips.yaml's) for why
# vec_multik/landscape is used here instead of pi_multik, and why q=1.00
# is reachable at all (no birth axis to be degenerate).
#
# No prior run of vec_multik_landscape_rips.yaml exists at ANY coverage
# for nested_thomas either -- all 10 seeds are fresh here.
#
# Results land under
# results/nested_thomas/rips/vec_multik/_runs/calib_q100/seed_<seed>/.
#
# ASSUMES data/nested_thomas/clouds.pkl and
# data/nested_thomas/rips/diagrams.pkl already exist (they do).
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/vec_multik_landscape_rips_calib_q100_nested_thomas.sh

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
  --seed "$SEED" \
  --set method.params.pd_calibration_coverage=1.00 \
  --run-tag calib_q100
