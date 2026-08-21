#!/bin/bash

#SBATCH -J pi_multik_calib_q100_nested_thomas_fill
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-6
#SBATCH --output=logs/pi_multik_calib_q100_nested_thomas_fill_%A_%a.out
#SBATCH --error=logs/pi_multik_calib_q100_nested_thomas_fill_%A_%a.err

# FILLS IN nested_thomas's HEADLINE model (fused DTM_{5,10,15}, H0+H1, PI,
# shared+concat -- configs/runs/nested_thomas/pi_multik.yaml, already the
# process-specific scale choice, so no filtration override needed here
# unlike the thomas arm) at coverage q=1.00, the "naive, uncalibrated" arm.
#
# pi_multik_calib_q_nested_thomas.sh already ran this exact
# config+run-tag (calib_q100) for seeds 9371-9373 (3 of the 4 swept q
# values x 3 seeds each). This script runs the other 7 seeds
# (9374-9380) under the SAME run-tag, so together they consolidate into a
# clean n=10 at
# results/nested_thomas/dtm_k5+10+15/pi_multik/_runs/calib_q100/seed_<seed>/
# -- same "_fill" pattern as pi_multik_headline_nested_thomas_fill.sh.
#
# ASSUMES data/nested_thomas/clouds.pkl and
# data/nested_thomas/dtm_k{5,10,15}/diagrams.pkl already exist (they do).
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/pi_multik_calib_q100_nested_thomas_fill.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

SEEDS=(9374 9375 9376 9377 9378 9379 9380)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unset}  Array task: ${SLURM_ARRAY_TASK_ID:-unset}  Seed: $SEED"
echo "Assigned GPU: ${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python -u scripts/train.py configs/runs/nested_thomas/pi_multik.yaml \
  --seed "$SEED" \
  --set method.params.pd_calibration_coverage=1.00 \
  --run-tag calib_q100
