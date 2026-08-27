#!/bin/bash

#SBATCH -J pi_multik_headline_nested_thomas_fill
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-6
#SBATCH --output=logs/pi_multik_headline_nested_thomas_fill_%A_%a.out
#SBATCH --error=logs/pi_multik_headline_nested_thomas_fill_%A_%a.err

# Fills in the MISSING headline run: nested_thomas, pi_multik, fused DTM
# k=5,10,15, H0+H1, at the CURRENT default calibration
# (configs/runs/nested_thomas/pi_multik.yaml already has sigma_pixels: 0.5
# baked in -- no --set override needed). No untagged run of this exact
# config exists anywhere in results/ right now (verified against
# results/experiments.jsonl: zero run_tag=None rows for
# (nested_thomas, pi_multik, dtm_k5+10+15)). The only prior full-seed run
# of this config lives in results_exp/'s Aug-11 archive at the OLD default
# sigma_pixels=0.75 (mean 0.205, not comparable); the only current-
# calibration data is 3 seeds from the calib_sigma05 sweep tag
# (seeds 9371-9373, mean ~0.191).
#
# This script runs the other 7 seeds (9374-9380) UNTAGGED (no --run-tag),
# so together with the existing calib_sigma05 seeds they should be
# manually consolidated (or the 3 calib_sigma05 seeds re-tagged/copied) to
# get a true, clean n=10 at results/nested_thomas/dtm_k5+10+15/pi_multik/
# seed_<seed>/. Cross-check against calib_sigma05's seeds 9371-9373
# (0.181, 0.201, 0.192) once these land -- they should be consistent with
# these 7 new points since it's the identical config, just seeds run
# separately.
#
# 7 seeds via --array=0-6.
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/pi_multik_headline_nested_thomas_fill.sh

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

python -u scripts/train.py configs/runs/nested_thomas/pi_multik.yaml --seed "$SEED"
