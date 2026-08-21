#!/bin/bash

#SBATCH -J vec_multik_landscape_rips_calib_q100_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-9
#SBATCH --output=logs/vec_multik_landscape_rips_calib_q100_thomas_%A_%a.out
#SBATCH --error=logs/vec_multik_landscape_rips_calib_q100_thomas_%A_%a.err

# Rips-alone arm, thomas, at coverage q=1.00 (the "naive, uncalibrated"
# point in writeup_new.tex's terminology). Uses vec_multik
# (vectorization=landscape, encoder_path=native) rather than pi_multik:
# plain Rips gives every H0 feature birth=0, which is degenerate for
# ANY coverage value under pi_multik's 2-D birth x persistence calibration
# (axis_bounds always sees birth_hi<=birth_lo=0, q doesn't change that --
# see pi_multik_rips_thomas.sh's header) -- q is therefore not swept for
# pi_multik+rips at all, there is no q at which it trains. Landscape's 1-D
# [t_min, T] calibration (axis_bounds_1d) has no birth-axis of its own, so
# it trains at q=1.00 same as at the 0.95 default -- see
# configs/runs/thomas/vec_multik_landscape_rips.yaml's header for the full
# story of why this is Rips's one working arm.
#
# No prior run of vec_multik_landscape_rips.yaml exists at ANY coverage
# (checked results/ and results_exp/experiments.jsonl -- zero rows with
# filtration_tag=rips and method=vec_multik) -- all 10 seeds are fresh here.
#
# Results land under
# results/thomas/rips/vec_multik/_runs/calib_q100/seed_<seed>/.
#
# ASSUMES data/thomas/clouds.pkl and data/thomas/rips/diagrams.pkl already
# exist (they do -- produced by pi_multik_rips_thomas.sh's featurize step /
# betti_cnn_rips_h0.yaml's earlier run).
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/vec_multik_landscape_rips_calib_q100_thomas.sh

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

python -u scripts/train.py configs/runs/thomas/vec_multik_landscape_rips.yaml \
  --seed "$SEED" \
  --set method.params.pd_calibration_coverage=1.00 \
  --run-tag calib_q100
