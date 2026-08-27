#!/bin/bash

#SBATCH -J pi_multik_rips_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-9
#SBATCH --output=logs/pi_multik_rips_thomas_%A_%a.out
#SBATCH --error=logs/pi_multik_rips_thomas_%A_%a.err

# Trains pi_multik on configs/runs/thomas/pi_multik_rips.yaml (rips alone),
# all 10 configured seeds, one per array task -- same array-per-seed
# pattern as betti_multik_thomas.sh.
#
# EXPECTED TO FAIL, on purpose: build_calibrated_imager's axis_bounds
# rejects a degenerate H0 birth axis ("H0 birth axis is degenerate; use a
# 1-D vectorizer"), and plain Rips gives every H0 feature birth=0 -- see
# pi_multik_rips.yaml's own header. Every task should die fast (at the
# calibration step inside train.py, before any real GPU work), each
# leaving that ValueError in its logs/pi_multik_rips_thomas_*.err file --
# that traceback IS the result this job exists to produce, not a bug to
# chase. Compare against betti_cnn_sweep_thomas.sh's rips configs, which
# train fine on the identical filtration (Betti curves have no birth axis
# of their own -- see vectorization/scalar_features/calibrated.py's module
# docstring).
#
# ASSUMES data/thomas/clouds.pkl and data/thomas/rips/diagrams.pkl already
# exist. clouds.pkl almost certainly does (shared across every thomas
# config); rips/diagrams.pkl almost certainly does NOT (no existing thomas
# config used rips before betti_cnn_rips_h0.yaml/pi_multik_rips.yaml). If
# missing, run once, serially, BEFORE submitting this array job (concurrent
# array tasks writing the same diagrams.pkl would race):
#   python scripts/generate.py  configs/runs/thomas/pi_multik_rips.yaml
#   python scripts/featurize.py configs/runs/thomas/pi_multik_rips.yaml
# (the featurize.py step itself will succeed -- diagram computation has no
# birth-axis concern; only pi_multik's own image calibration, inside
# train.py, hits the ValueError above.)
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/pi_multik_rips_thomas.sh

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

python -u scripts/train.py configs/runs/thomas/pi_multik_rips.yaml \
  --seed "$SEED"
