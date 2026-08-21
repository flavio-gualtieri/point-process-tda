#!/bin/bash

#SBATCH -J pi_multik_k5_calib_q100_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-9
#SBATCH --output=logs/pi_multik_k5_calib_q100_thomas_%A_%a.out
#SBATCH --error=logs/pi_multik_k5_calib_q100_thomas_%A_%a.err

# Thomas HEADLINE model (single-scale DTM_5, H0+H1, PI, shared+concat --
# configs/runs/thomas/pi_multik_k5.yaml, the process-specific scale choice
# of short_report.tex Section~ssec:method, NOT the k=5,10,15 fused testbed
# that pi_multik_calib_q_thomas.sh already swept) at coverage q=1.00 --
# the "naive, uncalibrated" arm in writeup_new.tex's terminology (pooled
# per-diagram extrema with no outlier trimming; see
# axis_bounds/axis_bounds_1d in
# src/cloudforger/calibration/diagram_calibration.py). All 10 seeds are
# fresh: results/experiments.jsonl has no run_tag=calib_q100 rows at all
# under (thomas, dtm_k5, pi_multik) -- the existing calib_q100 arm lives
# under dtm_k5+10+15 (thomas_pi_multik_k5k10k15.yaml), a different
# (fused) config, and only has 3 seeds.
#
# Results land under
# results/thomas/dtm_k5/pi_multik/_runs/calib_q100/seed_<seed>/.
#
# ASSUMES data/thomas/clouds.pkl and data/thomas/dtm_k5/diagrams.pkl
# already exist (they do -- shared with the existing dtm_k5 headline run).
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/pi_multik_k5_calib_q100_thomas.sh

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

python -u scripts/train.py configs/runs/thomas/pi_multik_k5.yaml \
  --seed "$SEED" \
  --set method.params.pd_calibration_coverage=1.00 \
  --run-tag calib_q100
