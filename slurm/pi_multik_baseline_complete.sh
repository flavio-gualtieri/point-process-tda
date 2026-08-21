#!/bin/bash

#SBATCH -J pi_multik_baseline_complete
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-9
#SBATCH --output=logs/pi_multik_baseline_complete_%A_%a.out
#SBATCH --error=logs/pi_multik_baseline_complete_%A_%a.err

# Completes the plain baseline pi_multik run (slurm/run_gpu.sh) that only
# finished 1/10 seeds (9371) before its 59-min limit -- see the loss-
# comparison report. Deliberately NO --run-tag and NO --set overrides here:
# defaults are encoder_mode=shared, fusion_mode=concat, identical to what's
# already on disk for seed 9371, so this writes into that SAME directory
# (results/.../pi_multik/seed_<seed>/, no _runs/ wrapper) instead of
# starting a new one -- "complete", not "replace".
#
# One seed per array task instead of run_gpu.sh's serial-in-one-job, so it
# actually finishes this time; already-done seed 9371 is_done()-skips near-
# instantly. nested_thomas, dtm k=5,10,15
# (configs/runs/nested_thomas/pi_multik.yaml), all 10 configured seeds via
# --array=0-9.
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/pi_multik_baseline_complete.sh

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

python -u scripts/train.py configs/runs/nested_thomas/pi_multik.yaml \
  --seed "$SEED"
