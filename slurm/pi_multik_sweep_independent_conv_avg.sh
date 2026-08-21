#!/bin/bash

#SBATCH -J pi_multik_sweep_independent_conv_avg
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 01:30:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --output=logs/pi_multik_sweep_independent_conv_avg_%A_%a.out
#SBATCH --error=logs/pi_multik_sweep_independent_conv_avg_%A_%a.err

# pi_multik encoder/fusion sweep -- combo 5/6: encoder_mode=independent,
# fusion_mode=conv, fusion_pool=avg (NEW combination: independent per-k
# encoders + Conv1d-over-k fused by avg-pooling -- see EncoderBank's and
# ConvFusion's docstrings in src/cloudforger/encoders/{encoder_bank,
# scaleconv_pi}.py). nested_thomas, dtm k=5,10,15
# (configs/runs/nested_thomas/pi_multik.yaml).
#
# Longer time limit than the shared-encoder combos: 3 independent per-k
# conv stacks instead of one batch-folded call means roughly 3x the
# per-epoch encoder compute.
#
# 5 seeds via --array=0-4 indexing SEEDS below (first 5 of that config's
# 10-seed list). Submit from the point-process-tda repo root:
#   sbatch slurm/pi_multik_sweep_independent_conv_avg.sh

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

python -u scripts/train.py configs/runs/nested_thomas/pi_multik.yaml \
  --seed "$SEED" \
  --run-tag encsweep_independent_conv_avg \
  --set method.params.encoder_mode=independent \
  --set method.params.fusion_mode=conv \
  --set method.params.fusion_pool=avg
