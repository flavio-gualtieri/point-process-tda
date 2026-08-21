#!/bin/bash

#SBATCH -J pi_multik_sweep_shared_conv_flatten
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --output=logs/pi_multik_sweep_shared_conv_flatten_%A_%a.out
#SBATCH --error=logs/pi_multik_sweep_shared_conv_flatten_%A_%a.err

# pi_multik encoder/fusion sweep -- combo 3/6: encoder_mode=shared,
# fusion_mode=conv, fusion_pool=flatten (NEW combination, not reachable
# under the old file-per-variant design: shared-weight encoder, Conv1d over
# the k axis, k positions flattened rather than pooled -- see ConvFusion's
# docstring in src/cloudforger/encoders/scaleconv_pi.py). nested_thomas,
# dtm k=5,10,15 (configs/runs/nested_thomas/pi_multik.yaml).
#
# 5 seeds via --array=0-4 indexing SEEDS below (first 5 of that config's
# 10-seed list). Submit from the point-process-tda repo root:
#   sbatch slurm/pi_multik_sweep_shared_conv_flatten.sh

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
  --run-tag encsweep_shared_conv_flatten \
  --set method.params.encoder_mode=shared \
  --set method.params.fusion_mode=conv \
  --set method.params.fusion_pool=flatten
