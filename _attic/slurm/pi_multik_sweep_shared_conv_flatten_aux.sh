#!/bin/bash

#SBATCH -J pi_multik_sweep_shared_conv_flatten_aux
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --output=logs/pi_multik_sweep_shared_conv_flatten_aux_%A_%a.out
#SBATCH --error=logs/pi_multik_sweep_shared_conv_flatten_aux_%A_%a.err

# pi_multik encoder/fusion sweep, WITH auxiliary variables -- combo 3/6:
# encoder_mode=shared, fusion_mode=conv, fusion_pool=flatten,
# include_entropy=true. See pi_multik_sweep_shared_concat_aux.sh's header
# for what include_entropy adds. Same architecture as
# encsweep_shared_conv_flatten (the worst/most unstable combo in the
# original sweep -- see the loss-comparison report) -- this rerun also
# serves as a second look at whether that instability reproduces.
# nested_thomas, dtm k=5,10,15 (configs/runs/nested_thomas/pi_multik.yaml).
#
# 5 seeds via --array=0-4 indexing SEEDS below (same 5 seeds as the
# original sweep). Submit from the point-process-tda repo root:
#   sbatch slurm/pi_multik_sweep_shared_conv_flatten_aux.sh

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
  --run-tag encsweep_shared_conv_flatten_aux \
  --set method.params.encoder_mode=shared \
  --set method.params.fusion_mode=conv \
  --set method.params.fusion_pool=flatten \
  --set method.params.include_entropy=true
