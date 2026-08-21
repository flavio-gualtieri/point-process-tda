#!/bin/bash

#SBATCH -J betti_multik_singlescale_nochi_nested_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-39
#SBATCH --output=logs/betti_multik_singlescale_nochi_nested_thomas_%A_%a.out
#SBATCH --error=logs/betti_multik_singlescale_nochi_nested_thomas_%A_%a.err

# Single-scale betti_multik (H0+H1 stacked, NO Euler-characteristic
# channel) across all four filtrations: rips, dtm k=5, dtm k=10,
# dtm k=15. Each config now sets method.params.include_euler: false (was
# implicit true before) -- see betti_multik_rips.yaml/betti_multik_k5.yaml/
# betti_multik_k10.yaml/betti_multik_k15.yaml's headers.
#
# REPLACES the single-scale portion (config_index 1-4, task ids 10-49) of
# slurm/betti_h01_vihrs_nested_thomas.sh's array, which was cancelled
# before any of those tasks ran (all 50 tasks were still pending -- see
# that script's updated header). No results existed yet for any of these
# four configs, so there was nothing to delete here; this is a
# from-scratch submission, not a re-run.
#
# 4 configs x 10 seeds = 40 tasks, flattened into one array
# (SLURM_ARRAY_TASK_ID = config_index * 10 + seed_index), same pattern as
# betti_cnn_sweep_nested_thomas.sh/betti_h01_vihrs_nested_thomas.sh.
# Already-done (config, seed) pairs is_done()-skip near-instantly, so
# re-submitting after a partial run is safe/cheap.
#
# ASSUMES data/nested_thomas/clouds.pkl and data/nested_thomas/{rips,
# dtm_k5,dtm_k10,dtm_k15}/diagrams.pkl already exist (they do -- same
# grid as betti_cnn_sweep_nested_thomas.sh).
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/betti_multik_singlescale_nochi_nested_thomas.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

CONFIGS=(
  configs/runs/nested_thomas/betti_multik_rips.yaml
  configs/runs/nested_thomas/betti_multik_k5.yaml
  configs/runs/nested_thomas/betti_multik_k10.yaml
  configs/runs/nested_thomas/betti_multik_k15.yaml
)
SEEDS=(9371 9372 9373 9374 9375 9376 9377 9378 9379 9380)
N_SEEDS=${#SEEDS[@]}

CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))
CONFIG="${CONFIGS[$CONFIG_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unset}  Array task: ${SLURM_ARRAY_TASK_ID:-unset}  Config: $CONFIG  Seed: $SEED"
echo "Assigned GPU: ${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python -u scripts/train.py "$CONFIG" --seed "$SEED"
