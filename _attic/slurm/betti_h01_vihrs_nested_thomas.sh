#!/bin/bash

#SBATCH -J betti_h01_vihrs_nested_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-49
#SBATCH --output=logs/betti_h01_vihrs_nested_thomas_%A_%a.out
#SBATCH --error=logs/betti_h01_vihrs_nested_thomas_%A_%a.err

# The vihrs baseline plus the betti_multik "H0+H1 stacked in one model"
# variant (homology_dims: [0, 1], the persistence-image-style channel
# stack -- NOT betti_cnn's one-model-per-dim sweep, see
# betti_multik_rips.yaml's header) across all four filtrations: rips,
# dtm k=5, dtm k=10, dtm k=15. 5 configs x 10 seeds = 50 tasks, flattened
# into one array (SLURM_ARRAY_TASK_ID = config_index * 10 + seed_index),
# same pattern as betti_cnn_sweep_nested_thomas.sh/
# betti_multik_dims_nested_thomas.sh. Already-done (config, seed) pairs
# is_done()-skip near-instantly, so re-submitting after a partial run
# (e.g. vihrs_checkpointed already has seeds 9371-9375 done) is
# safe/cheap.
#
# ASSUMES data/nested_thomas/clouds.pkl and data/nested_thomas/{rips,
# dtm_k5,dtm_k10,dtm_k15}/diagrams.pkl already exist (they do -- same grid
# as betti_cnn_sweep_nested_thomas.sh/betti_multik.yaml). vihrs reads
# clouds.pkl directly, no diagrams needed.
#
# UPDATE: the betti_multik configs referenced below (task ids 10-49,
# config_index 1-4) were cancelled (scancel <jobid>_[10-49]) before any
# ran, then switched to include_euler: false and moved to their own
# script, slurm/betti_multik_singlescale_nochi_nested_thomas.sh -- submit
# that instead for the betti_multik side of this sweep. Only the vihrs
# tasks (config_index 0, task ids 0-9) were left running/queued under this
# job; this script is kept as-is (still valid to resubmit for vihrs alone,
# the betti_multik tasks would just redo the now-no-euler configs --
# redundant with the dedicated script above, not wrong).
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/betti_h01_vihrs_nested_thomas.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

CONFIGS=(
  configs/runs/nested_thomas/nested_thomas_vihrs.yaml
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
