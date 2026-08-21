#!/bin/bash

#SBATCH -J vihrs_nested_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --output=logs/vihrs_nested_thomas_%A_%a.out
#SBATCH --error=logs/vihrs_nested_thomas_%A_%a.err

# vihrs baseline (L(r)-r + n(x) 1D-CNN, Vihrs 2022) on nested_thomas --
# the comparison baseline for the pi_multik encoder/fusion sweep
# (pi_multik_sweep_*.sh). No filtration/features needed: vihrs reads
# clouds.pkl directly. checkpoint_best: true in the config (reload
# best-val-loss state before final eval) writes to
# results/nested_thomas/<no-filtration-tag>/vihrs_checkpointed/ -- see
# configs/runs/nested_thomas/nested_thomas_vihrs.yaml's header for the
# paper-faithful fixed-epoch/no-checkpointing variant ("vihrs" subdir
# instead), not submitted here.
#
# 5 seeds via --array=0-4 indexing SEEDS below (first 5 of that config's
# 10-seed list, matching the pi_multik sweep's seeds). Submit from the
# point-process-tda repo root:
#   sbatch slurm/vihrs_nested_thomas.sh

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

python -u scripts/train.py configs/runs/nested_thomas/nested_thomas_vihrs.yaml \
  --seed "$SEED"
