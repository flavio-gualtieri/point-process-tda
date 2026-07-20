#!/bin/bash

#SBATCH -J dtm_train_k5
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 12:00:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-9
#SBATCH --output=logs/dtm_train_k5_%A_%a.out
#SBATCH --error=logs/dtm_train_k5_%A_%a.err

# Submit from the point-process-tda repo root: sbatch dtm_experiment/run_train_gpu_k5.sh
# One GPU array task per seed (10 seeds x {vihrs, fusion_pi, fusion_betti,
# fusion_scalars, fusion, ph_combined, pi_01, betti_cnn_01}, that order, per
# dtm_experiment/train_k5.py::run_seed) -- array index maps to SEEDS[index]
# below, so all 10 seeds train in parallel instead of one process looping
# over them sequentially. Resumable per task: train_k5.py skips any
# seed/method whose results.pt already exists, so re-submitting after
# hitting the walltime (or after a single array task failure) just picks up
# where that task left off.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

# Must match dtm_experiment/train_k5.py::SEEDS exactly.
SEEDS=(9371 9372 9373 9374 9375 9376 9377 9378 9379 9380)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unset}  Array task: ${SLURM_ARRAY_TASK_ID:-unset}  Seed: $SEED"
echo "Assigned GPU: ${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python dtm_experiment/train_k5_nested.py --seed "$SEED"
