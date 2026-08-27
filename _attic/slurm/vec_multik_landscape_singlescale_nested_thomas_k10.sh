#!/bin/bash

#SBATCH -J vec_multik_landscape_singlescale_nested_thomas_k10
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-9
#SBATCH --output=logs/vec_multik_landscape_singlescale_nested_thomas_k10_%A_%a.out
#SBATCH --error=logs/vec_multik_landscape_singlescale_nested_thomas_k10_%A_%a.err

# Single-scale sibling of vec_multik_landscape_nested_thomas.sh: same vectorization=
# landscape, encoder_path=native config, but dtm k=10 ONLY
# (configs/runs/nested_thomas/vec_multik_landscape_k10.yaml -- filtration: has just
# this one entry, so k_values auto-derives to [10], n_k=1). All 10
# configured seeds via --array=0-9. No --run-tag needed -- results land
# under results/nested_thomas/dtm_k10/vec_multik_landscape_native/seed_<seed>/,
# separate from the k5,10,15 fused run's dtm_k5+10+15/vec_multik_landscape_native/.
#
# ASSUMES data/nested_thomas/clouds.pkl and data/nested_thomas/dtm_k10/diagrams.pkl
# already exist -- see vec_multik_landscape_nested_thomas.sh's header for the
# generate.py/featurize.py commands if not.
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/vec_multik_landscape_singlescale_nested_thomas_k10.sh

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

python -u scripts/train.py configs/runs/nested_thomas/vec_multik_landscape_k10.yaml \
  --seed "$SEED"
