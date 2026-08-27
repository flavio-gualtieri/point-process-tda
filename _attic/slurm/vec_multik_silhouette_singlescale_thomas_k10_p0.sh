#!/bin/bash

#SBATCH -J vec_multik_silhouette_singlescale_thomas_k10_p0
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-9
#SBATCH --output=logs/vec_multik_silhouette_singlescale_thomas_k10_p0_%A_%a.out
#SBATCH --error=logs/vec_multik_silhouette_singlescale_thomas_k10_p0_%A_%a.err

# p-sweep sibling of vec_multik_silhouette_singlescale_thomas_k10.sh (p=1,
# already run): same vectorization=silhouette, encoder_path=native, dtm
# k=10-only config (configs/runs/thomas/vec_multik_silhouette_k10.yaml),
# p overridden to 0 via --set method.params.p=0 -- see
# vec_multik_silhouette_thomas_p0.sh's header for why p is swept this way
# instead of a separate YAML per value. Results land under
# results/thomas/dtm_k10/vec_multik_silhouette_native_p0/seed_<seed>/.
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/vec_multik_silhouette_singlescale_thomas_k10_p0.sh

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

python -u scripts/train.py configs/runs/thomas/vec_multik_silhouette_k10.yaml \
  --set method.params.p=0 \
  --seed "$SEED"
