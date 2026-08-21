#!/bin/bash

#SBATCH -J vec_multik_silhouette_nested_thomas_p3
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-9
#SBATCH --output=logs/vec_multik_silhouette_nested_thomas_p3_%A_%a.out
#SBATCH --error=logs/vec_multik_silhouette_nested_thomas_p3_%A_%a.err

# p-sweep sibling of vec_multik_silhouette_nested_thomas.sh (p=1, already run):
# same vectorization=silhouette, encoder_path=native config
# (configs/runs/nested_thomas/vec_multik_silhouette_native.yaml), p overridden to
# 3 via --set method.params.p=3 -- p is an EXPERIMENTAL AXIS (run every
# value in {0,1,2,3} and report all, never pick a 'winner', see
# vectorization/landscapes/tent.py's silhouette_from_tents). Results land
# under vec_multik_silhouette_native_p3/, separate from p=1's
# vec_multik_silhouette_native_p1/ -- no --run-tag needed, subdir already
# encodes p (VectorizedMultiKExperiment.subdir).
#
# ASSUMES data/nested_thomas/clouds.pkl and data/nested_thomas/dtm_k{5,10,15}/diagrams.pkl
# already exist (they do -- vec_multik_silhouette_native.yaml's p=1 run
# already used them).
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/vec_multik_silhouette_nested_thomas_p3.sh

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

python -u scripts/train.py configs/runs/nested_thomas/vec_multik_silhouette_native.yaml \
  --set method.params.p=3 \
  --seed "$SEED"
