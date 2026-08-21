#!/bin/bash

#SBATCH -J vec_multik_landscape_calib_koverride_nested_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-11
#SBATCH --output=logs/vec_multik_landscape_calib_koverride_nested_thomas_%A_%a.out
#SBATCH --error=logs/vec_multik_landscape_calib_koverride_nested_thomas_%A_%a.err

# nested_thomas sibling of vec_multik_landscape_calib_koverride_thomas.sh --
# see that script for the full rationale. Landscape K sweep, FUSED DTM
# k=5,10,15 (H0+H1, native CNN encoder): fixed K override vs. the coverage
# rule's data-derived K, 4 arms:
#   k4          -- vec_multik_landscape_native.yaml (G=128), --set K=4
#   k8_matched  -- vec_multik_landscape_matched.yaml (G=512, K=8 already
#                  fixed in that config); no --set needed, run as-is
#   k16         -- vec_multik_landscape_native.yaml (G=128), --set K=16
#   datadriven  -- vec_multik_landscape_native.yaml (G=128), K left unset
#                  -> choose_K's own answer
# k8_matched is the only arm at G=512 (budget-matched vs the 64x64 PI
# baseline); k4/k16/datadriven hold G=128 fixed.
# 3 seeds (9371, 9372, 9373).
#
# Each arm gets its own --run-tag, so results land under
# results/nested_thomas/dtm_k5+10+15/vec_multik_landscape_native/_runs/calib_koverride_<arm>/seed_<seed>/
# -- separate from the plain (untagged) baseline results, from the
# standalone "matched" run, and from every other arm here, never
# overwriting.
#
# 4 arms x 3 seeds = 12 tasks, flattened into one array
# (SLURM_ARRAY_TASK_ID = arm_index * 3 + seed_index).
#
# ASSUMES data/nested_thomas/clouds.pkl and data/nested_thomas/dtm_k{5,10,15}/diagrams.pkl
# already exist (they do).
# Submit from the point-process-tda repo root:
#   sbatch slurm/vec_multik_landscape_calib_koverride_nested_thomas.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

CONFIGS=(
  configs/runs/nested_thomas/vec_multik_landscape_native.yaml
  configs/runs/nested_thomas/vec_multik_landscape_matched.yaml
  configs/runs/nested_thomas/vec_multik_landscape_native.yaml
  configs/runs/nested_thomas/vec_multik_landscape_native.yaml
)
K_OVERRIDES=(4 "" 16 "")   # empty -> no --set: matched.yaml already fixes K=8, datadriven leaves K unset
TAGS=(calib_koverride_k4 calib_koverride_k8_matched calib_koverride_k16 calib_koverride_datadriven)
SEEDS=(9371 9372 9373)
N_SEEDS=${#SEEDS[@]}

ARM_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))
CONFIG="${CONFIGS[$ARM_IDX]}"
K_OVERRIDE="${K_OVERRIDES[$ARM_IDX]}"
TAG="${TAGS[$ARM_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

EXTRA_ARGS=()
if [[ -n "$K_OVERRIDE" ]]; then
  EXTRA_ARGS+=(--set "method.params.K=$K_OVERRIDE")
fi

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unset}  Array task: ${SLURM_ARRAY_TASK_ID:-unset}  Config: $CONFIG  K override: ${K_OVERRIDE:-<data-derived>}  Tag: $TAG  Seed: $SEED"
echo "Assigned GPU: ${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python -u scripts/train.py "$CONFIG" \
  --seed "$SEED" \
  "${EXTRA_ARGS[@]}" \
  --run-tag "$TAG"
