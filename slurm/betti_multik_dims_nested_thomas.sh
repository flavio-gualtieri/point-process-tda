#!/bin/bash

#SBATCH -J betti_multik_dims_nested_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-29
#SBATCH --output=logs/betti_multik_dims_nested_thomas_%A_%a.out
#SBATCH --error=logs/betti_multik_dims_nested_thomas_%A_%a.err

# Re-trains betti_multik (experiments/pi_multik/betti_multik.py) on DTM
# k=5,10,15, WITHOUT the Euler-characteristic channel, in 3
# homology-dimension variants (h0/h1/h01) -- same scheme as
# betti_multik_dims_thomas.sh, see that file's header for the full
# rationale (results-dir overwrite semantics, --set mechanism, array
# indexing). Only the base config differs:
# configs/runs/nested_thomas/betti_multik.yaml.
#
# h01 (homology_dims=[0, 1]) lands in the SAME results dir the earlier
# euler-INCLUDED run already populated
# (results/nested_thomas/dtm_k5+10+15/betti_multik/) -- --force below
# makes this OVERWRITE those results in place, per instruction.
#
# ASSUMES data/nested_thomas/clouds.pkl and
# data/nested_thomas/dtm_k{5,10,15}/diagrams.pkl already exist (they do --
# same grid the existing betti_multik results were trained on).
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/betti_multik_dims_nested_thomas.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

CONFIG=configs/runs/nested_thomas/betti_multik.yaml
DIMS=("[0]" "[1]" "[0, 1]")
LABELS=(h0 h1 h01)
SEEDS=(9371 9372 9373 9374 9375 9376 9377 9378 9379 9380)
N_SEEDS=${#SEEDS[@]}

VARIANT_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))
DIMS_VALUE="${DIMS[$VARIANT_IDX]}"
LABEL="${LABELS[$VARIANT_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unset}  Array task: ${SLURM_ARRAY_TASK_ID:-unset}  Variant: $LABEL (homology_dims=$DIMS_VALUE)  Seed: $SEED"
echo "Assigned GPU: ${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python -u scripts/train.py "$CONFIG" \
  --seed "$SEED" \
  --set "method.params.homology_dims=$DIMS_VALUE" \
  --set method.params.include_euler=false \
  --force
