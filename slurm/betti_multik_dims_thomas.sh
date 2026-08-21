#!/bin/bash

#SBATCH -J betti_multik_dims_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-29
#SBATCH --output=logs/betti_multik_dims_thomas_%A_%a.out
#SBATCH --error=logs/betti_multik_dims_thomas_%A_%a.err

# Re-trains betti_multik (experiments/pi_multik/betti_multik.py) on DTM
# k=5,10,15, WITHOUT the Euler-characteristic channel
# (method.params.include_euler: false, added specifically for this run --
# every earlier betti_multik config predates that flag and always included
# it), in 3 homology-dimension variants:
#   h0  -- homology_dims: [0]     -> results/.../betti_multik_h0/
#   h1  -- homology_dims: [1]     -> results/.../betti_multik_h1/
#   h01 -- homology_dims: [0, 1]  -> results/.../betti_multik/          (*)
# (*) h01 lands in the SAME results dir the euler-INCLUDED run already
# populated (results/{thomas,nested_thomas}/dtm_k5+10+15/betti_multik/) --
# that's deliberate, not a collision: --force below makes this run
# OVERWRITE those existing results in place, per instruction. h0/h1 are new
# variants with no prior results to collide with.
#
# Both process/design and every other hyperparameter are read from the
# existing configs/runs/thomas/thomas_betti_multik_k5k10k15.yaml --
# homology_dims/include_euler are overridden per task via --set rather than
# forking 3 new config files, since they're plain method.params leaves
# (config.py's apply_overrides parses the --set value with yaml.safe_load,
# so --set method.params.homology_dims=[0] parses to a real Python list).
#
# 3 variants x 10 seeds = 30 tasks, flattened into one array
# (SLURM_ARRAY_TASK_ID = variant_index * 10 + seed_index), same
# array-per-task pattern as betti_multik_thomas.sh/betti_cnn_sweep_thomas.sh.
#
# ASSUMES data/thomas/clouds.pkl and data/thomas/dtm_k{5,10,15}/diagrams.pkl
# already exist (they do -- this is the same grid the existing betti_multik
# results were trained on).
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/betti_multik_dims_thomas.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

CONFIG=configs/runs/thomas/thomas_betti_multik_k5k10k15.yaml
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
