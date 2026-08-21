#!/bin/bash

#SBATCH -J betti_cnn_sweep_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-79
#SBATCH --output=logs/betti_cnn_sweep_thomas_%A_%a.out
#SBATCH --error=logs/betti_cnn_sweep_thomas_%A_%a.err

# Trains betti_cnn (experiments/betti_cnn.py) across the full filtration x
# homology-dimension grid -- rips, dtm k=5, dtm k=10, dtm k=15, each at H0
# and H1 (8 configs, configs/runs/thomas/betti_cnn_{rips,k5,k10,k15}_h{0,1}.yaml)
# -- all 10 configured seeds. One (config, seed) pair per array task: 8
# configs x 10 seeds = 80 tasks, flattened into a single array
# (SLURM_ARRAY_TASK_ID = config_index * 10 + seed_index) rather than 8
# separate sbatch submissions. Same array-per-task pattern
# pi_multik_baseline_complete.sh/betti_multik_thomas.sh already established
# (a serial multi-seed loop in one job doesn't finish inside the 59-min
# limit; already-done (config, seed) pairs is_done()-skip near-instantly,
# so re-submitting after a partial run is safe/cheap).
#
# ASSUMES data/thomas/clouds.pkl and data/thomas/{rips,dtm_k5,dtm_k10,
# dtm_k15}/diagrams.pkl already exist. dtm_k{5,10,15} almost certainly do
# (same grid as thomas_pi_multik_k5k10k15.yaml/thomas_betti_multik_k5k10k15.yaml);
# rips almost certainly does NOT (no existing thomas config uses it -- PI-
# based methods can't, see betti_cnn_rips_h0.yaml's header). If any are
# missing, run once, serially, BEFORE submitting this array job (concurrent
# array tasks writing the same diagrams.pkl would race):
#   python scripts/generate.py  configs/runs/thomas/betti_cnn_rips_h0.yaml
#   python scripts/featurize.py configs/runs/thomas/betti_cnn_rips_h0.yaml
#   python scripts/featurize.py configs/runs/thomas/betti_cnn_k5_h0.yaml
#   python scripts/featurize.py configs/runs/thomas/betti_cnn_k10_h0.yaml
#   python scripts/featurize.py configs/runs/thomas/betti_cnn_k15_h0.yaml
# (generate.py only needs running once total -- clouds.pkl doesn't depend
# on filtration; featurize.py once per filtration -- h0/h1 configs share
# the same diagrams.pkl, so featurizing the _h0 config covers _h1 too.)
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/betti_cnn_sweep_thomas.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

CONFIGS=(
  configs/runs/thomas/betti_cnn_rips_h0.yaml
  configs/runs/thomas/betti_cnn_rips_h1.yaml
  configs/runs/thomas/betti_cnn_k5_h0.yaml
  configs/runs/thomas/betti_cnn_k5_h1.yaml
  configs/runs/thomas/betti_cnn_k10_h0.yaml
  configs/runs/thomas/betti_cnn_k10_h1.yaml
  configs/runs/thomas/betti_cnn_k15_h0.yaml
  configs/runs/thomas/betti_cnn_k15_h1.yaml
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
