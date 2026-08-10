#!/bin/bash
#SBATCH -J diagrams_split
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=24G
#SBATCH -t 00:30:00
#SBATCH --output=logs/diagrams_split_%j.out
#SBATCH --error=logs/diagrams_split_%j.err

# STAGE 1 of 3 -- run ONCE, on a single node.
# Loads each full clouds file once and writes 40 contiguous chunk files under
# data/params/2d/thomas/_chunks/. After this, each compute task reads only its
# own 1/40 slice instead of the whole 420M file.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

python scripts/processing/params/diagrams_slurm.py \
  --config configs/params/processing/new_features_k10.yaml \
  --process thomas \
  --dimension 2 \
  --n-groups 60 \
  --splits train_test adversarial \
  --mode split