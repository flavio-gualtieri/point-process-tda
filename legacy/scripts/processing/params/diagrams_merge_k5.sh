#!/bin/bash
#SBATCH -J diagrams_merge_k5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=24G
#SBATCH -t 00:59:00
#SBATCH --output=logs/diagrams_merge_k5_%j.out
#SBATCH --error=logs/diagrams_merge_k5_%j.err

# STAGE 3 of 3 -- k=5 variant. Run ONCE, after every diagrams_compute_k5
# array task has finished. Concatenates the 60 per-group partials into:
#   data/params/2d/thomas/diagrams.pkl
#   data/params/2d/thomas/adversarial_diagrams.pkl
# --n-groups 60 MUST match the compute value.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

python scripts/processing/params/diagrams_slurm.py \
  --config configs/params/processing/new_features.yaml \
  --process thomas \
  --dimension 2 \
  --n-groups 60 \
  --splits train_test adversarial \
  --mode merge \
  --overwrite
