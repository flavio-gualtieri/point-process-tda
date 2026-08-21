#!/bin/bash

#SBATCH -J error_floor_generate_thomas
#SBATCH -t 00:29:00
#SBATCH --mem-per-cpu=11G
#SBATCH --output=logs/error_floor_generate_thomas_%j.out
#SBATCH --error=logs/error_floor_generate_thomas_%j.err

# Generates the error-floor dataset: 50 parameter vectors x 200
# realizations, thomas (configs/runs/thomas/error_floor.yaml -- see that
# file's header). CPU-only, single task -- point-cloud sampling, no
# GPU/torch use at all. Writes data/error_floor/thomas/clouds.pkl (+
# adversarial_clouds.pkl, empty since adversarial is disabled, +
# cloud_generation_manifest.yaml), separate from the main
# data/thomas/clouds.pkl.
#
# Run this BEFORE slurm/error_floor_estimate_thomas.sh (which reads its
# output).
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/error_floor_generate_thomas.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unset}"

python -u scripts/generate.py configs/runs/thomas/error_floor.yaml
