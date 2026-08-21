#!/bin/bash

#SBATCH -J error_floor_generate_nested_thomas
#SBATCH -t 00:29:00
#SBATCH --mem-per-cpu=11G
#SBATCH --output=logs/error_floor_generate_nested_thomas_%j.out
#SBATCH --error=logs/error_floor_generate_nested_thomas_%j.err

# Generates the error-floor dataset, nested_thomas counterpart of
# error_floor_generate_thomas.sh -- see that file's header.
# configs/runs/nested_thomas/error_floor.yaml -> writes
# data/error_floor/nested_thomas/clouds.pkl.
#
# NOTE: unlike thomas, there is currently no estimate_error_floor.py
# consumer for this dataset (nested_thomas has no mincontrast baseline --
# see mincontrast_g.yaml's header). This just generates and holds the data
# for whenever an estimator exists.
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/error_floor_generate_nested_thomas.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unset}"

python -u scripts/generate.py configs/runs/nested_thomas/error_floor.yaml
