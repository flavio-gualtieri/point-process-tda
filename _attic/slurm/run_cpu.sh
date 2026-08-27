#!/bin/bash

#SBATCH -J pi_multik_cpu
#SBATCH -t 03:00:00
#SBATCH --mem-per-cpu=11G
#SBATCH --output=logs/pi_multik_cpu_%j.out
#SBATCH --error=logs/pi_multik_cpu_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unset}"

python -u scripts/train.py configs/runs/nested_thomas/pi_multik.yaml --run-tag no_pi_zscore
