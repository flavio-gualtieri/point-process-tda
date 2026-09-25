#!/bin/bash

#SBATCH -J oneshot_prepare
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --output=oneshot/logs/oneshot_prepare_%j.out
#SBATCH --error=oneshot/logs/oneshot_prepare_%j.err

# PH summary inputs for every family (resumable), then a check that every input the config uses exists.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p oneshot/logs
module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"   # the module sets 1, which serializes HGB
CONFIG="${CONFIG:-oneshot/configs/default.yaml}"
python -u oneshot/prepare.py --config "$CONFIG" ph --workers "$SLURM_CPUS_PER_TASK"
python -u oneshot/prepare.py --config "$CONFIG" check
