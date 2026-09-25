#!/bin/bash

#SBATCH -J oneshot_gpu
#SBATCH -A pilot_sae_gpu
#SBATCH -p sae
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH -t 24:00:00
#SBATCH --output=oneshot/logs/oneshot_gpu_%A_%a.out
#SBATCH --error=oneshot/logs/oneshot_gpu_%A_%a.err

# One network unit per array task (cloudforger's PHNet). The input is built over all 8 families'
# 800k patterns (cascade's 5-family runs used 64G), hence the memory.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p oneshot/logs
module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"   # the module sets 1, which serializes HGB
CONFIG="${CONFIG:-oneshot/configs/default.yaml}"
python -u oneshot/train.py --config "$CONFIG" unit --kind gpu --index "$SLURM_ARRAY_TASK_ID"
