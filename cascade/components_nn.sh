#!/bin/bash

#SBATCH -J cascade_comp_nn
#SBATCH -A pilot_sae_gpu
#SBATCH -p sae
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --array=0-4
#SBATCH --output=cascade/logs/cascade_comp_nn_%A_%a.out
#SBATCH --error=cascade/logs/cascade_comp_nn_%A_%a.err

# Stages 2 and 3 with cloudforger's network (config nn:), one GPU array task per regime.taus entry.
# Partition and account as slurm/train.sh.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p cascade/logs
module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"   # the module sets 1, which serializes HGB
CONFIG="${CONFIG:-cascade/configs/default.yaml}"

python cascade/components.py --config "$CONFIG" --model nn --tau-index "$SLURM_ARRAY_TASK_ID"
