#!/bin/bash

#SBATCH -J oneshot_compare
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH --output=oneshot/logs/oneshot_compare_%j.out
#SBATCH --error=oneshot/logs/oneshot_compare_%j.err

# Score everything trained so far (compare.py); safe to rerun at any point.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p oneshot/logs
module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"   # the module sets 1, which serializes HGB
CONFIG="${CONFIG:-oneshot/configs/default.yaml}"
python -u oneshot/compare.py --config "$CONFIG"
