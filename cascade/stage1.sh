#!/bin/bash

#SBATCH -J cascade_stage1
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --output=cascade/logs/cascade_stage1_%j.out
#SBATCH --error=cascade/logs/cascade_stage1_%j.err

# Features (skipped when on disk) and stage 1 with its out-of-fold train predictions, CPU.
#   CONFIG=cascade/configs/logreg.yaml sbatch cascade/stage1.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p cascade/logs
module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"   # the module sets 1, which serializes HGB
CONFIG="${CONFIG:-cascade/configs/default.yaml}"

python cascade/features.py --workers "${SLURM_CPUS_PER_TASK:-8}"
python cascade/stage1.py train --config "$CONFIG"
