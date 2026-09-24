#!/bin/bash

#SBATCH -J cascade_assemble
#SBATCH -p computeshort
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH --output=cascade/logs/cascade_assemble_%j.out
#SBATCH --error=cascade/logs/cascade_assemble_%j.err

# Assemble the components for every tau and model pair, and draw the sweep summary (pipeline.py).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p cascade/logs
module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"   # the module sets 1, which serializes HGB
CONFIG="${CONFIG:-cascade/configs/default.yaml}"

python cascade/pipeline.py --config "$CONFIG"
