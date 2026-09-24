#!/bin/bash

#SBATCH -J cascade_comp
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH --array=0-4
#SBATCH --output=cascade/logs/cascade_comp_%A_%a.out
#SBATCH --error=cascade/logs/cascade_comp_%A_%a.err

# Stages 2 and 3 with gradient-boosted trees, one array task per regime.taus entry (5 by default:
# keep --array in step with the config). Every component, trained on its own in-regime clouds.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p cascade/logs
module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"   # the module sets 1, which serializes HGB
CONFIG="${CONFIG:-cascade/configs/default.yaml}"

python cascade/components.py --config "$CONFIG" --model hgb --tau-index "$SLURM_ARRAY_TASK_ID"
