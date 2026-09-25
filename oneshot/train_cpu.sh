#!/bin/bash

#SBATCH -J oneshot_cpu
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --output=oneshot/logs/oneshot_cpu_%A_%a.out
#SBATCH --error=oneshot/logs/oneshot_cpu_%A_%a.err

# One CPU unit (table learner) per array task; submit.sh sets --array to the config's CPU units.
# The longest is a classifier on ~600k rows x 8 classes (plus 5 out-of-fold refits for the
# regime's reference classifier); estimators train on one family's ~74k rows.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p oneshot/logs
module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"   # the module sets 1, which serializes HGB
CONFIG="${CONFIG:-oneshot/configs/default.yaml}"
python -u oneshot/train.py --config "$CONFIG" unit --kind cpu --index "$SLURM_ARRAY_TASK_ID"
