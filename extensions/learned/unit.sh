#!/bin/bash

#SBATCH -J ext_learned
#SBATCH -A pilot_sae_gpu
#SBATCH -p sae
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --output=extensions/logs/learned_%A_%a.out
#SBATCH --error=extensions/logs/learned_%A_%a.err

# One learned-summary unit per array task (learned.py unit); submit.sh sets --array.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p extensions/logs
module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u
export OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
python -u extensions/learned/learned.py --config "${CONFIG:-extensions/learned/smoke.yaml}" unit --index "$SLURM_ARRAY_TASK_ID"
