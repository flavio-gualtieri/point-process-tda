#!/bin/bash

#SBATCH -J pilot_feat
#SBATCH -p computeshort
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --array=0-2
#SBATCH --output=pilot/logs/pilot_feat_%A_%a.out
#SBATCH --error=pilot/logs/pilot_feat_%A_%a.err

# Classical features + alpha and DTM(k=10) diagrams for one new family (10000 patterns). DTM is the
# cost: ~60 ms per pattern, so ~1 min on 16 workers; memory is ripser's O(n^2) scratch per worker.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p pilot/logs
module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

FAMILIES=(ring matern1 cell)
python -u pilot/featurize.py new --family "${FAMILIES[$SLURM_ARRAY_TASK_ID]}" --workers "$SLURM_CPUS_PER_TASK"
