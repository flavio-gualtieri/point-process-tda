#!/bin/bash

#SBATCH -J pilot_gen
#SBATCH -p computeshort
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t 01:00:00
#SBATCH --array=0-59
#SBATCH --output=pilot/logs/pilot_gen_%A_%a.out
#SBATCH --error=pilot/logs/pilot_gen_%A_%a.err

# One array task per (new family, shard): 3 families x (thetas 5000 / shard_size 250) = 60 tasks.
# Ring dominates: ~0.3 s per theta for the cv inversion, so a shard is ~2 min. Resumable.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p pilot/logs
module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

FAMILIES=(ring matern1 cell)
SHARDS=20
FAMILY=${FAMILIES[$((SLURM_ARRAY_TASK_ID / SHARDS))]}
python -u pilot/generate.py run --family "$FAMILY" --shard $((SLURM_ARRAY_TASK_ID % SHARDS))
