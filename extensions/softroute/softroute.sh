#!/bin/bash

#SBATCH -J ext_softroute
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH --output=extensions/logs/softroute_%j.out
#SBATCH --error=extensions/logs/softroute_%j.err

# Soft-routing score (softroute.py). One process per cloud; each simulates 10 models x 16 patterns.
#
#   sbatch extensions/softroute/softroute.sh
#   LIMIT=16 sbatch extensions/softroute/softroute.sh      # smoke test

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p extensions/logs
module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1     # no bytecode written beside cascade/scoring
python -u extensions/softroute/softroute.py ${CONFIG:+--config "$CONFIG"} ${LIMIT:+--limit "$LIMIT"}
