#!/bin/bash

#SBATCH -J pilot_analyze
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH --output=pilot/logs/pilot_analyze_%j.out
#SBATCH --error=pilot/logs/pilot_analyze_%j.err

# All the pilot's models (~150 gradient-boosted tree fits on <= 60k rows) and the bootstrap.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p pilot/logs
module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"   # the module sets 1, which serializes HGB
python -u pilot/analyze.py
