#!/bin/bash

#SBATCH -J scoring
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH -t 06:00:00
#SBATCH --output=cascade/scoring/logs/power_%j.out
#SBATCH --error=cascade/scoring/logs/power_%j.err

# Power check for the candidate scores (cascade/scoring/power.py), one cloud per worker process.
#
#   sbatch cascade/scoring/power.sh
#   CONFIG=cascade/scoring/<other>.yaml sbatch cascade/scoring/power.sh
#   CONFIG=cascade/scoring/smoke.yaml LIMIT=10 sbatch -p computeshort -t 00:30:00 cascade/scoring/power.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p cascade/scoring/logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u
export OMP_NUM_THREADS=1        # parallel over clouds, not within numpy

python cascade/scoring/power.py --config "${CONFIG:-cascade/scoring/config.yaml}" --workers "${SLURM_CPUS_PER_TASK:-16}" ${LIMIT:+--limit $LIMIT}
