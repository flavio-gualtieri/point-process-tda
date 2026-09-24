#!/bin/bash

#SBATCH -J pilot_assemble
#SBATCH -p computeshort
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --output=pilot/logs/pilot_assemble_%j.out
#SBATCH --error=pilot/logs/pilot_assemble_%j.err

# Every family's pilot rows -> data/pilot/<name>/features/{classical, ph_<tag>}/. Reads the bank's
# merged diagrams one file at a time (up to 1.1 G each) and keeps copies of the pilot rows only.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p pilot/logs
module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u
python -u pilot/featurize.py assemble
