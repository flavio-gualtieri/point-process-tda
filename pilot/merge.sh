#!/bin/bash

#SBATCH -J pilot_merge
#SBATCH -p computeshort
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH --output=pilot/logs/pilot_merge_%j.out
#SBATCH --error=pilot/logs/pilot_merge_%j.err

# Stitch the shards into data/pilot/<name>/bank/<family>/{points.npz, manifest.csv}, checking completeness.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p pilot/logs
module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u
python -u pilot/generate.py merge
