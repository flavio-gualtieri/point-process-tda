#!/bin/bash

#SBATCH -J nested_thomas_generate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH -t 00:15:00
#SBATCH --output=logs/nested_thomas_generate_%j.out
#SBATCH --error=logs/nested_thomas_generate_%j.err

# Submit from the point-process-tda repo root:
#   sbatch scripts/processing/params/nested_thomas_generate.sh
#
# STAGE 1 of 3 for the nested_thomas dataset. Cloud generation itself is
# cheap (~1,000 clouds, well under a second of actual sampling work) --
# this walltime/memory budget is generous headroom, not a real estimate.
# Writes data/params/2d/nested_thomas/{clouds.pkl, adversarial_clouds.pkl,
# cloud_generation_manifest.yaml}. Resumable: skip_missing/overwrite in the
# config control re-run behavior; see configs/params/processing/
# nested_thomas_generate.yaml.
#
# Next: scripts/processing/params/nested_thomas_diagrams_compute.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unset}"

python scripts/processing/params/pipeline.py \
  configs/params/processing/nested_thomas_generate.yaml
