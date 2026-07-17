#!/bin/bash

#SBATCH -J compute_features_nested_thomas
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH -t 00:30:00
#SBATCH --output=logs/compute_features_nested_thomas_%j.out
#SBATCH --error=logs/compute_features_nested_thomas_%j.err

# Submit from the point-process-tda repo root, AFTER
# scripts/processing/params/nested_thomas_diagrams_compute.sh (stage 2a) and
# nested_thomas_diagrams_merge.sh (stage 2b) have finished:
#   sbatch dtm_experiment/run_compute_features_nested_thomas.sh
#
# STAGE 3 of 3 for the nested_thomas dataset. Same script as
# run_compute_features.sh, just pointed at data/params/2d/nested_thomas/
# via --process (see dtm_experiment/compute_features.py's build_arg_parser).
# Resources are scaled down from run_compute_features.sh's 4 cpus/16G-per-
# cpu/4h (~40,000 clouds) for this dataset's ~1,000 clouds -- not
# separately timed, so bump -t if this doesn't finish in time.
#
# Reads data/params/2d/nested_thomas/{diagrams_dtm_k5.pkl,
# adversarial_diagrams_dtm_k5.pkl}. Writes {betti,betti_weighted,images}_
# dtm_k5.pkl (+ adversarial_*) alongside them. Single process, no CLI args
# beyond --process, not resumable -- reruns recompute both splits from
# scratch.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unset}"

python dtm_experiment/compute_features.py --process nested_thomas
