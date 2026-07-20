#!/bin/bash

#SBATCH -J validate_nested_thomas
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH -t 00:15:00
#SBATCH --output=logs/validate_nested_thomas_%j.out
#SBATCH --error=logs/validate_nested_thomas_%j.err

# Submit from the point-process-tda repo root, AFTER the full nested_thomas
# pipeline (generate -> diagrams compute+merge -> compute_features) has
# finished: sbatch dtm_experiment/run_validate_nested_thomas_data.sh
#
# Runs dtm_experiment/validate_nested_thomas_data.py, which loads every
# stage's output under data/params/2d/nested_thomas/ and checks counts,
# shapes, seed-set consistency across stages, realized parameter ranges vs.
# the manifest, and NaN/Inf -- see that file's docstring for the full list.
# Prints a report to stdout/the log and exits non-zero if any check fails.
# 15 min/8G is generous for this dataset's size (~1,000 clouds) -- this is
# just loading a handful of pickles and computing numpy summary stats, not
# real compute.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unset}"

python dtm_experiment/validate_nested_thomas_data.py
