#!/bin/bash

#SBATCH -J theory_sims
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH -t 01:30:00
#SBATCH --output=docs/theory/scripts/logs/theory_sims_%j.out
#SBATCH --error=docs/theory/scripts/logs/theory_sims_%j.err

# Simulations for docs/theory/notes.tex. Submit from the repo root:
#   sbatch docs/theory/scripts/theory_sims.sh                 # all parts
#   sbatch docs/theory/scripts/theory_sims.sh --parts kfun ph # a subset
# Writes docs/theory/scripts/out/*.npz; figures are then made on the login node by
#   python docs/theory/scripts/make_figures.py   (plotting only)

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

# one BLAS thread per pool worker
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

python docs/theory/scripts/theory_sims.py --workers "${SLURM_CPUS_PER_TASK:-1}" "$@"
