#!/bin/bash

#SBATCH -J cascade_regime
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --output=cascade/logs/cascade_regime_%j.out
#SBATCH --error=cascade/logs/cascade_regime_%j.err

# Regime analysis of stage 1 (cascade/regime_analysis.py): scores every candidate coordinate per
# family on the run's out-of-fold train predictions, draws the figures, and checks the boundary
# against other stage-1 models on val. CPU only.
#
#   sbatch cascade/regime_analysis.sh                               # run `default`
#   RUN=logreg STABILITY="default nn_curves" sbatch cascade/regime_analysis.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p cascade/logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"   # the module sets 1

python cascade/regime_analysis.py --config "${CONFIG:-cascade/configs/default.yaml}" --run "${RUN:-default}" \
    --stability ${STABILITY:-logreg nn_curves}
