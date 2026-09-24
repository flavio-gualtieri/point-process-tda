#!/bin/bash

#SBATCH -J cascade_eval
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH -t 12:00:00
#SBATCH --output=cascade/logs/cascade_eval_%j.out
#SBATCH --error=cascade/logs/cascade_eval_%j.err

# End-to-end score of the assembled cascade (cascade/evaluate.py). Sized as cascade/scoring/power.sh:
# LGCP simulation at large grids needs ~2 GB per worker, hence the memory.
#   SET=hgb sbatch cascade/evaluate.sh                                  # a set under evaluation.variants
#   SET=mixed LIMIT=10 sbatch -p computeshort -t 1:00:00 cascade/evaluate.sh      # smoke test

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p cascade/logs
module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u
export OMP_NUM_THREADS=1            # one process per cloud; keep BLAS from oversubscribing

python cascade/evaluate.py --config "${CONFIG:-cascade/configs/default.yaml}" --set "${SET:?SET=<variant set> required}" \
    ${LIMIT:+--limit "$LIMIT"}
