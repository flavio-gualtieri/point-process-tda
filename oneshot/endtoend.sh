#!/bin/bash

#SBATCH -J oneshot_e2e
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH -t 12:00:00
#SBATCH --output=oneshot/logs/oneshot_e2e_%j.out
#SBATCH --error=oneshot/logs/oneshot_e2e_%j.err

# End-to-end kernel / DSS score of the config's evaluation.pipelines (endtoend.py). One process per
# cloud; the cascade's 300-cloud run took ~40-65 min on 16 workers.
#
#   sbatch oneshot/endtoend.sh                       # score_on from the config (heldout)
#   SCORE_ON=same sbatch oneshot/endtoend.sh         # scored on the pattern the fit came from
#   LIMIT=16 sbatch oneshot/endtoend.sh              # smoke test

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p oneshot/logs
module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u
export OMP_NUM_THREADS=1            # one process per cloud; keep BLAS from oversubscribing
python -u oneshot/endtoend.py --config "${CONFIG:-oneshot/configs/default.yaml}" \
    ${SCORE_ON:+--score-on "$SCORE_ON"} ${LIMIT:+--limit "$LIMIT"}
