#!/bin/bash

#SBATCH -J oneshot_mc
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH --output=oneshot/logs/oneshot_mc_%x_%A_%a.out
#SBATCH --error=oneshot/logs/oneshot_mc_%x_%A_%a.err

# One step of the minimum-contrast baseline (oneshot/mincontrast.py): STEP=clouds | fit | assemble.
# fit runs as an array over 6 models x mincontrast.chunks; the task id picks (model, chunk).
# Normally submitted by oneshot/mincontrast_submit.sh.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p oneshot/logs
module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u
export OMP_NUM_THREADS=1            # one process per cloud
CONFIG="${CONFIG:-oneshot/configs/default.yaml}"
case "${STEP:?STEP=clouds|fit|assemble}" in
  fit) python -u oneshot/mincontrast.py --config "$CONFIG" fit --task "$SLURM_ARRAY_TASK_ID" ;;
  *)   python -u oneshot/mincontrast.py --config "$CONFIG" "$STEP" ;;
esac
