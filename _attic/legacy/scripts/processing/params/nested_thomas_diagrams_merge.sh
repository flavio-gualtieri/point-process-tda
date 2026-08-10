#!/bin/bash

#SBATCH -J nested_thomas_diagrams_merge
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH -t 00:15:00
#SBATCH --output=logs/nested_thomas_diagrams_merge_%j.out
#SBATCH --error=logs/nested_thomas_diagrams_merge_%j.err

# Submit from the point-process-tda repo root, AFTER every array task in
# nested_thomas_diagrams_compute.sh (stage 2a) has finished successfully:
#   sbatch scripts/processing/params/nested_thomas_diagrams_merge.sh
# (or chain it at submission time: sbatch --dependency=afterok:<array_job_id>
# scripts/processing/params/nested_thomas_diagrams_merge.sh)
#
# STAGE 2b of 3 for the nested_thomas dataset. Combines the 10 per-group
# partials under data/params/2d/nested_thomas/_chunks/ into the final
# data/params/2d/nested_thomas/{diagrams_dtm_k5.pkl,
# adversarial_diagrams_dtm_k5.pkl}, then removes the partials
# (--clean-chunks). --n-groups here MUST match nested_thomas_diagrams_
# compute.sh's --array/--n-groups (10) -- merge fails loudly (lists the
# missing group ids) if any partial isn't there yet, so it's safe to run
# this speculatively and just re-run once the stragglers finish.
#
# Next: dtm_experiment/run_compute_features_nested_thomas.sh

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

python scripts/processing/params/diagrams_slurm.py \
  --config configs/params/processing/nested_thomas_diagrams.yaml \
  --process nested_thomas \
  --dimension 2 \
  --n-groups 10 \
  --splits train_test adversarial \
  --mode merge \
  --overwrite \
  --clean-chunks
