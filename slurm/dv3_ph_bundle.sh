#!/bin/bash

#SBATCH -J dv3_ph_bundle
#SBATCH -p compute,computeshort
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH --output=logs/dv3_ph_bundle_%j.out
#SBATCH --error=logs/dv3_ph_bundle_%j.err

# Merge every family's diagrams into data/dv3/<set>/_classify/<tag>/diagrams.pkl
# (tags present in ALL of a set's families) so the PH classification cells can
# read them. Clouds are already merged (slurm/dv3_classical_prep.sh task 12),
# so only the diagram bundles are (re)written; up-to-date ones are skipped.
# Normally submitted by slurm/dv3_ph_submit.sh after the train diagrams merge.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

DTM_K="${DTM_K:-5}"
python -u scripts/processing/dv3_classification_bundle.py --sets train A B C

missing=0
for s in train A B C; do
  f="data/dv3/${s}/_classify/dtm_k${DTM_K}/diagrams.pkl"
  if [[ -f "$f" ]]; then echo "  ok       $f"; else echo "  MISSING  $f"; missing=1; fi
done
exit "$missing"
