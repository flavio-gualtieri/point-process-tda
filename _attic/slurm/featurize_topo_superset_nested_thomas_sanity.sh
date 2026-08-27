#!/bin/bash

#SBATCH -J topo_superset_sanity_nested_thomas
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH -t 00:15:00
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/topo_superset_sanity_nested_thomas_%j.out
#SBATCH --error=logs/topo_superset_sanity_nested_thomas_%j.err

# Stage 0/3 of the nested_thomas topo_superset refresh: fast (~10 diagrams)
# degeneracy pre-flight check on the CURRENT clouds.pkl, before spending the
# 50-CPU shard array's budget. CPU only, no GPU/-A needed (mirrors
# slurm/run_cpu.sh's convention: default partition, no account flag).
#
# --m-values restricted to the 5 fine/mid channels -- m=0.45/0.90 dropped
# for now. The first full run (7 channels, bounded thresh) showed m=0.01-
# 0.20 finishing fast and reliably on every shard, while m=0.45/0.90 alone
# blew the 6h shard time limit on 42/50 shards (the bounded thresh helps
# much less at the coarse end by construction -- see
# scripts/featurize_topo_superset.py's docstring). Revisit 0.45/0.90
# separately (more walltime, and/or a tighter coarse-specific thresh)
# rather than let them block the 5 channels that already work.
#
# Part of a 3-stage pipeline (sanity -> shard array -> merge) -- see
# slurm/submit_featurize_topo_superset_nested_thomas.sh to run all three
# chained via --dependency, or submit this one alone first to sanity-check
# quickly before committing to the full array.
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/featurize_topo_superset_nested_thomas_sanity.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unset}"

python -u scripts/featurize_topo_superset.py configs/runs/nested_thomas/pi_multik.yaml --sanity-only \
  --m-values 0.01 0.02 0.04 0.10 0.20
