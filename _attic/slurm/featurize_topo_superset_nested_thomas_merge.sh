#!/bin/bash

#SBATCH -J topo_superset_merge_nested_thomas
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH -t 02:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/topo_superset_merge_nested_thomas_%j.out
#SBATCH --error=logs/topo_superset_merge_nested_thomas_%j.err

# Stage 2/3 of the nested_thomas topo_superset refresh: merges the 50 shard
# fragments (stage 1) per m value, validates the merged count against the
# current clouds.pkl (fails loudly on a missing/incomplete shard rather
# than silently writing a truncated dataset), then calibrates + images --
# this part genuinely can't be sharded, the imager needs the FULL training
# diagram set at once. Single process (no parallelism to be had here), but
# holds the whole merged diagram/image set in memory hence the higher
# --mem. Time limit is a guess -- check the log and adjust if it's off.
#
# --m-values restricted to the 5 fine/mid channels, matching stages 0/1 --
# m=0.45/0.90 dropped for now, see those scripts' headers. Must stay
# identical across all three stages for one run.
#
# CPU only, no GPU/-A needed (mirrors slurm/run_cpu.sh's convention:
# default partition, no account flag).
#
# Part of a 3-stage pipeline (sanity -> shard array -> merge) -- see
# slurm/submit_featurize_topo_superset_nested_thomas.sh to run all three
# chained via --dependency. If submitting standalone, only run this after
# every task of slurm/featurize_topo_superset_nested_thomas_shards.sh's
# array has finished (it will error out per-m naming the first missing
# shard file if not).
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/featurize_topo_superset_nested_thomas_merge.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unset}"

python -u scripts/featurize_topo_superset.py configs/runs/nested_thomas/pi_multik.yaml --merge 50 \
  --m-values 0.01 0.02 0.04 0.10 0.20
