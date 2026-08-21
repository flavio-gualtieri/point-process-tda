#!/bin/bash

#SBATCH -J topo_superset_shards_nested_thomas
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH -t 02:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --array=0-49%50
#SBATCH --output=logs/topo_superset_shards_nested_thomas_%A_%a.out
#SBATCH --error=logs/topo_superset_shards_nested_thomas_%A_%a.err

# Stage 1/3 of the nested_thomas topo_superset refresh: diagram computation
# only (no calibration/imaging -- that needs the FULL merged set, done in
# stage 2), sharded by cloud index -- one array task per shard, 1 CPU each,
# 50 shards = 50 CPUs at once when the whole array is running (the "split
# the raw clouds into fragments" approach). Writes fragment files under
# data/nested_thomas/topo_superset/_shards/; stage 2 (merge) consumes them.
#
# --m-values restricted to the 5 fine/mid channels -- m=0.45/0.90 dropped
# for now. The first full run (7 channels, bounded thresh) showed every
# shard sailing through m=0.01-0.20 quickly and reliably, then 42/50 shards
# blowing the previous 6h limit on m=0.45/0.90 alone (the bounded thresh
# helps much less at the coarse end by construction -- see
# scripts/featurize_topo_superset.py's docstring). 2h here is generous
# relative to what m=0.01-0.20 actually took in that run; adjust if this
# guess is still off. Revisit 0.45/0.90 separately (more walltime, and/or a
# tighter coarse-specific thresh) rather than let them block the 5 channels
# that already work.
#
# nested_thomas: 7000 train + 1000 adversarial clouds -> 140 + 20
# clouds/shard, x5 m values now = 800 diagrams/shard.
#
# CPU only, no GPU/-A needed (mirrors slurm/run_cpu.sh's convention:
# default partition, no account flag).
#
# Part of a 3-stage pipeline (sanity -> shard array -> merge) -- see
# slurm/submit_featurize_topo_superset_nested_thomas.sh to run all three
# chained via --dependency. If submitting standalone, run the sanity check
# first (slurm/featurize_topo_superset_nested_thomas_sanity.sh) --
# otherwise a bad m range fails loudly only after burning the full array.
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/featurize_topo_superset_nested_thomas_shards.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unset}  Array task: ${SLURM_ARRAY_TASK_ID:-unset}"

python -u scripts/featurize_topo_superset.py configs/runs/nested_thomas/pi_multik.yaml \
  --shard "$SLURM_ARRAY_TASK_ID" 50 --m-values 0.01 0.02 0.04 0.10 0.20
