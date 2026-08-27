#!/bin/bash
# Submits the full 3-stage nested_thomas topo_superset refresh, chained via
# SLURM --dependency so each stage only starts once the previous one
# succeeds: sanity check -> 50-way sharded diagram computation (up to 50
# CPUs at once) -> merge + calibrate + image. NOT run automatically by
# anything -- run it yourself from the point-process-tda repo root:
#   bash slurm/submit_featurize_topo_superset_nested_thomas.sh
#
# afterok on the shard stage's array job ID waits for every array task to
# finish successfully before the merge stage starts (standard SLURM
# behavior for dependencies on an array job's base ID). If the sanity
# check or any shard fails, the dependent stage(s) never run -- check
# `sacct -j <jobid>` and the corresponding logs/*.err.

set -euo pipefail
cd "$(dirname "$0")/.."

SANITY=$(sbatch --parsable slurm/featurize_topo_superset_nested_thomas_sanity.sh)
echo "sanity check:      job $SANITY"

SHARDS=$(sbatch --parsable --dependency=afterok:"$SANITY" slurm/featurize_topo_superset_nested_thomas_shards.sh)
echo "shard array (x50):  job $SHARDS"

MERGE=$(sbatch --parsable --dependency=afterok:"$SHARDS" slurm/featurize_topo_superset_nested_thomas_merge.sh)
echo "merge:              job $MERGE"

echo
echo "Track with: squeue -u \$USER"
echo "or:         sacct -j $SANITY,$SHARDS,$MERGE --format=JobID,JobName%40,State,Elapsed,ExitCode"
