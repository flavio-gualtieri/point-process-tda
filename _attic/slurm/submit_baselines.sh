#!/bin/bash
# Submits the classical-baseline / error-floor batch: min contrast on K,
# min contrast on G, vihrs (thomas + nested_thomas where applicable), and
# the error-floor dataset generation + estimation. See each script's own
# header for what's covered and what's deliberately skipped (nested_thomas
# has no mincontrast baseline -- no derived closed-form K or g for that
# process anywhere in this repo; see mincontrast_g.yaml's header).
#
# error_floor_estimate_thomas.sh is submitted with a SLURM dependency on
# error_floor_generate_thomas.sh (afterok) so it can't start before its
# input data/error_floor/thomas/clouds.pkl exists -- everything else here
# is independent and submitted immediately.
#
# NOT run automatically by anything -- run it yourself from the
# point-process-tda repo root:
#   bash slurm/submit_baselines.sh
# or submit the sbatch files individually. Remember to run
# scripts/merge_error_floor_shards.py once error_floor_estimate_thomas.sh's
# 10 array tasks all finish (see that script's header) -- not automated
# here, since it's a fast local command, not a SLURM job.

set -euo pipefail
cd "$(dirname "$0")/.."

# --- vihrs (both datasets) ---
sbatch slurm/vihrs_thomas.sh
sbatch slurm/vihrs_nested_thomas.sh

# --- min contrast on K / on G (thomas only) ---
sbatch slurm/mincontrast_thomas.sh
sbatch slurm/mincontrast_g_thomas.sh

# --- error-floor: generate (both datasets), then estimate (thomas only,
#     depends on its own generate job completing) ---
sbatch slurm/error_floor_generate_nested_thomas.sh

gen_thomas_jobid=$(sbatch --parsable slurm/error_floor_generate_thomas.sh)
echo "error_floor_generate_thomas.sh submitted as job $gen_thomas_jobid"
sbatch --dependency=afterok:"$gen_thomas_jobid" slurm/error_floor_estimate_thomas.sh
