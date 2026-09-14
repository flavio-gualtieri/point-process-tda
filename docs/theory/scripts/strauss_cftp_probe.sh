#!/bin/bash

#SBATCH -J strauss_cftp_probe
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH -t 01:00:00
#SBATCH --array=0-71
#SBATCH --output=docs/theory/scripts/logs/strauss_probe_%A_%a.out
#SBATCH --error=docs/theory/scripts/logs/strauss_probe_%A_%a.err

# WP2(b) of docs/theory/generation.tex: how long do exact Strauss draws take?
# 72 cells = tau {0.3..0.8} x gamma {0, 0.1, 0.3, 0.5, 0.7, 0.9} x nbar {400, 800}
# (docs/theory/scripts/out/strauss_probe_grid.csv, beta from the DV1 lookup),
# 5 draws per cell, each capped at 600 s -> at most 50 min per task, <= 60 CPU-h in all.
#
# Prerequisite, once:  bash docs/theory/scripts/install_spatstat.sh
# Submit from the repo root:  sbatch docs/theory/scripts/strauss_cftp_probe.sh
# Results: docs/theory/scripts/out/strauss_probe/cell_<i>.csv (one row per draw).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p docs/theory/scripts/logs docs/theory/scripts/out/strauss_probe

module load R/4.5.1
export R_LIBS_USER="$HOME/R/library-4.5"     # off scratch: survives the purge

Rscript docs/theory/scripts/strauss_cftp_probe.R \
  docs/theory/scripts/out/strauss_probe_grid.csv \
  "$SLURM_ARRAY_TASK_ID" \
  "docs/theory/scripts/out/strauss_probe/cell_${SLURM_ARRAY_TASK_ID}.csv" \
  5 600
