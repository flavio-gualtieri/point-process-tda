#!/bin/bash

# PersLay with a fixed z-score of the pooled vector (--perslay-norm zscore), a couple of seeds for
# every diagram type: each filtration x {H0, H1, H0+H1}. Submits through slurm/matrix.sh, so each
# task is a regular slurm/train.sh array job with VARIANTS=perslay_z.
#
#   bash slurm/perslay_z.sh              # print the sbatch commands, submit nothing
#   bash slurm/perslay_z.sh submit       # submit them
#
# Output goes to results/<task>/<group>/<filtration>/perslay_z_h<dims>/seed_<n>, beside the
# unnormalized perslay_h<dims> runs, never over them. The default seeds, 1 and 2, match seeds that
# already exist for perslay_h<dims>, and calibration draws no random numbers, so each new run starts
# from the SAME initialization and data order as its unnormalized twin: the comparison is paired.
#
#   python scripts/regimes.py --task classify --reference rips/perslay_h01
#
# Defaults to classification only, the task where PersLay's seed spread showed up; any axis can be
# overridden as for slurm/matrix.sh, e.g. the parameter tasks as well:
#
#   TASKS="classify params:thomas params:matern2" bash slurm/perslay_z.sh submit

set -euo pipefail

cd "$(dirname "$0")/.."

export TASKS="${TASKS:-classify}"
export FILTRATIONS="${FILTRATIONS-rips alpha_diameter dtm_k5 dtm_k10 dtm_k15 dtm_k20}"
export DIMS="${DIMS:-0 1 0,1}"
export VARIANTS="perslay_z"
export CURVES=""                    # the classical arm has nothing to normalize
export SEEDS="${SEEDS:-1 2}"
export SEED_CHUNK="${SEED_CHUNK:-2}"

bash slurm/matrix.sh "${1:-print}" | sed "s|bash slurm/matrix.sh submit|bash slurm/perslay_z.sh submit|"
