#!/bin/bash

#SBATCH -J loglog_stats_baseline_nested_thomas
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --output=logs/loglog_stats_baseline_nested_thomas_%j.out
#SBATCH --error=logs/loglog_stats_baseline_nested_thomas_%j.err

# Track A baseline (see scripts/eval_loglog_stats_baseline.py's own
# docstring for the full rationale): fits closed-form ridge + weighted KNN
# regressors from the log(log(death/birth)) summary stats explored in
# notebooks/loglog_death_birth_vs_params_nested_thomas.ipynb, on the exact
# population/split/label-normalization pi_multik uses, and prints/report
# numbers directly comparable to
# results/nested_thomas/dtm_k5+10+15/pi_multik/seed_<seed>/results.json's
# test_loss_per_target.
#
# CPU only, no GPU/-A needed (mirrors slurm/run_cpu.sh's convention:
# default partition, no account flag) -- pure numpy/scipy over three
# ~56MB diagrams.pkl files, no torch training loop involved. --mem-per-cpu
# and -t both carry generous headroom (matches slurm/pi_multik_baseline_
# complete.sh / vihrs_nested_thomas.sh's convention for a job this size);
# the fit itself should finish in a couple of minutes.
#
# Prerequisite: data/nested_thomas/dtm_k{5,10,15}/diagrams.pkl must exist
# (scripts/featurize.py) and results/nested_thomas/dtm_k5+10+15/pi_multik/
# seed_*/results.json for the comparison column (scripts/train.py) -- the
# script runs and reports baseline-only if the latter is missing.
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/eval_loglog_stats_baseline_nested_thomas.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs notebooks/out

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unset}"

python -u scripts/eval_loglog_stats_baseline.py \
    --process nested_thomas \
    --k-values 5 10 15 \
    --out notebooks/out/loglog_stats_baseline_nested_thomas.json
