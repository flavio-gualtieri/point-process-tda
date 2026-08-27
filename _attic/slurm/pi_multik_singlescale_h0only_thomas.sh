#!/bin/bash

#SBATCH -J pi_multik_singlescale_h0only_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-9
#SBATCH --output=logs/pi_multik_singlescale_h0only_thomas_%A_%a.out
#SBATCH --error=logs/pi_multik_singlescale_h0only_thomas_%A_%a.err

# [PI | H0 | DTM5]: the actual ParamNet configuration as specified in
# short_report.tex ssec:method -- DTM_5 filtration ONLY (n_k=1,
# configs/runs/thomas/pi_multik_k5.yaml) combined with the H0-ONLY
# homology restriction (in_channels=1 instead of 2 -- see PIMultiK's
# docstring), both simplifications applied AT ONCE.
#
# Every existing ablation (pi_multik_h0only_thomas.sh,
# pi_multik_singlescale_thomas_k5.sh, ...) only ever changes ONE axis
# at a time relative to the DTM{5,10,15}+H0+H1 starting configuration;
# no run combines them, so this is the missing headline number for the
# model as described in the method section (see short_report.tex's
# "Outstanding experiments" bullet).
#
# --run-tag h0only keeps this alongside, not overwriting, the existing
# H0+H1 single-scale run at results/thomas/dtm_k5/pi_multik/seed_<seed>/;
# this lands at results/thomas/dtm_k5/pi_multik/_runs/h0only/seed_<seed>/.
#
# All 10 configured seeds via --array=0-9, matching the other headline
# n=10 entries in Table tab:classical-comparison / tab:branches.
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/pi_multik_singlescale_h0only_thomas.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

SEEDS=(9371 9372 9373 9374 9375 9376 9377 9378 9379 9380)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unset}  Array task: ${SLURM_ARRAY_TASK_ID:-unset}  Seed: $SEED"
echo "Assigned GPU: ${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python -u scripts/train.py configs/runs/thomas/pi_multik_k5.yaml \
  --seed "$SEED" \
  --run-tag h0only \
  --set method.params.homology_dims=[0]
