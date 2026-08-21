#!/bin/bash

#SBATCH -J pi_multik_coordconv_removed_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --output=logs/pi_multik_coordconv_removed_thomas_%A_%a.out
#SBATCH --error=logs/pi_multik_coordconv_removed_thomas_%A_%a.err

# [PI | H0+H1 | DTM{5,10,15} | 5 | CoordConv channels removed] --
# substantiates the calibration<->CoordConv claim in the writeup's SS6.3.
# method.params.coordconv=false drops CoordConvPIEncoder's two appended
# coordinate-ramp channels (see src/cloudforger/encoders/coordconv_pi.py's
# use_coords flag, added for this arm -- default true reproduces every
# other run's architecture unchanged). thomas, FUSED DTM k=5,10,15
# (configs/runs/thomas/thomas_pi_multik_k5k10k15.yaml), AT THE NEWLY
# ESTABLISHED CALIBRATION already baked into that config.
#
# --run-tag coordconv_removed ->
# results/thomas/dtm_k5+10+15/pi_multik/_runs/coordconv_removed/seed_<seed>/.
#
# 5 seeds via --array=0-4 (first 5 of the config's 10-seed list).
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/pi_multik_coordconv_removed_thomas.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

SEEDS=(9371 9372 9373 9374 9375)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unset}  Array task: ${SLURM_ARRAY_TASK_ID:-unset}  Seed: $SEED"
echo "Assigned GPU: ${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python -u scripts/train.py configs/runs/thomas/thomas_pi_multik_k5k10k15.yaml \
  --seed "$SEED" \
  --run-tag coordconv_removed \
  --set method.params.coordconv=false
