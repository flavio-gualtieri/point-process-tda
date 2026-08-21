#!/bin/bash

#SBATCH -J pi_multik_headline_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-9
#SBATCH --output=logs/pi_multik_headline_thomas_%A_%a.out
#SBATCH --error=logs/pi_multik_headline_thomas_%A_%a.err

# HEADLINE configuration: [PI | H0+H1 | DTM{5,10,15} | 10]. thomas, FUSED
# DTM k=5,10,15, shared-weight encoder + flat concat (the pi_multik
# defaults -- no encoder_mode/fusion_mode/coordconv/include_log_n
# overrides), AT THE NEWLY ESTABLISHED CALIBRATION baked into
# configs/runs/thomas/thomas_pi_multik_k5k10k15.yaml itself (resolution=64,
# sigma_pixels=0.5, pd_calibration_coverage=0.95, pad=1.05 code default) --
# no --set needed here, unlike the sibling pi_multik_confirm_shared_thomas.sh/
# pi_multik_confirm_independent_thomas.sh scripts written before that
# calibration was the config's own default (they still pass
# --set method.params.sigma_pixels=0.5 explicitly, now redundant but
# harmless).
#
# nested_thomas counterpart: no new script needed -- see
# pi_multik_baseline_complete.sh (identical 10-seed/array-per-task pattern
# against configs/runs/nested_thomas/pi_multik.yaml, already correct now
# that config's own sigma_pixels/coverage defaults were updated).
#
# All 10 configured seeds via --array=0-9. No --run-tag: this is the
# reference result other arms' --run-tag variants (h0only, h1only,
# coordconv_removed, logn_removed, shuffled_labels, encsweep_*) are compared
# against, so it must own the default (untagged)
# results/thomas/dtm_k5+10+15/pi_multik/seed_<seed>/ path.
#
# ASSUMES data/thomas/clouds.pkl and data/thomas/dtm_k{5,10,15}/diagrams.pkl
# already exist.
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/pi_multik_headline_thomas.sh

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

python -u scripts/train.py configs/runs/thomas/thomas_pi_multik_k5k10k15.yaml \
  --seed "$SEED"
