#!/bin/bash

#SBATCH -J pi_multik_dropoutprobe_shared_conv_avg
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 00:59:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/pi_multik_dropoutprobe_shared_conv_avg_%j.out
#SBATCH --error=logs/pi_multik_dropoutprobe_shared_conv_avg_%j.err

# Single-seed probe: does restoring dropout in the ConvFusion path fix the
# training collapse? shared + conv/avg, seed 9373 (failed at 0.910 test
# loss in encsweep_shared_conv_avg -- see encoder_fusion_sweep_report.md).
# scale_fusion_dropout=0.2 / fusion_dropout=0.1 are the pre-refactor
# TowerConv/PIMultiKTowers defaults, which the encoder/fusion consolidation
# silently zeroed out for every combo (see this run's config vs. git
# history of pi_multik_towers.py) -- this is a direct test of that as the
# instability's cause, not a rigorous rerun. If this one seed clears the
# ~0.5 failure threshold, the full 5-seed dropout sweep is the next step.
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/pi_multik_dropoutprobe_shared_conv_avg.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unset}"
echo "Assigned GPU: ${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python -u scripts/train.py configs/runs/nested_thomas/pi_multik.yaml \
  --seed 9373 \
  --run-tag encsweep_shared_conv_avg_dropoutprobe \
  --set method.params.encoder_mode=shared \
  --set method.params.fusion_mode=conv \
  --set method.params.fusion_pool=avg \
  --set method.params.scale_fusion_dropout=0.2 \
  --set method.params.fusion_dropout=0.1
