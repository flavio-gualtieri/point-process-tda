#!/bin/bash

#SBATCH -J cascade_nn
#SBATCH -A pilot_sae_gpu
#SBATCH -p sae
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH --output=cascade/logs/cascade_nn_%j.out
#SBATCH --error=cascade/logs/cascade_nn_%j.err

# Neural stage 1 (cascade/stage1_nn.py) on a GPU. Partition and account as slurm/train.sh, whose
# notes explain them. Then run the rest of the cascade on CPU from stage 2:
#
#   jid=$(CONFIG=cascade/configs/nn.yaml sbatch --parsable cascade/stage1_nn.sh)
#   FROM=stage2 CONFIG=cascade/configs/nn.yaml sbatch --dependency=afterok:$jid cascade/run.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p cascade/logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python cascade/stage1_nn.py --config "${CONFIG:-cascade/configs/nn.yaml}"
