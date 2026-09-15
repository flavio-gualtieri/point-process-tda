#!/bin/bash

#SBATCH -J generate_vihrs
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH -t 24:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/generate_vihrs_%A_%a.out
#SBATCH --error=logs/generate_vihrs_%A_%a.err

# Point-cloud generation for the three Vihrs (2022) LGCP-family processes.
# Each is an independent RunConfig -> one array task, run in parallel:
#
#   0  lgcp           (Sec. 3.1)  -- log-Gaussian Cox, 8000 clouds
#   1  strauss        (Sec. 3.2)  -- Strauss inhibition, 8000 clouds
#   2  lgcp_strauss   (Sec. 3.3)  -- doubly intractable, birth-death MH, 8000 clouds
#
# Output lands under data/<process>/{clouds.pkl, adversarial_clouds.pkl,
# cloud_generation_manifest.yaml}. generate.py skips a process whose
# clouds.pkl already exists, so a plain re-submit only fills gaps; pass
# --force below to regenerate.
#
# lgcp_strauss is the slow one (MCMC per cloud); 24 h is generous headroom.
# If a task times out, rerun just that id:
#   sbatch --array=2 slurm/generate_vihrs.sh
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/generate_vihrs.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

CONFIGS=(
  configs/runs/lgcp/lgcp_vihrs.yaml
  configs/runs/strauss/strauss_vihrs.yaml
  configs/runs/lgcp_strauss/lgcp_strauss_vihrs.yaml
)

t="${SLURM_ARRAY_TASK_ID:?run this as an array job (sbatch slurm/generate_vihrs.sh)}"
cfg="${CONFIGS[$t]}"

echo "Host: $(hostname)"
echo "Job ${SLURM_JOB_ID:-unset}  Array task ${t}  ->  ${cfg}"

python -u scripts/generate.py "$cfg"
