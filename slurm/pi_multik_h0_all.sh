#!/bin/bash

#SBATCH -J pi_multik_h0_all
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 02:00:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-24
#SBATCH --output=logs/pi_multik_h0_all_%A_%a.out
#SBATCH --error=logs/pi_multik_h0_all_%A_%a.err

# ARM 1 of 3 -- [PI | H0 only | DTM{5,10,15} | 5 seeds], every process
# EXCEPT the Vihrs LGCP-family pair (lgcp, lgcp_strauss).
#
# Homology-dimension ablation: H0 persistence-image channels only
# (in_channels = 1 per k instead of 2 -- see PIMultiK's docstring). SAME
# config, calibration and hyperparameters as ARM 2
# (pi_multik_h0h1_all.sh); the ONLY difference is
#   --set method.params.homology_dims=[0]
# and --run-tag h0only, which keeps this alongside -- not overwriting --
# the H0+H1 reference at
#   results/<process>/dtm_k5+10+15/pi_multik/_runs/h0only/seed_<seed>/
#
#   5 processes  (thomas, matern_cluster, nested_thomas, aniso_thomas, strauss)
# x 5 seeds      (9371..9375)
#   = 25 array tasks, one GPU per task.
#
# Each task runs that PROCESS'S OWN config -- filename and target set both
# vary by process (3 params for thomas / matern_cluster / strauss, 5 for
# nested_thomas / aniso_thomas):
#
#   thomas          configs/runs/thomas/thomas_pi_multik_k5k10k15.yaml                 (3)
#   matern_cluster  configs/runs/matern_cluster/matern_cluster_pi_multik_k5k10k15.yaml (3)
#   nested_thomas   configs/runs/nested_thomas/pi_multik.yaml                          (5)
#   aniso_thomas    configs/runs/aniso_thomas/aniso_thomas_pi_multik_k5k10k15.yaml     (5)
#   strauss         configs/runs/strauss/strauss_pi_multik_k5k10k15.yaml              (3)
#
# The head output dim follows each config's target_label_names, so the
# 3- vs 5-parameter split is handled by using the right config per task,
# never a shared template. --set only touches method.params.homology_dims;
# pi_multik reads raw per-k diagrams.pkl and builds its own images, so the
# features: block in the YAML is irrelevant to this run.
#
# Resumable: is_done() skips a finished seed. Rerun specific failures with
#   sbatch --array=<comma,ids> slurm/pi_multik_h0_all.sh
#
# ASSUMES (per process): data/<process>/clouds.pkl, adversarial_clouds.pkl,
# data/<process>/dtm_k{5,10,15}/{,adversarial_}diagrams.pkl.
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/pi_multik_h0_all.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

PROCESSES=(thomas          matern_cluster   nested_thomas   aniso_thomas    strauss)
CONFIGS=(
  configs/runs/thomas/thomas_pi_multik_k5k10k15.yaml
  configs/runs/matern_cluster/matern_cluster_pi_multik_k5k10k15.yaml
  configs/runs/nested_thomas/pi_multik.yaml
  configs/runs/aniso_thomas/aniso_thomas_pi_multik_k5k10k15.yaml
  configs/runs/strauss/strauss_pi_multik_k5k10k15.yaml
)
NPARAMS=(3 3 5 5 3)          # estimated-parameter count per process (doc only)
SEEDS=(9371 9372 9373 9374 9375)

NSEED=${#SEEDS[@]}                              # 5
NTASKS=$(( ${#PROCESSES[@]} * NSEED ))          # 25 -- must equal the --array span (0-24)

t="${SLURM_ARRAY_TASK_ID:?run this as an array job (sbatch slurm/pi_multik_h0_all.sh)}"
if (( t >= NTASKS )); then
  echo "task $t >= $NTASKS -- nothing to do (check --array span)"; exit 0
fi

p=$(( t / NSEED ))
s=$(( t % NSEED ))
proc="${PROCESSES[$p]}"
cfg="${CONFIGS[$p]}"
seed="${SEEDS[$s]}"

echo "Host: $(hostname)"
echo "Job ${SLURM_JOB_ID:-unset}  Array task ${t}  ->  process=${proc} (${NPARAMS[$p]} params)  seed=${seed}  [H0 only]"
echo "  config=${cfg}"
echo "  Assigned GPU: ${SLURM_JOB_GPUS:-unset}   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python -u scripts/train.py "$cfg" \
  --seed "$seed" \
  --run-tag h0only \
  --set method.params.homology_dims=[0]
