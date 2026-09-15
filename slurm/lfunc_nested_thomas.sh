#!/bin/bash

#SBATCH -J lfunc_nested_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 4
#SBATCH --cpus-per-gpu=4
#SBATCH -t 02:00:00
#SBATCH --mem-per-cpu=14G
#SBATCH --gres=gpu:1
#SBATCH --array=0-29
#SBATCH --output=logs/lfunc_nested_thomas_%A_%a.out
#SBATCH --error=logs/lfunc_nested_thomas_%A_%a.err

# L-REPARAMETERIZATION 2x2 -- NESTED THOMAS parameter estimation, DTM k=5.
# See slurm/lfunc_thomas.sh for the full rationale.
#
#   arm  filtration  include_lfunc  run-tag              config
#   A    dtm_k5      0              dimsplit__shared_e64  ALREADY RUN -- do not resubmit
#                                                         (L = 0.2072 +/- 0.0060, n=10)
#   0/B  l_dtm_k5    0              lfunc__B_ldtm         lfunc_k5.yaml
#   1/C  dtm_k5      8              lfunc__C_dtm_lx       dimsplit_k5.yaml
#   2/D  l_dtm_k5    8              lfunc__D_ldtm_lx      lfunc_k5.yaml
#
# READ D vs C, NOT A vs B (the transform moves the second-order trend into L,
# so arm B is expected to look worse than A by construction).
#
# Nested Thomas is the process where the topological branch already carries its
# own weight (it is the one process where PH beat vihrs), so it is the most
# informative case for whether the L-reparameterization sharpens or blunts that.
#
#   3 arms x 10 seeds (9371..9380) -> 30 array tasks
#   task t -> arm = t / 10, seed = SEEDS[t % 10]
#
# PREREQUISITE: sbatch slurm/lfunc_transform.sh
# Submit from the repo root:  sbatch slurm/lfunc_nested_thomas.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

SEEDS=(9371 9372 9373 9374 9375 9376 9377 9378 9379 9380)
N_SEEDS=${#SEEDS[@]}

ARM_TAG=(B_ldtm C_dtm_lx D_ldtm_lx)
ARM_CFG=(configs/runs/nested_thomas/lfunc_k5.yaml configs/runs/nested_thomas/dimsplit_k5.yaml configs/runs/nested_thomas/lfunc_k5.yaml)
ARM_LFUNC=(0 8 8)
N_ARMS=${#ARM_TAG[@]}

t="${SLURM_ARRAY_TASK_ID:?run this as an array job (sbatch slurm/lfunc_nested_thomas.sh)}"
if (( t >= N_ARMS * N_SEEDS )); then
  echo "task $t >= $((N_ARMS * N_SEEDS)) -- nothing to do (check --array span)"; exit 0
fi
arm=$(( t / N_SEEDS ))
seed="${SEEDS[$(( t % N_SEEDS ))]}"
tag="${ARM_TAG[$arm]}"
cfg="${ARM_CFG[$arm]}"
lfunc="${ARM_LFUNC[$arm]}"

echo "Host: $(hostname)"
echo "Job ${SLURM_JOB_ID:-unset}  task ${t}  ->  arm=${tag}  seed=${seed}  include_lfunc=${lfunc}"
echo "  config=${cfg}  run-tag=lfunc__${tag}"
echo "  Assigned GPU: ${SLURM_JOB_GPUS:-unset}   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python - "$cfg" <<'PYEOF'
import pickle, sys, yaml
from pathlib import Path
cfg = yaml.safe_load(open(sys.argv[1]))
tag = "l_dtm_k5" if cfg["filtration"][0]["name"] == "l_dtm" else "dtm_k5"
req = [f"data/nested_thomas/{tag}/diagrams.pkl", f"data/nested_thomas/{tag}/adversarial_diagrams.pkl",
       "data/nested_thomas/clouds.pkl", "data/nested_thomas/adversarial_clouds.pkl",
       "data/nested_thomas/clouds.lfunc_cache.npz", "data/nested_thomas/adversarial_clouds.lfunc_cache.npz"]
bad = []
for f in req:
    try:
        if f.endswith(".npz"):
            if not Path(f).exists(): raise FileNotFoundError
        else:
            with open(f, "rb") as fh: pickle.load(fh)
    except FileNotFoundError:
        bad.append(f"{f}: MISSING")
    except Exception as e:
        bad.append(f"{f}: UNREADABLE ({type(e).__name__})")
if bad:
    print("preflight FAILED:", *bad, sep="\n  ")
    print("  (l_dtm_k5 missing? run: sbatch slurm/lfunc_transform.sh)")
    sys.exit(1)
print(f"preflight OK ({tag})")
PYEOF

python -u scripts/train.py "$cfg" --seed "$seed" --run-tag "lfunc__${tag}" \
    --set "method.params.include_lfunc=${lfunc}"
