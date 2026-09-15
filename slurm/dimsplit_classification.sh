#!/bin/bash

#SBATCH -J dimsplit_classification
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 6
#SBATCH --cpus-per-gpu=6
#SBATCH -t 03:00:00
#SBATCH --mem-per-cpu=14G
#SBATCH --gres=gpu:1
#SBATCH --array=0-24
#SBATCH --output=logs/dimsplit_classification_%A_%a.out
#SBATCH --error=logs/dimsplit_classification_%A_%a.err

# CHECK A -- does giving H0 and H1 their own encoder (instead of stacking
# them as 2 channels into one shared CoordConvPIEncoder) help?  PROCESS
# CLASSIFICATION, DTM k=5 only, 4-way softmax.
#
#   base config   configs/runs/classification/dimsplit_k5.yaml
#   5 arms x 5 seeds (9371..9375)  ->  25 array tasks, one GPU each
#   task t  ->  arm = t / 5,  seed = SEEDS[t % 5]
#
#   arm idx  run-tag                    method              emb  conv          role
#   0        dimsplit__shared_e64       pi_multik            64   [32,64,128]   control (k5 baseline)         <- core
#   1        dimsplit__split_e32        pi_multik_dimsplit   32   [32,64,128]   per-dim, head width = arm 0   <- core
#   2        dimsplit__split_e64        pi_multik_dimsplit   64   [32,64,128]   per-dim, unconstrained        <- core
#   3        dimsplit__shared_e128      pi_multik            128  [32,64,128]   capacity control (shared wiring)
#   4        dimsplit__split_e32_c24    pi_multik_dimsplit   32   [24,48,96]    capacity control (conv params ~ arm 0)
#
#   Core arms only (0-2) first pass:  sbatch --array=0-14 slurm/dimsplit_classification.sh
#   Add the capacity controls only if the core arms show a signal.
#
# Results (resumable -- is_done() skips a finished seed):
#   results/classification/dtm_k5/pi_multik/_runs/<run-tag>/seed_<seed>/
#   results/classification/dtm_k5/pi_multik_dimsplit/_runs/<run-tag>/seed_<seed>/
# Compare later with scripts/evaluate.py's method@run-tag syntax; pair
# seed->seed (do NOT pool), read test_accuracy_per_class not just the
# imbalance-inflated overall accuracy.
#
# Rerun specific (arm,seed) failures by their task id:
#   sbatch --array=<comma,ids> slurm/dimsplit_classification.sh
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/dimsplit_classification.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

CFG=configs/runs/classification/dimsplit_k5.yaml
SEEDS=(9371 9372 9373 9374 9375)
N_SEEDS=${#SEEDS[@]}

# arm i is (run-tag, method, embedding_dim, conv_channels).  conv "-" = keep
# the config default [32,64,128].
ARM_TAG=(shared_e64 split_e32 split_e64 shared_e128 split_e32_c24)
ARM_METHOD=(pi_multik pi_multik_dimsplit pi_multik_dimsplit pi_multik pi_multik_dimsplit)
ARM_EMB=(64 32 64 128 32)
ARM_CONV=(- - - - "[24,48,96]")
N_ARMS=${#ARM_TAG[@]}

t="${SLURM_ARRAY_TASK_ID:?run this as an array job (sbatch slurm/dimsplit_classification.sh)}"
if (( t >= N_ARMS * N_SEEDS )); then
  echo "task $t >= $((N_ARMS * N_SEEDS)) -- nothing to do (check --array span)"; exit 0
fi
arm=$(( t / N_SEEDS ))
seed="${SEEDS[$(( t % N_SEEDS ))]}"
tag="${ARM_TAG[$arm]}"
method="${ARM_METHOD[$arm]}"
emb="${ARM_EMB[$arm]}"
conv="${ARM_CONV[$arm]}"

SET_ARGS=(--set "method.name=${method}" --set "method.params.embedding_dim=${emb}")
if [[ "$conv" != "-" ]]; then
  SET_ARGS+=(--set "method.params.conv_channels=${conv}")
fi

echo "Host: $(hostname)"
echo "Job ${SLURM_JOB_ID:-unset}  task ${t}  ->  arm=${tag} (${method}, emb=${emb}, conv=${conv})  seed=${seed}"
echo "  config=${CFG}  run-tag=dimsplit__${tag}"
echo "  Assigned GPU: ${SLURM_JOB_GPUS:-unset}   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Preflight: k=5 diagram bundles + both cloud bundles must exist and fully
# unpickle (a truncated .pkl otherwise fails deep in training after burning
# queue time).
python - <<'PYEOF'
import pickle, sys
req = ["data/classification/dtm_k5/diagrams.pkl",
       "data/classification/dtm_k5/adversarial_diagrams.pkl",
       "data/classification/clouds.pkl",
       "data/classification/adversarial_clouds.pkl"]
bad = []
for f in req:
    try:
        with open(f, "rb") as fh:
            pickle.load(fh)
    except FileNotFoundError:
        bad.append(f"{f}: MISSING")
    except Exception as e:
        bad.append(f"{f}: UNREADABLE ({type(e).__name__})")
if bad:
    print("preflight FAILED:", *bad, sep="\n  "); sys.exit(1)
print("preflight OK")
PYEOF

python -u scripts/train.py "$CFG" --seed "$seed" --run-tag "dimsplit__${tag}" "${SET_ARGS[@]}"
