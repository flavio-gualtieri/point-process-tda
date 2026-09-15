#!/bin/bash

#SBATCH -J dimsplit_nested_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 4
#SBATCH --cpus-per-gpu=4
#SBATCH -t 02:00:00
#SBATCH --mem-per-cpu=14G
#SBATCH --gres=gpu:1
#SBATCH --array=0-49
#SBATCH --output=logs/dimsplit_nested_thomas_%A_%a.out
#SBATCH --error=logs/dimsplit_nested_thomas_%A_%a.err

# CHECK A -- per-dimension H0/H1 encoder ablation.  NESTED-THOMAS parameter
# estimation (parent_intensity, meta_offspring, meta_cluster_scale,
# mean_offspring, cluster_scale), DTM k=5 only.  Nested clusters produce H1
# void structure, so this is the parameter target where a per-dim H1 encoder
# is most likely to pay off.
#
#   base config   configs/runs/nested_thomas/dimsplit_k5.yaml
#   5 arms x 10 seeds (9371..9380)  ->  50 array tasks, one GPU each
#   task t  ->  arm = t / 10,  seed = SEEDS[t % 10]
#
#   arm idx  run-tag                    method              emb  conv          role
#   0        dimsplit__shared_e64       pi_multik            64   [32,64,128]   control (k5 baseline)         <- core
#   1        dimsplit__split_e32        pi_multik_dimsplit   32   [32,64,128]   per-dim, head width = arm 0   <- core
#   2        dimsplit__split_e64        pi_multik_dimsplit   64   [32,64,128]   per-dim, unconstrained        <- core
#   3        dimsplit__shared_e128      pi_multik            128  [32,64,128]   capacity control (shared wiring)
#   4        dimsplit__split_e32_c24    pi_multik_dimsplit   32   [24,48,96]    capacity control (conv params ~ arm 0)
#
#   Core arms only (0-2) first pass:  sbatch --array=0-29 slurm/dimsplit_nested_thomas.sh
#   Add the capacity controls only if the core arms show a signal.
#
# NOTE: k5-only nested-Thomas sits ~1 seed-sigma above the paper's fused
# DTM{5,10,15} number -- compare arms against dimsplit__shared_e64 here, not
# the fused-k table. Results (resumable):
#   results/nested_thomas/dtm_k5/pi_multik/_runs/<run-tag>/seed_<seed>/
#   results/nested_thomas/dtm_k5/pi_multik_dimsplit/_runs/<run-tag>/seed_<seed>/
#
# Rerun specific (arm,seed) failures by task id:
#   sbatch --array=<comma,ids> slurm/dimsplit_nested_thomas.sh
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/dimsplit_nested_thomas.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

CFG=configs/runs/nested_thomas/dimsplit_k5.yaml
SEEDS=(9371 9372 9373 9374 9375 9376 9377 9378 9379 9380)
N_SEEDS=${#SEEDS[@]}

ARM_TAG=(shared_e64 split_e32 split_e64 shared_e128 split_e32_c24)
ARM_METHOD=(pi_multik pi_multik_dimsplit pi_multik_dimsplit pi_multik pi_multik_dimsplit)
ARM_EMB=(64 32 64 128 32)
ARM_CONV=(- - - - "[24,48,96]")
N_ARMS=${#ARM_TAG[@]}

t="${SLURM_ARRAY_TASK_ID:?run this as an array job (sbatch slurm/dimsplit_nested_thomas.sh)}"
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

python - <<'PYEOF'
import pickle, sys
req = ["data/nested_thomas/dtm_k5/diagrams.pkl",
       "data/nested_thomas/dtm_k5/adversarial_diagrams.pkl",
       "data/nested_thomas/clouds.pkl",
       "data/nested_thomas/adversarial_clouds.pkl"]
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
