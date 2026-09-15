#!/bin/bash

#SBATCH -J vihrs_classify
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 4
#SBATCH --cpus-per-gpu=4
#SBATCH -t 01:00:00
#SBATCH --mem-per-cpu=14G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --output=logs/vihrs_classify_%A_%a.out
#SBATCH --error=logs/vihrs_classify_%A_%a.err

# PROCESS CLASSIFICATION -- vihrs baseline adapted to classification.
#
#   [ Ripley L(r)-r + n(x) | Vihrs 2022 1-D CNN | 5-way softmax | 5 seeds ]
#
# The classical-second-order-statistic comparison arm for the pi_multik
# classify run (slurm/pi_multik_classify_h0h1.sh). Same clouds, same seeds,
# same train/val/test split (vihrs reads the per-k diagram bundles' seed/
# label arrays only, to reproduce pi_multik's split exactly -- never the
# diagrams themselves), so the two results.json's compare head-to-head.
#
#   config   configs/runs/classification/vihrs_lr_nx_k5k10k15.yaml
#   task     method.params.task: classify  (CrossEntropyLoss + per-class recall)
#   classes  aniso_thomas, matern_cluster, nested_thomas, strauss, thomas
#   seeds    9371..9375  -> 5 array tasks, one GPU each
#
# Reference run: NO --run-tag, so it owns the default
#   results/classification/raw/vihrs/seed_<seed>/
# path (vihrs is filtration-independent -> the "raw" tag; see paths.py).
# Resumable: is_done() skips a seed whose results.pt already exists, so a
# plain re-submit only fills gaps. Rerun specific failures with
#   sbatch --array=<comma,ids> slurm/vihrs_classify.sh
#
# Resources: much lighter than pi_multik classify -- no persistence-image
# tensors. Each task loads clouds.pkl (~232 MB) + the 3 diagram bundles
# (~760 MB, freed after the seed/label read) and builds the L(r)-r feature
# set once (~2 min for 35k + 5k clouds, single-threaded O(points^2)). That
# feature set is cached to
#   data/classification/{,adversarial_}clouds.lfunc_classify_cache.npz
# with an atomic write (tmp file + os.replace), so the 5 array tasks
# recomputing it in parallel can't tear each other's file; a later re-submit
# loads the cache in ~1 s. 4 CPUs x 14 GB = 56 GB is generous; 1 h wall is
# ample for ~2 min prep + <=200 short epochs with early stopping.
#
# ASSUMES:
#   data/classification/clouds.pkl, adversarial_clouds.pkl
#   data/classification/dtm_k{5,10,15}/{,adversarial_}diagrams.pkl
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/vihrs_classify.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

CFG=configs/runs/classification/vihrs_lr_nx_k5k10k15.yaml
SEEDS=(9371 9372 9373 9374 9375)

t="${SLURM_ARRAY_TASK_ID:?run this as an array job (sbatch slurm/vihrs_classify.sh)}"
if (( t >= ${#SEEDS[@]} )); then
  echo "task $t >= ${#SEEDS[@]} -- nothing to do (check --array span)"; exit 0
fi
seed="${SEEDS[$t]}"

echo "Host: $(hostname)"
echo "Job ${SLURM_JOB_ID:-unset}  Array task ${t}  ->  process=classification  seed=${seed}  [vihrs classify]"
echo "  config=${CFG}"
echo "  Assigned GPU: ${SLURM_JOB_GPUS:-unset}   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Preflight: every per-k diagram bundle + both cloud bundles must exist AND
# fully unpickle -- a truncated .pkl otherwise fails deep inside prep after
# burning queue time. (Same guard as slurm/pi_multik_classify_h0h1.sh.)
python - <<'PYEOF'
import pickle, sys
req = []
for k in (5, 10, 15):
    req += [f"data/classification/dtm_k{k}/diagrams.pkl",
            f"data/classification/dtm_k{k}/adversarial_diagrams.pkl"]
req += ["data/classification/clouds.pkl", "data/classification/adversarial_clouds.pkl"]
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
    print("preflight FAILED -- data/classification/ not ready:", *bad, sep="\n  ")
    sys.exit(1)
print("preflight OK -- all 6 diagram bundles + 2 cloud bundles intact")
PYEOF

python -u scripts/train.py "$CFG" --seed "$seed"
