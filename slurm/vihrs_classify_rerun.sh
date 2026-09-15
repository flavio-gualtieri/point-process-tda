#!/bin/bash

#SBATCH -J vihrs_classify_rerun
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 4
#SBATCH --cpus-per-gpu=4
#SBATCH -t 01:00:00
#SBATCH --mem-per-cpu=14G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --output=logs/vihrs_classify_rerun_%A_%a.out
#SBATCH --error=logs/vihrs_classify_rerun_%A_%a.err

# RE-RUN of slurm/vihrs_classify.sh after the classification labels changed:
# aniso_thomas is now FOLDED INTO thomas (manifest merged_labels), so the
# task is 4-way (matern_cluster, nested_thomas, strauss, thomas) with class 3
# twice the size of the others. The old 5-way results in
#   results/classification/raw/vihrs/seed_<seed>/
# are stale -- --force makes scripts/train.py OVERWRITE them in place.
#
# The L(r)-r feature cache
#   data/classification/{,adversarial_}clouds.lfunc_classify_cache.npz
# is keyed only on r_grid + cloud count, NOT on the point coordinates, so a
# cache left over from the previous clouds.pkl would be silently reused.
# --set method.params.recompute_features=true forces a fresh L(r)-r build
# from the current clouds.pkl (~1.5 min for 35k + ~15 s for 5k adversarial;
# the atomic tmp+os.replace write tolerates the 5 array tasks recomputing in
# parallel). Drop that flag on any later gap-fill re-submit once the cache
# is valid.
#
#   config  configs/runs/classification/vihrs_lr_nx_k5k10k15.yaml
#           (target_label_names: [matern_cluster, nested_thomas, strauss, thomas])
#   seeds   9371..9375  -> 5 array tasks, one GPU each
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/vihrs_classify_rerun.sh

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

t="${SLURM_ARRAY_TASK_ID:?run this as an array job (sbatch slurm/vihrs_classify_rerun.sh)}"
if (( t >= ${#SEEDS[@]} )); then
  echo "task $t >= ${#SEEDS[@]} -- nothing to do (check --array span)"; exit 0
fi
seed="${SEEDS[$t]}"

echo "Host: $(hostname)"
echo "Job ${SLURM_JOB_ID:-unset}  Array task ${t}  ->  process=classification  seed=${seed}  [vihrs RE-RUN, --force, features recomputed]"
echo "  config=${CFG}"
echo "  Assigned GPU: ${SLURM_JOB_GPUS:-unset}   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Preflight: bundles intact + 4-way label_names.
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
b = pickle.load(open("data/classification/dtm_k5/diagrams.pkl", "rb"))
ln = list(b["label_names"])
if ln != ["matern_cluster", "nested_thomas", "strauss", "thomas"]:
    print(f"preflight FAILED -- unexpected label_names {ln}; config target_label_names must match."); sys.exit(1)
print("preflight OK -- 8 bundles intact, label_names =", ln)
PYEOF

python -u scripts/train.py "$CFG" --seed "$seed" --force --set method.params.recompute_features=true
