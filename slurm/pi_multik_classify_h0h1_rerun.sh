#!/bin/bash

#SBATCH -J pi_multik_classify_h0h1_rerun
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 04:00:00
#SBATCH --mem-per-cpu=14G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --output=logs/pi_multik_classify_h0h1_rerun_%A_%a.out
#SBATCH --error=logs/pi_multik_classify_h0h1_rerun_%A_%a.err

# RE-RUN of slurm/pi_multik_classify_h0h1.sh after the classification labels
# changed: aniso_thomas is now FOLDED INTO thomas (manifest merged_labels),
# so the task is 4-way (matern_cluster, nested_thomas, strauss, thomas) with
# class 3 twice the size of the others. The old 5-way results in
#   results/classification/dtm_k5+10+15/pi_multik/seed_<seed>/
# are stale -- this passes --force so scripts/train.py OVERWRITES them in
# place (results.pt / results.json / model.pt) instead of is_done()-skipping.
#
# pi_multik reads the per-k diagram bundles fresh every run (no feature
# cache), so the relabelled data is picked up automatically; only --force
# and the config's updated target_label_names matter.
#
#   config  configs/runs/classification/pi_multik_h0h1_k5k10k15.yaml
#           (target_label_names: [matern_cluster, nested_thomas, strauss, thomas])
#   seeds   9371..9375  -> 5 array tasks, one GPU each
#
# A fresh row per seed is appended to results/experiments.jsonl (the old
# 5-way rows stay, distinguished by git_commit / timestamp).
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/pi_multik_classify_h0h1_rerun.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

CFG=configs/runs/classification/pi_multik_h0h1_k5k10k15.yaml
SEEDS=(9371 9372 9373 9374 9375)

t="${SLURM_ARRAY_TASK_ID:?run this as an array job (sbatch slurm/pi_multik_classify_h0h1_rerun.sh)}"
if (( t >= ${#SEEDS[@]} )); then
  echo "task $t >= ${#SEEDS[@]} -- nothing to do (check --array span)"; exit 0
fi
seed="${SEEDS[$t]}"

echo "Host: $(hostname)"
echo "Job ${SLURM_JOB_ID:-unset}  Array task ${t}  ->  process=classification  seed=${seed}  [pi_multik RE-RUN, --force]"
echo "  config=${CFG}"
echo "  Assigned GPU: ${SLURM_JOB_GPUS:-unset}   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Preflight: every per-k diagram bundle + both cloud bundles must exist AND
# fully unpickle before burning queue time.
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
# sanity: labels must be the 4-way set
import numpy as np
b = pickle.load(open("data/classification/dtm_k5/diagrams.pkl", "rb"))
ln = list(b["label_names"])
if ln != ["matern_cluster", "nested_thomas", "strauss", "thomas"]:
    print(f"preflight FAILED -- unexpected label_names {ln}; config target_label_names must match."); sys.exit(1)
print("preflight OK -- 8 bundles intact, label_names =", ln)
PYEOF

python -u scripts/train.py "$CFG" --seed "$seed" --force
