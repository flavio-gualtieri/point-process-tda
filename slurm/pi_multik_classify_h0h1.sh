#!/bin/bash

#SBATCH -J pi_multik_classify_h0h1
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 04:00:00
#SBATCH --mem-per-cpu=14G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --output=logs/pi_multik_classify_h0h1_%A_%a.out
#SBATCH --error=logs/pi_multik_classify_h0h1_%A_%a.err

# PROCESS CLASSIFICATION -- pi_multik with the classification head.
#
#   [ PI | H0+H1 | DTM{5,10,15} | 5-way softmax | 5 seeds ]
#
# One config, one process ("classification"), 5 seeds -> 5 array tasks,
# one GPU each:
#
#   config   configs/runs/classification/pi_multik_h0h1_k5k10k15.yaml
#   task     method.params.task: classify  (ClassificationHead + cross-entropy)
#   classes  aniso_thomas, matern_cluster, nested_thomas, strauss, thomas
#            (lgcp / lgcp_strauss excluded -- see classification_manifest.yaml)
#   seeds    9371..9375
#
# Reference run: NO --run-tag, so it owns the default
#   results/classification/dtm_k5+10+15/pi_multik/seed_<seed>/
# path. Resumable: is_done() skips a seed whose results.pt already exists,
# so a plain re-submit only fills gaps. Rerun specific failures with
#   sbatch --array=<comma,ids> slurm/pi_multik_classify_h0h1.sh
#
# Resources: the classification pool is 5 x 7k = 35k train clouds + 5k
# adversarial -- ~5x the per-process regression runs. 8 CPUs x 14 GB = 112 GB
# covers the 3-k persistence-image tensors (~4 GB f32) plus the in-RAM
# diagram bundles; 4 h is generous for 200 epochs + per-seed PI calibration.
#
# ASSUMES (data still uploading at author time -- verify present first):
#   data/classification/clouds.pkl, adversarial_clouds.pkl
#   data/classification/dtm_k{5,10,15}/{,adversarial_}diagrams.pkl
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/pi_multik_classify_h0h1.sh

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

t="${SLURM_ARRAY_TASK_ID:?run this as an array job (sbatch slurm/pi_multik_classify_h0h1.sh)}"
if (( t >= ${#SEEDS[@]} )); then
  echo "task $t >= ${#SEEDS[@]} -- nothing to do (check --array span)"; exit 0
fi
seed="${SEEDS[$t]}"

echo "Host: $(hostname)"
echo "Job ${SLURM_JOB_ID:-unset}  Array task ${t}  ->  process=classification  seed=${seed}"
echo "  config=${CFG}"
echo "  Assigned GPU: ${SLURM_JOB_GPUS:-unset}   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Preflight: every per-k diagram bundle must exist AND fully unpickle -- the
# upload of data/classification/ was still in flight when this was written,
# and a truncated .pkl otherwise fails deep inside training after burning
# queue time. Missing files are already reported cleanly by load_multik_split;
# this catches the half-written case up front.
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
