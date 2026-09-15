#!/bin/bash

#SBATCH -J fusion_nested_thomas
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 4
#SBATCH --cpus-per-gpu=4
#SBATCH -t 02:00:00
#SBATCH --mem-per-cpu=14G
#SBATCH --gres=gpu:1
#SBATCH --array=0-49
#SBATCH --output=logs/fusion_nested_thomas_%A_%a.out
#SBATCH --error=logs/fusion_nested_thomas_%A_%a.err

# PH + L(r) FUSION SWEEP -- NESTED THOMAS, plain DTM k=5.
#
# Chases the one clearly positive result from the L-reparameterization 2x2:
# feeding L(r)-r into the scalar side-vector took Thomas 0.1300 -> 0.1110
# (p=0.002), tying vihrs (0.1123) on its home ground. This sweeps HOW that L
# information is encoded, because the raw columns are badly conditioned
# (16 samples of one smooth curve; corr-matrix condition number ~2.8e4) and
# that produced bimodal seed collapse on classification.
#
#   arm idx  run-tag        include_lfunc  lfunc_pca  note
#   0        fusion__Loff   0              0          PH only, baseline
#   1        fusion__raw8   8              0          raw collinear columns ("before")
#   2        fusion__pca2   16             2          95.4% of L-curve variance
#   3        fusion__pca3   16             3          98.7%
#   4        fusion__pca4   16             4          99.5%
#
#   5 arms x 10 seeds (9371..9380) -> 50 array tasks
#   task t -> arm = t / 10, seed = SEEDS[t % 10]
#
# Every arm is re-run here under a fusion__ tag rather than cross-referencing
# the earlier dimsplit__/lfunc__ results, so the whole table is one internally
# consistent experiment at a single n. Pair against the vihrs baseline at the
# same 10 seeds -- run slurm/vihrs_topup.sh, which extends vihrs from 5 to 10.
#
# Submit from the repo root:  sbatch slurm/fusion_nested_thomas.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

CFG=configs/runs/nested_thomas/fusion_k5.yaml
SEEDS=(9371 9372 9373 9374 9375 9376 9377 9378 9379 9380)
N_SEEDS=${#SEEDS[@]}

ARM_TAG=(Loff raw8 pca2 pca3 pca4)
ARM_NL=(0 8 16 16 16)      # include_lfunc: number of raw radii sampled
ARM_PCA=(0 0 2 3 4)        # lfunc_pca: components kept (0 = use raw columns)
N_ARMS=${#ARM_TAG[@]}

t="${SLURM_ARRAY_TASK_ID:?run this as an array job (sbatch slurm/fusion_nested_thomas.sh)}"
if (( t >= N_ARMS * N_SEEDS )); then
  echo "task $t >= $((N_ARMS * N_SEEDS)) -- nothing to do (check --array span)"; exit 0
fi
arm=$(( t / N_SEEDS ))
seed="${SEEDS[$(( t % N_SEEDS ))]}"
tag="${ARM_TAG[$arm]}"
nl="${ARM_NL[$arm]}"
pca="${ARM_PCA[$arm]}"

echo "Host: $(hostname)"
echo "Job ${SLURM_JOB_ID:-unset}  task ${t}  ->  arm=${tag}  seed=${seed}  include_lfunc=${nl} lfunc_pca=${pca}"
echo "  config=${CFG}  run-tag=fusion__${tag}"
echo "  Assigned GPU: ${SLURM_JOB_GPUS:-unset}   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python - <<'PYEOF2'
import pickle, sys
from pathlib import Path
req = ["data/nested_thomas/dtm_k5/diagrams.pkl", "data/nested_thomas/dtm_k5/adversarial_diagrams.pkl",
       "data/nested_thomas/clouds.pkl", "data/nested_thomas/adversarial_clouds.pkl"]
npz = ["data/nested_thomas/clouds.lfunc_cache.npz", "data/nested_thomas/adversarial_clouds.lfunc_cache.npz"]
bad = []
for f in req:
    try:
        with open(f, "rb") as fh: pickle.load(fh)
    except FileNotFoundError: bad.append(f"{f}: MISSING")
    except Exception as e:  bad.append(f"{f}: UNREADABLE ({type(e).__name__})")
for f in npz:
    if not Path(f).exists(): bad.append(f"{f}: MISSING (L cache -- run the vihrs baseline once)")
if bad:
    print("preflight FAILED:", *bad, sep="\n  "); sys.exit(1)
print("preflight OK")
PYEOF2

python -u scripts/train.py "$CFG" --seed "$seed" --run-tag "fusion__${tag}" \
    --set "method.params.include_lfunc=${nl}" \
    --set "method.params.lfunc_pca=${pca}"
