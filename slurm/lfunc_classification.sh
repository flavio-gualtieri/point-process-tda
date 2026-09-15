#!/bin/bash

#SBATCH -J lfunc_classification
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 6
#SBATCH --cpus-per-gpu=6
#SBATCH -t 03:00:00
#SBATCH --mem-per-cpu=14G
#SBATCH --gres=gpu:1
#SBATCH --array=0-14
#SBATCH --output=logs/lfunc_classification_%A_%a.out
#SBATCH --error=logs/lfunc_classification_%A_%a.err

# L-REPARAMETERIZATION 2x2 -- 4-way PROCESS-FAMILY CLASSIFICATION, DTM k=5.
# See slurm/lfunc_thomas.sh for the full rationale.
#
#   arm  filtration  include_lfunc  run-tag              config
#   A    dtm_k5      0              dimsplit__shared_e64  ALREADY RUN -- do not resubmit
#                                                         (acc = 0.8729 +/- 0.0023, n=5)
#   0/B  l_dtm_k5    0              lfunc__B_ldtm         lfunc_k5.yaml
#   1/C  dtm_k5      8              lfunc__C_dtm_lx       dimsplit_k5.yaml
#   2/D  l_dtm_k5    8              lfunc__D_ldtm_lx      lfunc_k5.yaml
#
# include_log_n is false in all four arms, so the ONLY difference between C/D
# and A/B is the 8 L(r)-r columns -- the point count must not confound it.
# Arms C and D are therefore no longer "topology-only"; that is deliberate.
#
# READ D vs C, NOT A vs B. Classification is also where PH's margin over vihrs
# is largest (0.877 vs 0.840 at k5/10/15), so this is the arm where an
# L-normalized diagram has the most to lose as well as gain.
#
#   3 arms x 5 seeds (9371..9375) -> 15 array tasks
#   task t -> arm = t / 5, seed = SEEDS[t % 5]
#
# n=5 is a screen, not a significance test (Wilcoxon cannot reach p<0.05 at
# n=5); read effect size and sign consistency, and extend the surviving arm to
# n>=8 before claiming anything.
#
# PREREQUISITE: sbatch slurm/lfunc_transform.sh
# Submit from the repo root:  sbatch slurm/lfunc_classification.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

SEEDS=(9371 9372 9373 9374 9375)
N_SEEDS=${#SEEDS[@]}

ARM_TAG=(B_ldtm C_dtm_lx D_ldtm_lx)
ARM_CFG=(configs/runs/classification/lfunc_k5.yaml configs/runs/classification/dimsplit_k5.yaml configs/runs/classification/lfunc_k5.yaml)
ARM_LFUNC=(0 8 8)
N_ARMS=${#ARM_TAG[@]}

t="${SLURM_ARRAY_TASK_ID:?run this as an array job (sbatch slurm/lfunc_classification.sh)}"
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
# classification caches L under the *_classify_cache.npz name (baselines/vihrs.py)
req = [f"data/classification/{tag}/diagrams.pkl", f"data/classification/{tag}/adversarial_diagrams.pkl",
       "data/classification/clouds.pkl", "data/classification/adversarial_clouds.pkl",
       "data/classification/clouds.lfunc_classify_cache.npz",
       "data/classification/adversarial_clouds.lfunc_classify_cache.npz"]
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
