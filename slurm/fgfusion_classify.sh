#!/bin/bash

#SBATCH -J fgfusion_classify
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 6
#SBATCH --cpus-per-gpu=6
#SBATCH -t 01:30:00
#SBATCH --mem-per-cpu=14G
#SBATCH --gres=gpu:1
#SBATCH --array=0-29
#SBATCH --output=logs/fgfusion_classify_%A_%a.out
#SBATCH --error=logs/fgfusion_classify_%A_%a.err

# THE DECIDING EXPERIMENT for the classification claim:
# does PH add anything ON TOP of the full classical summary set?
#
# WHERE THIS COMES FROM. slurm/summstats_classify.sh established that the
# union of the standard summary functions beats the topological headline:
#
#   L + F + G   0.8838 +/- 0.0036      <- classical union
#   PH (+) L    0.8806 +/- 0.0028      <- previous headline
#   PH only     0.8729 +/- 0.0029
#   L only      0.8371 +/- 0.0044      <- the baseline the paper reports against
#
# paired L+F+G vs PH(+)L: +0.0032, 8/10 seeds, Wilcoxon p ~ 0.04, and the
# adversarial split is dead even (0.8772 vs 0.8769). So "PH beats the radial
# baseline" does not survive a properly specified baseline.
#
# The only version of the claim still standing is that persistence images add
# information the union does NOT already carry. That is this job:
# PH (+) L (+) F (+) G against the L+F+G number above.
#
# HONEST PRIOR: probably not. Adding F and G to L raised matern_cluster recall
# 0.563 -> 0.697, overshooting PH's 0.643 -- and matern-vs-thomas (matched on
# intensity, count and overlap index, differing ONLY in offspring kernel
# shape) is exactly the pair PH's mechanism was supposed to own. F/G and PH
# look like they read the SAME kernel-shape signal, which predicts heavy
# redundancy and PH(+)LFG ~= LFG. Worth an hour of GPU to find out rather
# than arguing about it.
#
#   arm idx  run-tag           include_lfunc  lfunc_pca  include_fgfunc
#   0        fgfus__PHfg       0              0          8    PH + F/G, no L
#   1        fgfus__PHLfg      8              0          8    PH + raw L + F/G
#   2        fgfus__PHLpca2fg  16             2          8    PH + PCA-2 L + F/G
#
# Arm 2 is the natural "PH + the full union": PCA-2 was the L encoding that
# won classification (raw L columns are ill-conditioned and make the CE head
# bimodal), so it is the fair partner for the F/G columns. Arm 1 keeps the raw
# encoding for comparability with the parameter-estimation arms. Arm 0 asks
# whether F/G alone already subsume L's contribution to PH.
#
#   3 arms x 10 seeds (9371..9380) -> 30 array tasks, all independent.
#
# COMPARE AGAINST (all n=10, all at n_epochs=600 / patience=60, same seeds):
#   results/classification/raw/vihrs/_runs/summ__LFG/          0.8838  <- to beat
#   results/classification/dtm_k5/pi_multik/_runs/cap600__pca2/ 0.8806
#   results/classification/dtm_k5/pi_multik/_runs/cap600__Loff/ 0.8729
# Those already exist at this budget, so they are NOT re-run here.
#
# READ THE RESULT AS: arm 2 (or 1) minus 0.8838, paired by seed. A gain
# under ~0.003 is inside the spread of these arms and should be reported as
# "no evidence PH adds to the union", not as a win.
#
# WATCH FOR THE BIMODALITY. The raw-L arms collapse on a fraction of seeds
# (an initialization lottery, NOT overfitting -- see slurm/
# classification_epochcap.sh). At n_epochs=600 raw8 escaped 5/10. Judge these
# arms on the CONVERGED seeds and report the escape rate alongside, exactly
# as the epoch-cap run does; do not average across both modes.
#
# ASSUMES (all on disk; nothing to regenerate, no persistence to recompute):
#   data/classification/{clouds,adversarial_clouds}.pkl
#   data/classification/dtm_k5/{,adversarial_}diagrams.pkl
#   data/classification/{,adversarial_}clouds.lfunc_classify_cache.npz   (L)
#   data/classification/{,adversarial_}clouds.summ_fg0250_classify_cache.npz  (F/G/J)
# the last pair built by slurm/summstats_featurize.sh -- already present.
#
# Resumable: is_done() skips a seed whose results.pt exists.
#   sbatch --array=<comma,ids> slurm/fgfusion_classify.sh
#
# Read out with:
#   for f in results/classification/dtm_k5/pi_multik/_runs/fgfus__*/seed_*/results.json; \
#     do jq -r '[.config.run_tag,.seed,.test_accuracy]|@tsv' $f; done | sort

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

CFG=configs/runs/classification/fusion_k5.yaml
SEEDS=(9371 9372 9373 9374 9375 9376 9377 9378 9379 9380)
N_SEEDS=${#SEEDS[@]}

ARM_TAG=(PHfg  PHLfg  PHLpca2fg)
ARM_NL=(0      8      16)          # include_lfunc
ARM_PCA=(0     0      2)           # lfunc_pca
ARM_NFG=(8     8      8)           # include_fgfunc (N each of F and G -> 2N cols)
N_ARMS=${#ARM_TAG[@]}

# Matched to slurm/classification_epochcap.sh and slurm/summstats_classify.sh
# so every number in the comparison shares one training budget.
N_EPOCHS=600
PATIENCE=60
FG_R_MAX=0.25                      # the grid that won the baseline sweep

t="${SLURM_ARRAY_TASK_ID:?run this as an array job (sbatch slurm/fgfusion_classify.sh)}"
if (( t >= N_ARMS * N_SEEDS )); then
  echo "task $t >= $((N_ARMS * N_SEEDS)) -- nothing to do (check --array span)"; exit 0
fi
arm=$(( t / N_SEEDS ))
seed="${SEEDS[$(( t % N_SEEDS ))]}"
tag="${ARM_TAG[$arm]}"

echo "Host: $(hostname)"
echo "Job ${SLURM_JOB_ID:-unset}  task ${t}  ->  arm=${tag}  seed=${seed}"
echo "  include_lfunc=${ARM_NL[$arm]} lfunc_pca=${ARM_PCA[$arm]} include_fgfunc=${ARM_NFG[$arm]}"
echo "  run-tag=fgfus__${tag}   Assigned GPU: ${SLURM_JOB_GPUS:-unset}"

python - <<'PYEOF'
import sys
from pathlib import Path
import torch  # noqa: F401  -- interpreter/env check
from cloudforger.baselines import summstats  # noqa: F401
req = [
    "data/classification/clouds.pkl",
    "data/classification/adversarial_clouds.pkl",
    "data/classification/dtm_k5/diagrams.pkl",
    "data/classification/dtm_k5/adversarial_diagrams.pkl",
    "data/classification/clouds.lfunc_classify_cache.npz",
    "data/classification/adversarial_clouds.lfunc_classify_cache.npz",
    "data/classification/clouds.summ_fg0250_classify_cache.npz",
    "data/classification/adversarial_clouds.summ_fg0250_classify_cache.npz",
]
bad = [f"{f}: MISSING" for f in req if not Path(f).exists()]
if bad:
    print("preflight FAILED:", *bad,
          "\n  (F/G caches come from slurm/summstats_featurize.sh)", sep="\n  ")
    sys.exit(1)
print(f"preflight OK (torch {torch.__version__}, cuda={torch.cuda.is_available()})")
PYEOF

python -u scripts/train.py "$CFG" --seed "$seed" --run-tag "fgfus__${tag}" \
    --set "method.params.include_lfunc=${ARM_NL[$arm]}" \
    --set "method.params.lfunc_pca=${ARM_PCA[$arm]}" \
    --set "method.params.include_fgfunc=${ARM_NFG[$arm]}" \
    --set "method.params.fg_r_max=${FG_R_MAX}" \
    --set "method.params.n_epochs=${N_EPOCHS}" \
    --set "method.params.early_stopping_patience=${PATIENCE}"
