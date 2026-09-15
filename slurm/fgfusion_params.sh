#!/bin/bash

#SBATCH -J fgfusion_params
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 6
#SBATCH --cpus-per-gpu=6
#SBATCH -t 01:00:00
#SBATCH --mem-per-cpu=14G
#SBATCH --gres=gpu:1
#SBATCH --array=0-149
#SBATCH --output=logs/fgfusion_params_%A_%a.out
#SBATCH --error=logs/fgfusion_params_%A_%a.err

# STAGE 2 of 2 -- does the F/G story hold for PARAMETER ESTIMATION?
#
# On 4-way classification the union of the standard summary functions beat the
# topological headline (slurm/summstats_classify.sh: L+F+G 0.8838 vs PH (+) L
# 0.8806, 8/10 seeds, p ~ 0.04). Parameter estimation is a different question
# and the prior points the other way: the vihrs baseline there consumes L(r)
# on a 513-point grid through a 1-D CNN, which is already close to a
# sufficient statistic for a KNOWN Neyman-Scott family -- F and G may add
# little where L is nearly sufficient, even though they added a lot where the
# task was telling families apart. This job settles it on both sides:
#
#   * does L+F+G beat L alone for parameter estimation?  (arms vL vs vLFG)
#   * does PH add anything on top of L+F+G?              (arms PHLfg vs vLFG)
#
#   arm idx  run-tag       method      feature set
#   0        fgp__vL       vihrs       L                       (baseline control)
#   1        fgp__vLFG     vihrs       L + F + G               <- classical union
#   2        fgp__Loff     pi_multik   PH only                 (control)
#   3        fgp__PHLfg    pi_multik   PH + 8 L cols + 8 F/G   <- THE experiment
#   4        fgp__PHfg     pi_multik   PH + 8 F/G cols, no L
#
#   3 processes x 5 arms x 10 seeds -> 150 array tasks, all independent.
#   task -> process = t / 50, arm = (t % 50) / 10, seed = SEEDS[t % 10]
#
# EVERY ARM IS RE-RUN UNDER A fgp__ TAG, including the two controls that
# already exist elsewhere. That costs 60 redundant tasks (~2 GPU-h) and buys
# one internally consistent table at a single budget on a single filtration --
# the discipline TODAY.MD section 8 asks for, since the existing Loff/raw8
# numbers are k5-only fusion-sweep runs and the existing vihrs numbers are
# from a different job. Do NOT mix these against the writeup's fused-k table.
#
# PROCESSES. thomas is where the fusion parity result lives (PH (+) L 0.1110
# vs vihrs 0.1120, p=0.922). matern_cluster is the scientifically pointed one:
# its design is matched to thomas on K/EN/c with c1 reducing to exactly c in
# both, so the two families differ ONLY in offspring kernel shape, and F/G
# read kernel shape well (they lifted matern classification recall
# 0.563 -> 0.697). nested_thomas is the negative control -- PH already ties
# vihrs there (0.2091 both) and L fusion added nothing (p=0.160), so F/G
# should add nothing either. A mechanism that predicts its own null is worth
# more than one that only predicts wins.
#
# NOT INCLUDED: strauss (pi_multik training there is a bimodal initialization
# lottery -- fix it with slurm/strauss_restarts.sh first, or any fusion arm is
# uninterpretable) and aniso_thomas (cluster_theta is at chance for every
# method, so 44% of its loss is noise).
#
# REFERENCE NUMBERS (n=10, k5, existing runs -- for orientation only; use the
# fgp__ controls for the actual paired tests):
#   thomas          PH only 0.1309   PH (+) L raw8 0.1110   vihrs 0.1120
#   nested_thomas   PH only 0.2091   PH (+) L raw8 0.2180   vihrs 0.2091
#   matern_cluster  (fused-k, n=5)   PH 0.1406              vihrs 0.1242
#
# REQUIRES STAGE 1 (slurm/summstats_featurize_params.sh); the preflight below
# hard-fails on a missing F/G cache rather than letting 150 GPU tasks each
# rebuild it. Chain them:
#   fid=$(sbatch --parsable slurm/summstats_featurize_params.sh)
#   sbatch --dependency=afterok:$fid slurm/fgfusion_params.sh
#
# Resumable: is_done() skips a seed whose results.pt exists.
#   sbatch --array=<comma,ids> slurm/fgfusion_params.sh
#
# Read out with:
#   for f in results/*/dtm_k5/pi_multik/_runs/fgp__*/seed_*/results.json \
#            results/*/raw/vihrs*/_runs/fgp__*/seed_*/results.json; do \
#     jq -r '[.config.run_tag,.seed,.test_loss]|@tsv' $f; done | sort

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

PROCESSES=(thomas matern_cluster nested_thomas)
VIHRS_CFG=(
  configs/runs/thomas/thomas_vihrs.yaml
  configs/runs/matern_cluster/matern_cluster_vihrs.yaml
  configs/runs/nested_thomas/nested_thomas_vihrs.yaml
)
PI_CFG=(
  configs/runs/thomas/fusion_k5.yaml
  configs/runs/matern_cluster/fusion_k5.yaml
  configs/runs/nested_thomas/fusion_k5.yaml
)

SEEDS=(9371 9372 9373 9374 9375 9376 9377 9378 9379 9380)
N_SEEDS=${#SEEDS[@]}

ARM_TAG=(vL     vLFG      Loff  PHLfg  PHfg)
ARM_KIND=(vihrs vihrs     pi    pi     pi)
ARM_CH=("[L]"  "[L,F,G]"  -     -      -)     # vihrs arms only
ARM_NL=(-      -          0     8      0)     # pi arms: include_lfunc
ARM_NFG=(-     -          0     8      8)     # pi arms: include_fgfunc
N_ARMS=${#ARM_TAG[@]}

FG_R_MAX=0.25
PER_PROC=$(( N_ARMS * N_SEEDS ))              # 50
NTASKS=$(( ${#PROCESSES[@]} * PER_PROC ))     # 150 -- must equal the --array span

t="${SLURM_ARRAY_TASK_ID:?run this as an array job (sbatch slurm/fgfusion_params.sh)}"
if (( t >= NTASKS )); then
  echo "task $t >= $NTASKS -- nothing to do (check --array span)"; exit 0
fi
p=$(( t / PER_PROC ))
rem=$(( t % PER_PROC ))
arm=$(( rem / N_SEEDS ))
seed="${SEEDS[$(( rem % N_SEEDS ))]}"
proc="${PROCESSES[$p]}"
tag="${ARM_TAG[$arm]}"
kind="${ARM_KIND[$arm]}"

echo "Host: $(hostname)"
echo "Job ${SLURM_JOB_ID:-unset}  task ${t}  ->  process=${proc}  arm=${tag} (${kind})  seed=${seed}"
echo "  run-tag=fgp__${tag}   Assigned GPU: ${SLURM_JOB_GPUS:-unset}"

# Preflight: dead interpreter, missing clouds/diagrams, and -- for every arm
# that actually uses F/G -- the STAGE 1 cache. The L-only arms (vL, Loff) are
# exempt: they never touch it.
PROC_ENV="$proc" TAG_ENV="$tag" python - <<'PYEOF'
import os, sys
from pathlib import Path
import torch  # noqa: F401
from cloudforger.baselines import summstats  # noqa: F401

proc, tag = os.environ["PROC_ENV"], os.environ["TAG_ENV"]
req = [f"data/{proc}/clouds.pkl", f"data/{proc}/adversarial_clouds.pkl"]
if tag in ("Loff", "PHLfg", "PHfg"):
    req += [f"data/{proc}/dtm_k5/diagrams.pkl",
            f"data/{proc}/dtm_k5/adversarial_diagrams.pkl"]
if tag in ("PHLfg",):
    req += [f"data/{proc}/clouds.lfunc_cache.npz",
            f"data/{proc}/adversarial_clouds.lfunc_cache.npz"]
if tag in ("vLFG", "PHLfg", "PHfg"):
    req += [f"data/{proc}/clouds.summ_fg0250_cache.npz",
            f"data/{proc}/adversarial_clouds.summ_fg0250_cache.npz"]

bad = [f"{f}: MISSING" for f in req if not Path(f).exists()]
if bad:
    print("preflight FAILED:", *bad,
          "\n  (summ_fg* caches come from slurm/summstats_featurize_params.sh)", sep="\n  ")
    sys.exit(1)
print(f"preflight OK (torch {torch.__version__}, cuda={torch.cuda.is_available()})")
PYEOF

if [[ "$kind" == "vihrs" ]]; then
  # skip_mincontrast: the bonus minimum-contrast comparison is a Thomas-only
  # side product of run_one_seed and is irrelevant to this feature ablation.
  python -u scripts/train.py "${VIHRS_CFG[$p]}" --seed "$seed" --run-tag "fgp__${tag}" \
      --set "method.params.summary_channels=${ARM_CH[$arm]}" \
      --set "method.params.fg_r_max=${FG_R_MAX}" \
      --set "method.params.skip_mincontrast=true"
else
  python -u scripts/train.py "${PI_CFG[$p]}" --seed "$seed" --run-tag "fgp__${tag}" \
      --set "method.params.include_lfunc=${ARM_NL[$arm]}" \
      --set "method.params.include_fgfunc=${ARM_NFG[$arm]}" \
      --set "method.params.fg_r_max=${FG_R_MAX}"
fi
