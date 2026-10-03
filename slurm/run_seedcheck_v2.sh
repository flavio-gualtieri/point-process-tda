#!/bin/bash
# Training-seed noise on the fusion - curves contrast: both networks retrained over seeds {1,2,3}.
# configs/seedcheck.yaml, v2 library. 2 models x 3 seeds x (1 classifier + 7 estimators) = 48 GPU units.
#
#   bash slurm/run_seedcheck_v2.sh          # submit the chain
#   DRY=1 bash slurm/run_seedcheck_v2.sh    # print the sbatch calls, submit nothing
#
# Settles whether the introduction's "persistent homology never makes estimates significantly worse" can
# stand. At one training seed it cannot: Matern II is resolved worse in all three subsets (+0.003 on all
# patterns, +0.013 at tau 0.5, +0.006 at 0.9) and Strauss pooled over all patterns (+0.008). Those
# intervals cover test-theta sampling only, and the paper's own caveat puts seed noise near 0.005, which
# straddles the smallest of them. Three seeds give the s.d. that decides it.
#
#   prebuild_<m>   the two input caches; both already exist from the paper run (same specs, same nn
#                  block), so these are near no-ops kept for the case of a cold cache
#   train       -> both prebuilds. 48 units, unthrottled.
#   compare     -> train. `seeds` in the config makes compare score every (model, seed) as model/seed_<s>
#                  and difference them against `baseline: nn_curves`, paired seed by seed.
#
# Writes results_v2/seedcheck/ only: disjoint from results_v2/pipeline/ and results_v2/ablation/.
# Sized from the paper run's own units: the two-filtration fusion units peaked at 21-34G and ran 4-43 min.

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD

export CLOUDFORGER_CONFIGS="$ROOT/configs/v2"
export CLOUDFORGER_DATA="$ROOT/data_v2"
export CLOUDFORGER_RESULTS="$ROOT/results_v2"

source slurm/env.sh
RUN=$(basename "$CLOUDFORGER_RESULTS")
LOGS="logs/${RUN}_seedcheck"
mkdir -p "$LOGS"

idx=$(python - <<'PY'
from cloudforger.paths import read_config
from cloudforger.pipeline.units import units, done
rc = read_config("seedcheck.yaml")
print(",".join(str(i) for i, u in enumerate(units(rc, "gpu")) if not done(rc, *u)))
PY
)
[ -n "$idx" ] || { echo "nothing to train -- stop"; exit 0; }

py="python -u"
submit() {                                         # submit <key> "<dependency ids>" <opts...> -- <cmd...>
  local key=$1 deps=$2 opts=()
  shift 2
  while [ "$1" != "--" ]; do opts+=("$1"); shift; done
  shift
  local dep=()
  [ -n "$deps" ] && dep=("--dependency=afterok:$deps")
  local log="$LOGS/${key}_%j.out"
  [[ " ${opts[*]} " == *" --array="* ]] && log="$LOGS/${key}_%A_%a.out"
  local cmd=(sbatch --parsable -J "${RUN}_sc_${key}" -o "$log" "${dep[@]}" "${opts[@]}" slurm/job.sh "$@")
  if [ -n "${DRY:-}" ]; then echo "${cmd[*]}" >&2; echo "<$key>"
  else "${cmd[@]}"; fi
}

echo "data $CLOUDFORGER_DATA  results $CLOUDFORGER_RESULTS  configs $CLOUDFORGER_CONFIGS"
echo "untrained units: $(($(grep -o ',' <<<"$idx" | wc -l) + 1))"

pb=()
for m in nn_curves fusion_curves_alpha_dtm10; do
  pb+=("$(submit "prebuild_$m" "" $CPU_SBATCH -c 8 --mem 128G -t 2:00:00 -- \
    $py scripts/train.py --config seedcheck.yaml prebuild --model "$m" | tail -1)")
done
tr=$(submit train "$(IFS=:; echo "${pb[*]}")" $GPU_SBATCH -c 8 --mem 64G -t 6:00:00 --array="$idx" -- \
  $py scripts/train.py --config seedcheck.yaml unit --kind gpu --index '{task}' | tail -1)
cp=$(submit compare "$tr" $CPU_SBATCH -c 4 --mem 48G -t 4:00:00 -- \
  $py scripts/compare.py --config seedcheck.yaml | tail -1)
printf 'prebuilds %s  train %s  compare %s\n' "${pb[*]}" "$tr" "$cp"
echo "logs: $LOGS/"
