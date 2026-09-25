#!/bin/bash
# Submit what the config asks for and has not been trained yet; this script only calls sbatch.
#
#   bash oneshot/submit.sh                         # prepare -> every untrained unit (CPU + GPU) -> compare
#   SKIP_PREPARE=1 bash oneshot/submit.sh          # inputs already built
#   KIND=cpu bash oneshot/submit.sh                # only the CPU units (or KIND=gpu)
#   AFTER=<jobid> bash oneshot/submit.sh           # start once another job has succeeded
#   CONFIG=oneshot/configs/<name>.yaml bash oneshot/submit.sh
#
# Array indices are positions in `train.py list --kind K`; only untrained ones are submitted, and a
# task whose unit finished meanwhile skips itself.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p oneshot/logs
export CONFIG="${CONFIG:-oneshot/configs/default.yaml}"
PY="${PY:-/gpfs/scratch/qp252676/globus/envs/cloud-env/bin/python}"
dep="${AFTER:-}"
if [[ -z "${SKIP_PREPARE:-}" ]]; then
  dep=$(sbatch --parsable --export=ALL ${dep:+--dependency=afterok:$dep} oneshot/prepare.sh)
  echo "prepare: $dep"
fi
trained=()
for kind in ${KIND:-cpu gpu}; do
  idx=$($PY oneshot/train.py --config "$CONFIG" list --kind "$kind" --todo | awk '{print $1}' | paste -sd, -)
  if [[ -z "$idx" ]]; then echo "$kind: nothing to train"; continue; fi
  id=$(sbatch --parsable --export=ALL --array="$idx" ${dep:+--dependency=afterok:$dep} "oneshot/train_$kind.sh")
  echo "$kind units [$idx]: $id"
  trained+=("$id")
done
# compare once every unit has ended, whether or not each succeeded (it reports what is missing)
if (( ${#trained[@]} )); then
  cdep="--dependency=afterany:$(IFS=:; echo "${trained[*]}")"
elif [[ -n "$dep" ]]; then
  cdep="--dependency=afterok:$dep"
else
  cdep=""
fi
id=$(sbatch --parsable --export=ALL $cdep oneshot/compare.sh)
echo "compare: $id"
