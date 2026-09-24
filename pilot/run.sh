#!/bin/bash
# The whole pilot as a chain of CPU jobs; this script only calls sbatch.
#
#   bash pilot/run.sh                          # generate -> merge -> featurize -> assemble -> analyze
#   FROM=assemble bash pilot/run.sh            # start later in the chain (earlier outputs on disk)
#   PILOT_CONFIG=pilot/smoke.yaml bash pilot/run.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p pilot/logs
export PILOT_CONFIG="${PILOT_CONFIG:-pilot/config.yaml}"
STEPS=(generate merge featurize assemble analyze)
FROM="${FROM:-generate}"
dep=""
started=0
for step in "${STEPS[@]}"; do
  [[ "$step" == "$FROM" ]] && started=1
  (( started )) || continue
  id=$(sbatch --parsable --export=ALL ${dep:+--dependency=afterok:$dep} "pilot/$step.sh")
  echo "$step: $id"
  dep="${id%%;*}"
done
