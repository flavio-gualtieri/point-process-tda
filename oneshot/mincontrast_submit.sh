#!/bin/bash
# The minimum-contrast baseline end to end; this script only calls sbatch.
#
#   bash oneshot/mincontrast_submit.sh                 # clouds -> fit array -> assemble -> endtoend (set mincontrast)
#   SKIP_CLOUDS=1 bash oneshot/mincontrast_submit.sh   # K-hat already built
#
# Finished fit chunks skip themselves, so resubmitting redoes only what is missing.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p oneshot/logs
export CONFIG="${CONFIG:-oneshot/configs/default.yaml}"
PY="${PY:-/gpfs/scratch/qp252676/globus/envs/cloud-env/bin/python}"
chunks=$($PY -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['mincontrast']['chunks'])")
dep=""
if [[ -z "${SKIP_CLOUDS:-}" ]]; then
  dep=$(sbatch --parsable --export=ALL,STEP=clouds -J mc_clouds oneshot/mincontrast.sh)
  echo "clouds: $dep"
fi
fit=$(sbatch --parsable --export=ALL,STEP=fit -J mc_fit --mem=16G --array="0-$((6 * chunks - 1))" \
      ${dep:+--dependency=afterok:$dep} oneshot/mincontrast.sh)
echo "fit [0-$((6 * chunks - 1))]: $fit"
asm=$(sbatch --parsable --export=ALL,STEP=assemble -J mc_assemble --cpus-per-task=4 \
      --dependency=afterok:$fit oneshot/mincontrast.sh)
echo "assemble: $asm"
e2e=$(sbatch --parsable --export=ALL,SET=mincontrast --dependency=afterok:$asm oneshot/endtoend.sh)
echo "endtoend (set mincontrast): $e2e"
