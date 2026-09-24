#!/bin/bash
# Submit the cascade as a chain of SLURM jobs (this script only calls sbatch; run it on the login node).
#
#   bash cascade/run.sh                       # regime analysis -> components (hgb + nn) -> assemble
#   FROM=stage1 bash cascade/run.sh           # also redo features + stage 1 first
#   FROM=components bash cascade/run.sh       # cutoffs already exist
#   MODELS=hgb bash cascade/run.sh            # skip the GPU components
#   CONFIG=cascade/configs/<name>.yaml bash cascade/run.sh
#
#   stage1.sh           CPU    features + stage 1 (+ out-of-fold train predictions)
#   regime_analysis.sh  CPU    coordinates scored, cutoffs.json written
#   components.sh       CPU    array over regime.taus: stage 2 + stage 3, hgb
#   components_nn.sh    GPU    array over regime.taus: stage 2 + stage 3, nn
#   assemble.sh         CPU    components put together per tau, sweep summary

set -euo pipefail
cd "$(dirname "$0")/.."
export CONFIG="${CONFIG:-cascade/configs/default.yaml}"
FROM="${FROM:-regime}"
MODELS="${MODELS:-hgb nn}"

dep=""
after() { [[ -n "$1" ]] && echo "--dependency=afterok:$1" || true; }

if [[ $FROM == stage1 ]]; then
    dep=$(sbatch --parsable cascade/stage1.sh); echo "stage1          $dep"
fi
if [[ $FROM == stage1 || $FROM == regime ]]; then
    dep=$(sbatch --parsable $(after "$dep") cascade/regime_analysis.sh); echo "regime_analysis $dep"
fi
comp=()
for m in $MODELS; do
    script=cascade/components.sh; [[ $m == nn ]] && script=cascade/components_nn.sh
    j=$(sbatch --parsable $(after "$dep") "$script"); echo "components $m   $j"
    comp+=("$j")
done
all=$(IFS=:; echo "${comp[*]}")
j=$(sbatch --parsable --dependency=afterany:"$all" cascade/assemble.sh); echo "assemble        $j"
