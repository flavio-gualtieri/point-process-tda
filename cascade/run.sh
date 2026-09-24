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
#   components_nn.sh    GPU    array over (nn model, tau): stage 2 + stage 3
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
    if [[ $m == nn ]]; then
        # one GPU task per (nn model, tau) pair; reading the config is all this does
        n=$(python3 -c "import yaml,sys; c=yaml.safe_load(open('$CONFIG')); ms=dict.fromkeys(c['stage2']['models']+c['stage3']['models']); print(sum(c['models'][m]['kind']=='nn' for m in ms)*len(c['regime']['taus']))")
        j=$(sbatch --parsable --array=0-$((n - 1)) $(after "$dep") cascade/components_nn.sh)
    else
        j=$(sbatch --parsable $(after "$dep") cascade/components.sh)
    fi
    echo "components $m   $j"
    comp+=("$j")
done
all=$(IFS=:; echo "${comp[*]}")
j=$(sbatch --parsable --dependency=afterany:"$all" cascade/assemble.sh); echo "assemble        $j"
