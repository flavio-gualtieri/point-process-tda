"""Array sizes for slurm/run_all.sh, as shell assignments, in ONE light Python start-up (numpy/pandas
imports cost tens of seconds each on a cold GPFS cache). Each count is the matching script's own
`tasks` / `list` (tests/test_pipeline.py checks they agree).

    eval "$(python slurm/plan.py pipeline ablation)"
    # -> simulate featurize mincontrast, and train_cpu_<run> train_gpu_<run> per run config
"""

import sys

from cloudforger.paths import read_config
from cloudforger.pipeline.units import done, units

# baselines.mincontrast.FITTABLE, without its imports: the minimum-contrast array fits the run's families among these
MINCONTRAST_FITTABLE = ("thomas", "nested", "lgcp", "matern2", "ring", "matern1")


def main(runs: list[str]) -> None:
    sim, feat, cfg = read_config("simulation.yaml"), read_config("featurization.yaml"), read_config("pipeline.yaml")
    thetas = len(range(0, sim["thetas"], sim.get("stride", 1)))
    print(f"simulate={len(sim['families']) * -(-thetas // sim['shard_size'])}")
    print(f"featurize={len(sim['families']) * len(feat['filtrations'])}")
    print(f"mincontrast={sum(f in cfg['families'] for f in MINCONTRAST_FITTABLE) * cfg['mincontrast']['chunks']}")
    for run in runs:
        rc = read_config(f"{run}.yaml")
        for kind in ("cpu", "gpu"):
            todo = [str(i) for i, u in enumerate(units(rc, kind)) if not done(rc, *u)]
            print(f"train_{kind}_{run}={','.join(todo)}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["pipeline"])
