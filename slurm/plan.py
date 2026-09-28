"""Array sizes for slurm/run_all.sh, as shell assignments, in ONE light Python start-up (numpy/pandas
imports cost tens of seconds each on a cold GPFS cache). Each count is the matching script's own
`tasks` (tests/test_pipeline.py checks they agree).

    eval "$(python slurm/plan.py)"      # -> simulate featurize mincontrast train_cpu train_gpu
"""

from cloudforger.paths import read_config
from cloudforger.pipeline.units import done, units

MINCONTRAST_MODELS = 6          # baselines.mincontrast.MODELS: every family with a closed-form K but poisson and cell


def main() -> None:
    sim, feat, cfg = read_config("simulation.yaml"), read_config("featurization.yaml"), read_config("pipeline.yaml")
    thetas = len(range(0, sim["thetas"], sim.get("stride", 1)))
    print(f"simulate={len(sim['families']) * -(-thetas // sim['shard_size'])}")
    print(f"featurize={len(sim['families']) * len(feat['filtrations'])}")
    print(f"mincontrast={MINCONTRAST_MODELS * cfg['mincontrast']['chunks']}")
    for kind in ("cpu", "gpu"):
        todo = [str(i) for i, u in enumerate(units(cfg, kind)) if not done(cfg, *u)]
        print(f"train_{kind}={','.join(todo)}")


if __name__ == "__main__":
    main()
