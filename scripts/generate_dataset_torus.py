# scripts/generate_dataset_torus.py
import numpy as np
import pickle
from pathlib import Path
from cloudforger.processes.matern import MaternHardCoreProcess
from cloudforger.processes.poisson import PoissonProcess
from cloudforger.processes.thomas import ThomasProcess
from cloudforger.core.region import Torus

CONFIG = {
    "n_clouds_per_class": 500,
    "n_points": 500,
    "region": Torus(R=1.0, r=0.5),
    "seed_base": 202650519,  # arbitrary fixed seed
    "output_dir": Path("data"),
}

PROCESSES = {
    "poisson": PoissonProcess(),
    "thomas": ThomasProcess(parent_intensity=20, mean_offspring=25, cluster_scale=0.04),
    "matern": MaternHardCoreProcess(parent_intensity=1000, hardcore_radius=0.03),
}

def main():
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    clouds, labels, label_names = [], [], list(PROCESSES.keys())

    seed_counter = 0
    for class_idx, (name, proc) in enumerate(PROCESSES.items()):
        for _ in range(CONFIG["n_clouds_per_class"]):
            seed = CONFIG["seed_base"] + seed_counter
            seed_counter += 1
            cloud = proc.sample(
                n=CONFIG["n_points"],
                region=CONFIG["region"],
                seed=seed,
            )
            clouds.append(cloud)
            labels.append(class_idx)

    with open(CONFIG["output_dir"] / "clouds_torus.pkl", "wb") as f:
        pickle.dump({
            "clouds": clouds,
            "labels": np.array(labels),
            "label_names": label_names,
            "config": {k: v for k, v in CONFIG.items() if k != "output_dir"},
        }, f)

    print(f"Saved {len(clouds)} clouds ({len(label_names)} classes) to {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()