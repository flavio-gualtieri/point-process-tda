# scripts/run_sweep.py

import yaml
from scripts.runners.classify.runner_classify import run

with open("configs/classify/raw_pc.yaml") as f:
    cfg = yaml.safe_load(f)   # gives you a plain Python dict — same as your CONFIG

# Expand the sweep axis


for dim in cfg["ambient_dims"]:
    concrete = cfg.copy()
    concrete["method"] = "raw_pc"
    concrete["ambient_dim"] = dim
    concrete["dataset_path"] = cfg["dataset_path_template"].format(ambient_dim=dim)
    concrete["output_dir"] = cfg["output_dir_template"].format(ambient_dim=dim)
    run(concrete)   # calls runner.py with a fully resolved dict