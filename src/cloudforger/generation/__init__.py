# src/cloudforger/generation/__init__.py
"""DV3 synthetic data generation (docs/theory/generation.tex).

    spec.py       reads configs/generation/dv3.yaml
    seeding.py    one addressable random stream per (case, role)
    prior.py      draw n-bar and the shape, apply constraints, invert to model parameters
    samplers.py   model parameters + pattern stream -> points in W = [0,1]^2
    plan.py       enumerate every case of the prior-drawn sets into plan.csv
    store.py      shard files, merged per-(set, family) outputs, checksums
    pipeline.py   run_shard / merge / regen_case, driven by scripts/generation/dv3.py

Nothing is re-exported here; import from the submodule you need.
"""
