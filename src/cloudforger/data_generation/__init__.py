# src/cloudforger/data_generation/__init__.py
"""Stage 1: point clouds -> persistence diagrams / signed measures.

    point_processes/   sampling clouds from a spatial point process (registry)
    filtration/          clouds -> diagrams/signed measures (registry)
    design.py             design-space sampling driving scripts/generate.py

An explicit (non-namespace) package so setuptools' find_packages() picks it
up for installation. Nothing is re-exported at this level -- import from
the specific submodule/subpackage you need (e.g.
`from cloudforger.data_generation.point_processes import REGISTRY`), same
convention as vectorization/.
"""
