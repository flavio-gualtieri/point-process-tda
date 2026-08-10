# src/cloudforger/vectorization/__init__.py
"""Stage 2: persistence diagrams -> persistence images + scalar features.

    persistence_images/   diagram -> fixed-size raster (registry-free; see
                            calibrated.py for the auto-calibrated builder)
    scalar_features/        diagram -> scalar/vector summary (registry)

An explicit (non-namespace) package so setuptools' find_packages() picks it
up for installation. Nothing is re-exported at this level -- import from
the specific subpackage you need (e.g.
`from cloudforger.vectorization.scalar_features import REGISTRY`).
"""
