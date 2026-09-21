# src/cloudforger/vectorization/__init__.py
"""Stage 2: persistence diagrams -> persistence images.

    persistence_images/   diagram -> fixed-size raster (image.py's fit_imager
                            calibrates the box on training diagrams)

An explicit (non-namespace) package so setuptools' find_packages() picks it
up for installation. Nothing is re-exported at this level -- import from
the specific subpackage you need (e.g.
`from cloudforger.vectorization.persistence_images import fit_imager`).
"""
