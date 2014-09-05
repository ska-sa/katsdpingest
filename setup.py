#!/usr/bin/env python
from setuptools import setup, find_packages

setup (
    name = "katsdpingest",
    version = "trunk",
    description = "Karoo Array Telescope Data Capture",
    author = "Simon Ratcliffe",
    packages = find_packages(),
    package_data={'': ['conf/*']},
    include_package_data = True,
    scripts = [
        "scripts/ingest.py",
        "scripts/cbf_data_simulator.py",
        "scripts/cam2spead.py",
        "scripts/sim_observe.py",
        ],
    install_requires = [
        'scikits.fitting',
        'katsdpsigproc'
    ],
    zip_safe = False,
)
