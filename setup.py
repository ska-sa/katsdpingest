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
        "scripts/cbf_simulator.py",
        "scripts/katcp2spead.py",
        ],
    install_requires = [
        'scikits.fitting',
    ],
    zip_safe = False,
)
