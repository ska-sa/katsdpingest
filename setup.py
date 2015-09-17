#!/usr/bin/env python
from setuptools import setup, find_packages

setup (
    name = "katsdpingest",
    version = "trunk",
    description = "Karoo Array Telescope Data Capture",
    author = "Simon Ratcliffe",
    packages = find_packages(),
    package_data={'': ['conf/*', 'ingest_kernels/*.mako']},
    include_package_data = True,
    scripts = [
        "scripts/ingest.py",
        "scripts/ingest_autotune.py",
        "scripts/cbf_data_simulator.py",
        "scripts/cam2spead.py",
        "scripts/sim_observe.py",
        ],
    install_requires = [
        'numpy',
        'scipy',
        'iniparse',
        'blinker',
        'netifaces',
        'scikits.fitting',
        'spead2>=0.2.0',
        'katcp',
        'katpoint',
        'katsdpsigproc',
        'katsdpdisp',
        'katsdptelstate'
    ],
    zip_safe = False,
)
