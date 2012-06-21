#!/usr/bin/env python
from setuptools import setup, find_packages

setup (
    name = "katcapture",
    version = "trunk",
    description = "Karoo Array Telescope Data Capture",
    author = "Simon Ratcliffe",
    packages = find_packages(),
    package_data={'': ['conf/*']},
    include_package_data = True,
    scripts = [
        "scripts/k7_augment.py",
        "scripts/k7_capture.py",
        "scripts/k7_simulator.py",
        ],
    install_requires = [
        'scikits.fitting',
        'pysolr',
    ],
    zip_safe = False,
)
