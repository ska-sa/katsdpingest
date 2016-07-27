#!/usr/bin/env python
from setuptools import setup, find_packages

tests_require = ['mock', 'nose']

setup(
    name="katsdpingest",
    description="Karoo Array Telescope Data Capture",
    author="Simon Ratcliffe",
    packages=find_packages(),
    package_data={'': ['ingest_kernels/*.mako']},
    include_package_data=True,
    scripts=[
        "scripts/ingest.py",
        "scripts/bf_ingest.py",
        "scripts/ingest_autotune.py",
        "scripts/cam2telstate.py"
    ],
    setup_requires=['katversion'],
    install_requires=[
        'h5py',
        'manhole',
        'netifaces',
        'numpy',
        'spead2>=0.10.2',
        'ipaddress',
        'katcp',
        'katsdpsigproc',
        'katsdpdisp',
        'katsdpfilewriter',
        'katsdptelstate',
        'psutil',
        'trollius'
    ],
    extras_require={
        'cam2telstate': ['katportalclient', 'tornado>=4.0', 'six'],
        'test': tests_require
    },
    tests_require=tests_require,
    zip_safe=False,
    use_katversion=True
)
