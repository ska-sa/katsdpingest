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
        "scripts/ingest_autotune.py"
    ],
    setup_requires=['katversion'],
    install_requires=[
        'h5py',
        'manhole',
        'numpy',
        'spead2>=0.6.3',
        'katcp',
        'katsdpsigproc',
        'katsdpdisp',
        'katsdptelstate',
        'psutil',
        'trollius'
    ],
    tests_require=tests_require,
    extras_require={'test': tests_require},
    zip_safe=False,
    use_katversion=True
)
