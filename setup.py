#!/usr/bin/env python
from setuptools import setup, find_packages


tests_require = ['nose', 'asynctest', 'async_timeout', 'katsdpsigproc[test]']


setup(
    name="katsdpingest",
    description="Karoo Array Telescope Data Capture",
    author="MeerKAT SDP Team",
    author_email="sdpdev+katsdpingest@ska.ac.za",
    packages=find_packages(),
    package_data={'': ['ingest_kernels/*.mako']},
    include_package_data=True,
    scripts=[
        "scripts/ingest.py",
        "scripts/ingest_autotune.py"
    ],
    setup_requires=['katversion'],
    install_requires=[
        'aiokatcp>=0.7.0',   # Need 0.7 for auto_strategy
        'aiomonitor',
        'numpy>=1.13.0',     # For np.unique with axis (might really need a higher version)
        'spead2',
        'katsdpsigproc',
        'katsdpservices[argparse,aiomonitor]',
        'katsdptelstate[aio]',
        'katpoint',
        'katdal',
        'katsdpmodels[aiohttp]'
    ],
    extras_require={
        'test': tests_require
    },
    tests_require=tests_require,
    zip_safe=False,
    use_katversion=True
)
