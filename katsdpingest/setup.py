#!/usr/bin/env python
from setuptools import setup, find_packages


tests_require = ['nose', 'asynctest', 'async_timeout', 'katsdpsigproc[test]']


setup(
    name="katsdpingest",
    description="Karoo Array Telescope Data Capture",
    author="Simon Ratcliffe",
    packages=find_packages(),
    package_data={'': ['ingest_kernels/*.mako']},
    include_package_data=True,
    scripts=[
        "scripts/ingest.py",
        "scripts/ingest_autotune.py"
    ],
    setup_requires=['katversion'],
    install_requires=[
        'aiokatcp',
        'aiomonitor',
        'numpy',
        'spead2>=1.8.0',   # Needed for inproc transport for unit tests
        'katsdpsigproc',
        'katsdpservices',
        'katsdptelstate',
        'katdal'
    ],
    extras_require={
        'test': tests_require
    },
    tests_require=tests_require,
    zip_safe=False,
    use_katversion=True
)
