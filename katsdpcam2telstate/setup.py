#!/usr/bin/env python
from setuptools import setup, find_packages


setup(
    name="katsdpcam2telstate",
    description="Karoo Array Telescope Data Capture",
    author="Bruce Merry",
    author_email="bmerry@ska.ac.za",
    packages=find_packages(),
    scripts=['scripts/cam2telstate.py'],
    setup_requires=['katversion'],
    install_requires=[
        'numpy',
        'aiokatcp',
        'katsdpservices',
        'katsdptelstate',
        'katportalclient',
        # Tornado is not used directly, but katportalclient uses it and we need
        # 5.0+ to get seamless asyncio integration.
        'tornado>=5.0',
    ],
    python_requires='>=3.6',
    use_katversion=True
)
