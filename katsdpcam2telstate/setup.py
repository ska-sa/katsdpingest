#!/usr/bin/env python
from setuptools import setup


setup(
    name="katsdpingest",
    description="Karoo Array Telescope Data Capture",
    author="Bruce Merry",
    author_email="bmerry@ska.ac.za",
    scripts=['cam2telstate.py'],
    setup_requires=['katversion'],
    install_requires=[
        'numpy',
        'katcp',
        'katsdpservices',
        'katsdptelstate',
        'katportalclient',
        'tornado>=4.0',
        'six'
    ],
    use_katversion=True
)
