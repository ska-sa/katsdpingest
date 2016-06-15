#!/usr/bin/env python
from setuptools import setup, find_packages, Extension
import glob
import sys
import ctypes.util

tests_require = ['mock', 'nose']

# Different OSes install the Boost.Python library under different names
bp_library_names = [
    'boost_python-py{0}{1}'.format(sys.version_info.major, sys.version_info.minor),
    'boost_python{0}'.format(sys.version_info.major),
    'boost_python',
    'boost_python-mt']
for name in bp_library_names:
    if ctypes.util.find_library(name):
        bp_library = name
        break
else:
    raise RuntimeError('Cannot find Boost.Python library')

extensions = [
    Extension(
        '_bf_ingest_session',
        sources=(glob.glob('spead2/src/common_*.cpp') +
                 glob.glob('spead2/src/recv_*.cpp') +
                 glob.glob('spead2/src/send_*.cpp') +
                 ['katsdpingest/bf_ingest_session.cpp']),
        depends=glob.glob('spead2/src/*.h'),
        language='c++',
        include_dirs=['spead2/src'],
        define_macros=[('SPEAD2_USE_IBV', '1')],
        extra_compile_args=['-std=c++11', '-g0'],
        libraries=[bp_library, 'rdmacm', 'ibverbs', 'hdf5_cpp', 'hdf5', 'boost_system', 'boost_regex'])
]

setup(
    name="katsdpingest",
    description="Karoo Array Telescope Data Capture",
    author="Simon Ratcliffe",
    packages=find_packages(),
    package_data={'': ['ingest_kernels/*.mako']},
    include_package_data=True,
    ext_package='katsdpingest',
    ext_modules=extensions,
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
    extras_require = {
        'cam2telstate': ['katportalclient', 'tornado>=4.0', 'six'],
        'test': tests_require
    },
    tests_require=tests_require,
    zip_safe=False,
    use_katversion=True
)
