#!/usr/bin/env python
from setuptools import setup, find_packages, Extension
from distutils.command.build_ext import build_ext
import glob
import sys
import subprocess
import os.path
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


# Hack: this is copied and edited from spead2, so that we can run configure
# inside the spead2 submodule.
class BuildExt(build_ext):
    def run(self):
        self.mkpath(self.build_temp)
        subprocess.check_call(['./bootstrap.sh'], cwd='spead2')
        subprocess.check_call(os.path.abspath('spead2/configure'), cwd=self.build_temp)
        # Ugly hack to add libraries conditional on configure result
        have_ibv = False
        with open(os.path.join(self.build_temp, 'include', 'spead2', 'common_features.h')) as f:
            for line in f:
                if line.strip() == '#define SPEAD2_USE_IBV 1':
                    have_ibv = True
        for extension in self.extensions:
            if have_ibv:
                extension.libraries.extend(['rdmacm', 'ibverbs'])
            extension.include_dirs.insert(0, os.path.join(self.build_temp, 'include'))
        # distutils uses old-style classes, so no super
        build_ext.run(self)


extensions = [
    Extension(
        '_bf_ingest_session',
        sources=(glob.glob('spead2/src/common_*.cpp') +
                 glob.glob('spead2/src/recv_*.cpp') +
                 glob.glob('spead2/src/send_*.cpp') +
                 ['spead2/src/py_common.cpp', 'katsdpingest/bf_ingest_session.cpp']),
        depends=glob.glob('spead2/src/*.h'),
        language='c++',
        include_dirs=['spead2/include'],
        extra_compile_args=['-std=c++11', '-g0'],
        libraries=[bp_library, 'hdf5_cpp', 'hdf5', 'boost_system', 'boost_regex'])
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
    cmdclass={'build_ext': BuildExt},
    scripts=[
        "scripts/ingest.py",
        "scripts/bf_ingest.py",
        "scripts/ingest_autotune.py",
        "scripts/cam2telstate.py",
        "scripts/cam2spead_recv.py"
    ],
    setup_requires=['katversion'],
    install_requires=[
        'h5py',
        'futures',
        'manhole',
        'netifaces',
        'numpy',
        'spead2>=1.1.0',
        'ipaddress',
        'katcp',
        'katsdpsigproc',
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
