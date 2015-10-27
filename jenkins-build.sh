#!/bin/bash
set -e -x
pip install -U pip setuptools wheel
pip install numpy git+ssh://git@github.com/ska-sa/katversion # Some requirements need it already installed
pip install -r requirements.txt
pip install coverage
if [ "$label" = "cuda" ]; then
    pip install pycuda
    export CUDA_DEVICE=0
elif [ "$label" = "opencl" ]; then
    pip install pyopencl
    export PYOPENCL_CTX=0:0
fi
nosetests -e simulator --with-xunit --cover-erase --with-coverage --cover-package=katsdpingest --cover-xml
