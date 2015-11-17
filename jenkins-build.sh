#!/bin/bash
set -e -x
pip install -U pip setuptools wheel
install-requirements.py \
    -d ~/docker-base/base-requirements.txt -d ~/docker-base/gpu-requirements.txt \
    -r requirements.txt -r test-requirements.txt
if [ "$label" = "cuda" ]; then
    export CUDA_DEVICE=0
elif [ "$label" = "opencl" ]; then
    export PYOPENCL_CTX=0:0
fi
nosetests -e simulator --with-xunit --cover-erase --with-coverage --cover-package=katsdpingest --cover-xml
