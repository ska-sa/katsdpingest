FROM ubuntu:14.04

MAINTAINER Bruce Merry "bmerry@ska.ac.za"

# Suppress debconf warnings
ENV DEBIAN_FRONTEND noninteractive

# Set up access to github private repositories
COPY conf/id_rsa /root/.ssh/
RUN echo "Host *\n\tStrictHostKeyChecking no\n" >> ~/.ssh/config
RUN chmod -R go-rwx ~/.ssh

# The drivers must match the version of the kernel module running on the host
ENV CUDA_RUN_FILE cuda_6.5.19_linux_64.run
ENV CUDA_RUN http://developer.download.nvidia.com/compute/cuda/6_5/rel/installers/cuda_6.5.19_linux_64.run
ENV DRIVER_RUN_FILE NVIDIA-Linux-x86_64-346.35.run
ENV DRIVER_RUN http://uk.download.nvidia.com/XFree86/Linux-x86_64/346.35/NVIDIA-Linux-x86_64-346.35.run

# Install system packages. Python packages are mostly installed here, but
# certain packages are handled by pip:
# - Not available in Ubuntu 14.04 (universe): pyephem, scikits.fitting, pycuda, katcp, ansicolors
# - Ubuntu 14.04 version is too old: six
RUN apt-get -y update && apt-get -y install \
    build-essential software-properties-common wget git-core \
    python python-dev python-pip \
    python-appdirs \
    python-blinker \
    python-decorator \
    python-h5py \
    python-iniparse \
    python-mako \
    python-markupsafe \
    python-mock \
    python-netifaces \
    python-nose \
    python-numpy \
    python-ply \
    python-py \
    python-pytools \
    python-scipy \
    python-twisted \
    python-unittest2 \
    python-zope.interface
RUN wget -q $CUDA_RUN && sh ./$CUDA_RUN_FILE -silent -toolkit && rm -- $CUDA_RUN_FILE
RUN wget -q $DRIVER_RUN && sh ./$DRIVER_RUN_FILE --no-kernel-module --silent --no-network && rm -- $DRIVER_RUN_FILE
ENV PATH="$PATH:/usr/local/cuda/bin"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64"

# Install Python dependencies. Versions are explicitly listed and pinned, so
# that the docker image is reproducible. There were all up-to-date versions
# at the time of writing i.e. there are no currently known reasons not to
# update to newer versions.
RUN pip install --no-deps \
    ansicolors==1.0.2 \
    katcp==0.5.5 \
    pycuda==2014.1 \
    pyephem==3.7.5.3 \
    scikits.fitting==0.5.1 \
    six==1.9.0
COPY requirements.txt /tmp/install/requirements.txt
# Keep only dependent git repositories; everything else is installed explicitly
# by this Dockerfile.
RUN sed -n '/^git/p' /tmp/install/requirements.txt > /tmp/install/requirements-git.txt && \
    pip install --no-deps -r /tmp/install/requirements-git.txt

# Install the current package
COPY . /tmp/install/katsdpingest
WORKDIR /tmp/install/katsdpingest
RUN python ./setup.py clean && python ./setup.py install

# Run ingest as a non-root user
RUN adduser --system ingest
WORKDIR /home/ingest
USER ingest
