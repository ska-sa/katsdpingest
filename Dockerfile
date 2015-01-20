FROM ubuntu:14.04

MAINTAINER Bruce Merry "bmerry@ska.ac.za"

# Set up access to github private repositories
COPY conf/id_rsa /root/.ssh/
RUN echo "Host *\n\tStrictHostKeyChecking no\n" >> ~/.ssh/config
RUN chmod -R go-rwx ~/.ssh

# The drivers must match the version of the kernel module running on the host
ENV CUDA_RUN_FILE cuda_6.5.19_linux_64.run
ENV CUDA_RUN http://developer.download.nvidia.com/compute/cuda/6_5/rel/installers/cuda_6.5.19_linux_64.run
ENV DRIVER_RUN_FILE NVIDIA-Linux-x86_64-346.22.run
ENV DRIVER_RUN http://uk.download.nvidia.com/XFree86/Linux-x86_64/346.22/NVIDIA-Linux-x86_64-346.22.run

# Work in a tmpfs, to avoid bloating the image. Note that the contents
# disappear between RUN steps, so each step must completely use the files
# it needs.
WORKDIR /dev
ENV TMPDIR /dev

# Install system packages
RUN apt-get -y update && apt-get -y install \
    build-essential software-properties-common wget git-core \
    python python-dev python-pip \
    libhdf5-serial-dev libblas-dev gfortran liblapack-dev
RUN wget -q $CUDA_RUN && sh ./$CUDA_RUN_FILE -silent -toolkit
RUN wget -q $DRIVER_RUN && sh ./$DRIVER_RUN_FILE --no-kernel-module --silent --no-network
ENV PATH="$PATH:/usr/local/cuda/bin"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64"

# Install Python dependencies
RUN pip install -U pip setuptools
COPY requirements.txt /tmp/install/requirements.txt
# numpy has to be done first, because scikits.fitting does not declare its dependency
# pycuda indirectly depends on a newer version of six than Ubuntu provides
RUN pip install numpy && pip install -r /tmp/install/requirements.txt
RUN pip install 'six>=1.8.0' pycuda

# Install the current package
COPY . /tmp/install/katsdpingest
WORKDIR /tmp/install/katsdpingest
RUN python ./setup.py install

# Run ingest as a non-root user
RUN adduser --system ingest
WORKDIR /home/ingest
ENV TMPDIR /tmp
USER ingest
