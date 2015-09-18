FROM sdp-ingest5.kat.ac.za:5000/docker-base

MAINTAINER Bruce Merry "bmerry@ska.ac.za"

# The drivers must match the version of the kernel module running on the host
ENV CUDA_RUN_FILE cuda_6.5.19_linux_64.run
ENV CUDA_RUN http://mirror.kat.ac.za/nvidia/cuda_6.5.19_linux_64.run
ENV DRIVER_RUN_FILE NVIDIA-Linux-x86_64-346.35.run
ENV DRIVER_RUN http://mirror.kat.ac.za/nvidia/NVIDIA-Linux-x86_64-346.35.run
ENV MIRROR_IP 192.168.1.25

# Install system packages. Python packages are mostly installed here, but
# certain packages are handled by pip because they're not available.
RUN apt-get -y update && apt-get -y install \
    python-appdirs \
    python-blinker \
    python-concurrent.futures \
    python-decorator \
    python-h5py \
    python-iniparse \
    python-mako \
    python-markupsafe \
    python-py \
    python-pytools \
    python-scipy \
    libboost-python1.55-dev \
    libboost-system1.55-dev
RUN echo "$MIRROR_IP mirror.kat.ac.za" >> /etc/hosts && wget -q $CUDA_RUN && sh ./$CUDA_RUN_FILE -silent -toolkit && rm -- $CUDA_RUN_FILE
RUN echo "$MIRROR_IP mirror.kat.ac.za" >> /etc/hosts && wget -q $DRIVER_RUN && sh ./$DRIVER_RUN_FILE --no-kernel-module --silent --no-network && rm -- $DRIVER_RUN_FILE
ENV PATH="$PATH:/usr/local/cuda/bin"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64"

# Install Python dependencies. Versions are explicitly listed and pinned, so
# that the docker image is reproducible. There were all up-to-date versions
# at the time of writing i.e. there are no currently known reasons not to
# update to newer versions.
RUN pip install --no-deps \
    ansicolors==1.0.2 \
    katcp==0.5.5 \
    pycuda==2015.1.3 \
    scikits.fitting==0.5.1 \
    manhole==1.0.0 \
    six==1.9.0 \
    spead2==0.3.0 \
    git+ssh://git@github.com/ska-sa/katpoint \
    git+ssh://git@github.com/ska-sa/katsdpsigproc \
    git+ssh://git@github.com/ska-sa/katsdpdisp \
    git+ssh://git@github.com/ska-sa/katsdptelstate

# Install the current package
COPY . /tmp/install/katsdpingest
WORKDIR /tmp/install/katsdpingest
RUN python ./setup.py clean && pip install --no-deps .

# Run ingest as a non-root user
USER kat
WORKDIR /home/kat

EXPOSE 2040
EXPOSE 7147/udp
EXPOSE 7148/udp
