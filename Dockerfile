FROM sdp-docker-registry.kat.ac.za:5000/docker-base-gpu

MAINTAINER Bruce Merry "bmerry@ska.ac.za"

# Install libhdf5 from source so that the direct I/O VFD can be used.
# 1.8.17 also fixes a resource leak bug (reported as an infinite loop on
# shutdown) that affects the version shipped with Ubuntu 14.04.
# The other flags are a subset of those used by debian.rules (subsetted
# mostly because the flags were default anyway).
#
# Also remove the wheel built for h5py, because it was linked against the
# system version and needs to be recompiled
USER root
RUN cd /tmp && \
    wget http://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.17.tar.bz2 && \
    tar -jxf hdf5-1.8.17.tar.bz2 && \
    cd hdf5-1.8.17 && \
    ./configure --prefix=/usr/local --enable-production --enable-threadsafe \
                --enable-fortran --enable-cxx --enable-direct-vfd \
                --enable-unsupported && \
    make -j4 && \
    make install && \
    ldconfig && \
    rm -rf /tmp/libhdf5-1.8.17 && \
    find /home/kat/.cache/pip/wheels -name 'h5py-*.whl' -exec rm -- '{}' ';'
USER kat

# Install dependencies. We need to set library-dirs so that the new libhdf5
# will be found. And we need to create /tmp/install and
# /tmp/install/katsdpingest as user kat so that we can write to it later.
RUN mkdir -p /tmp/install/katsdpingest
COPY requirements.txt /tmp/install/requirements.txt
RUN cd /tmp/install && \
    /bin/echo -e '[build_ext]\nlibrary-dirs=/usr/local/lib' > setup.cfg && \
    install-requirements.py -d ~/docker-base/base-requirements.txt -d ~/docker-base/gpu-requirements.txt -r /tmp/install/requirements.txt

# Install the current package
COPY . /tmp/install/katsdpingest
RUN cd /tmp/install/katsdpingest && cp ../setup.cfg . && \
    python ./setup.py clean && pip install --no-index .

EXPOSE 2040
EXPOSE 7147/udp
EXPOSE 7148/udp
