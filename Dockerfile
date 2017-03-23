FROM sdp-docker-registry.kat.ac.za:5000/docker-base-gpu

MAINTAINER Christopher Schollar "cschollar@ska.ac.za"

# Install dependencies.
COPY requirements.txt /tmp/install/requirements.txt
RUN install-requirements.py -d ~/docker-base/base-requirements.txt -d ~/docker-base/gpu-requirements.txt -r /tmp/install/requirements.txt

USER root

EXPOSE 2040
EXPOSE 7147/udp
EXPOSE 7148/udp

# Suppress debconf warnings
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get install -y autotools-dev \
                      autoconf \
                      libfftw3-dev \
                      libcfitsio3-dev \
                      cvs \
                      libtool \
                      lsof \
                      csh \
                      numactl \
                      libx11-dev \
                      x11-common \
                      libgsl0-dev \
                      libxml2-dev \
                      hwloc \
                      libhwloc-dev \
                      libboost-program-options1.58.0 \
                      libboost-program-options1.58-dev \
                      pgplot5
RUN mkdir /usr/local/kat
RUN chown kat:kat /usr/local/kat

RUN mkdir /usr/local/kat/pulsar

ENV PSRHOME /usr/local/kat/pulsar
ENV LOGIN_ARCH linux_64
ENV PACKAGES $PSRHOME/$LOGIN_ARCH
ENV CFLAGS "-mtune=native -O3 -ffast-math -pthread -fPIC"
ENV CXXFLAGS "-mtune=native -O3 -ffast-math -pthread"
ENV CUDA_NVCC_FLAGS "-O3 -arch sm_30 -m64 -lineinfo"

# psrcat
RUN echo $HOME
ENV HOME /home/kat
WORKDIR $HOME
ENV PSRCAT_FILE $PSRHOME/$LOGIN_ARCH/psrcat/psrcat.db
ENV PATH $PATH:$PSRHOME/psrcat_tar
RUN wget http://www.atnf.csiro.au/people/pulsar/psrcat/downloads/psrcat_pkg.tar.gz
RUN tar -xvf psrcat_pkg.tar.gz -C $PSRHOME
WORKDIR $PSRHOME/psrcat_tar
RUN /bin/bash makeit
RUN mkdir -p $PSRHOME/$LOGIN_ARCH/bin
RUN cp psrcat $PSRHOME/$LOGIN_ARCH/bin/
RUN mkdir -p $PSRHOME/$LOGIN_ARCH/psrcat
RUN cp psrcat.db $PSRHOME/$LOGIN_ARCH/psrcat/
WORKDIR $HOME
COPY ./beamformer_docker_software/obs99.db .
RUN cp obs99.db $PSRHOME/$LOGIN_ARCH/psrcat/

# Tempo2
ENV TEMPO2 $PSRHOME/$LOGIN_ARCH/tempo2
ENV PATH $PATH:$TEMPO2/bin
ENV C_INCLUDE_PATH $C_INCLUDE_PATH:/usr/local/src/tempo2/T2runtime/include
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/src/tempo2/T2runtime/lib
WORKDIR $PSRHOME
#RUN mkdir -p $PSRHOME/tempo2
#WORKDIR $PSRHOME/tempo2
#COPY ./beamformer_docker_software/tempo2 .
RUN git clone https://bitbucket.org/psrsoft/tempo2
WORKDIR $PSRHOME/tempo2
RUN sync && perl -pi -e 's/chmod \+x/#chmod +x/' bootstrap # Get rid of: returned a non-zero code: 126.
RUN ./bootstrap
RUN ./configure --x-libraries=/usr/lib/x86_64-linux-gnu --with-cfitsio-lib-dir=$PSRHOME/cfitsio/install/lib --enable-shared --enable-static --with-pic F77=gfortran CPPFLAGS="-I/home/kat/pulsar_software/calceph-2.2.4/install/include"
RUN make clean
RUN make -j 16
RUN make install
RUN make plugins -j 16
RUN make plugins-install
RUN mkdir -p $TEMPO2
RUN rsync -at T2runtime/* $TEMPO2/

###############################
# PSRCHIVE
WORKDIR $PSRHOME
RUN git clone https://git.code.sf.net/p/psrchive/code psrchive 
WORKDIR $PSRHOME/psrchive

ENV PSRCHIVE $PSRHOME/psrchive
ENV PATH $PATH:$PSRHOME/$LOGIN_ARCH/bin
ENV PGPLOT_DIR $PSRHOME/pgplot
ENV PGPLOT_FONT $PGPLOT_DIR/grfont.dat
RUN ./bootstrap
RUN ./configure --prefix=$PSRHOME/$LOGIN_ARCH
RUN make clean
RUN make -j 16
RUN make libs
RUN make install

###############################
# CUB library
RUN mkdir $HOME/opt
RUN mkdir $HOME/opt/cub-1.3.2
WORKDIR $HOME/opt/cub-1.3.2
COPY ./beamformer_docker_software/cub-1.3.2 .

RUN chown -R kat:kat .
RUN chmod -R a+r .

###############################
# PSRDADA build 
RUN touch $HOME/.cvspass
#RUN mkdir $PSRHOME/psrdada
WORKDIR $PSRHOME
#COPY ./beamformer_docker_software/psrdada .
RUN echo "yolo"
RUN cvs -d:pserver:anonymous@psrdada.cvs.sourceforge.net:/cvsroot/psrdada login
RUN cvs -z3 -d:pserver:anonymous@psrdada.cvs.sourceforge.net:/cvsroot/psrdada co -P psrdada

RUN chown -R kat .
RUN chown -R kat $PSRHOME/$LOGIN_ARCH

ENV PSRHOME /usr/local/kat/pulsar
ENV HOME /home/kat
ENV PSRCAT_FILE $PSRHOME/psrcat_tar/psrcat.db
ENV PATH $PATH:$PSRHOME/psrcat_tar
ENV TEMPO2 $PSRHOME/tempo2/T2runtime
ENV PATH $PATH:$PSRHOME/tempo2/T2runtime/bin
ENV C_INCLUDE_PATH $C_INCLUDE_PATH:/usr/local/src/tempo2/T2runtime/include:/usr/local/cuda-8.0/include
ENV LDFLAGS -lstdc++
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/src/tempo2/T2runtime/lib:/usr/local/cuda-8.0/include
WORKDIR $PSRHOME/psrdada
RUN ./bootstrap
RUN ./configure --prefix=$PSRHOME/$LOGIN_ARCH --with-hwloc-dir=/usr
RUN make clean
RUN make -j 16
RUN make install

###############################
# SPEAD2
#WORKDIR $PSRHOME
#RUN git clone https://github.com/ska-sa/spead2

#WORKDIR $PSRHOME/spead2
#RUN ./bootstrap.sh
#RUN ./configure
#WORKDIR $PSRHOME/spead2/src 
#RUN make -j 16
#RUN cp ./libspead2.a $PSRHOME/$LOGIN_ARCH/lib/
#RUN mkdir $PSRHOME/$LOGIN_ARCH/include/spead2
#RUN ls ../include/spead2/*.h
#RUN cp ../include/spead2/*.h $PSRHOME/$LOGIN_ARCH/include/spead2/
#RUN cp spead2_bench spead2_recv $PSRHOME/$LOGIN_ARCH/bin/

RUN mkdir $PSRHOME/spead2
WORKDIR $PSRHOME/spead2
COPY ./beamformer_docker_software/spead2 .

WORKDIR $PSRHOME/spead2/src
RUN make -j 16
RUN cp ./libspead2.a $PSRHOME/$LOGIN_ARCH/lib/
RUN mkdir $PSRHOME/$LOGIN_ARCH/include/spead2
RUN cp ./*.h $PSRHOME/$LOGIN_ARCH/include/spead2/
RUN cp test_recv test_send test_ringbuffer spead2_bench spead2_recv $PSRHOME/$LOGIN_ARCH/bin/


###################################
# SPIP Built
RUN echo "build form here"
WORKDIR $PSRHOME
RUN git clone https://github.com/ajameson/spip.git
WORKDIR $PSRHOME/spip

ENV LDFLAGS -lstdc++
RUN ./bootstrap
RUN ./configure --prefix=$PSRHOME/$LOGIN_ARCH --with-spead2-dir=$PSRHOME/$LOGIN_ARCH
RUN make clean
RUN make 
RUN make install
ENV LDFLAGS ""

###################################
# DSPSR Build
#RUN mkdir $PSRHOME/dspsr
WORKDIR $PSRHOME
RUN git clone https://git.code.sf.net/p/dspsr/code dspsr-code
WORKDIR $PSRHOME/dspsr-code

#RUN chown -R kat:kat .
#RUN chmod -R  a+rw .

ENV C_INCLUDE_PATH $C_INCLUDE_PATH:/usr/local/kat/pulsar/include/
ENV CPLUS_INCLUDE_PATH $CPLUS_INCLUDE_PATH:/usr/local/kat/pulsar/include/
ENV PATH $PATH:/usr/local/kat/pulsar/include/
ENV PGPLOT_DIR $PSRHOME/pgplot
ENV PGPLOT_FONT $PGPLOT_DIR/grfont.dat
ENV CFLAGS -lstdc++
RUN ./bootstrap
RUN echo "dada dummy fits kat sigproc" > backends.list
#RUN ./configure --prefix=$PSRHOME/$LOGIN_ARCH
RUN ls $PSRHOME/psrdada
RUN ./configure --prefix=$PSRHOME/$LOGIN_ARCH --x-libraries=/usr/lib/x86_64-linux-gnu CPPFLAGS="-I/usr/local/cuda/include -I"$PSRHOME/psrdada"/install/include" LDFLAGS="-L/usr/local/cuda/lib64" LIBS="-lpgplot -lcpgplot -lcuda -lstdc++"
RUN make clean
run make -j 16
run make install

RUN pip install pyfits

WORKDIR $HOME
RUN mkdir /usr/local/kat/pulsar/psrchive/share/
RUN cp $PSRHOME/psrchive/Base/Formats/PSRFITS/psrheader.fits /usr/local/kat/pulsar/psrchive/share/
COPY ./beamformer_docker_software/hardware_cbf_2048chan_2pol.cfg.template $HOME
COPY ./beamformer_docker_software/hardware_cbf_4096chan_2pol.cfg.template $HOME
COPY ./beamformer_docker_software/hardware_cbf_4096chan_2pol.cfg $HOME
COPY ./beamformer_docker_software/hardware_cbf_2048chan_2pol.cfg $HOME
COPY ./beamformer_docker_software/dada.info $HOME


# Install the current package
COPY . /tmp/install/katsdpingest
WORKDIR /tmp/install/katsdpingest
RUN python ./setup.py clean && pip install --no-index .
