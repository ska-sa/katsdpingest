FROM sdp-docker-registry.kat.ac.za:5000/docker-base-build as build
MAINTAINER Bruce Merry "bmerry@ska.ac.za"

# Enable Python 2 venv
ENV PATH="$PATH_PYTHON2" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON2"

# Install dependencies
COPY requirements.txt /tmp/install/requirements.txt
RUN install-requirements.py -d ~/docker-base/base-requirements.txt -r /tmp/install/requirements.txt

# Install current package
COPY . /tmp/install/katsdpcam2telstate
WORKDIR /tmp/install/katsdpcam2telstate
RUN python ./setup.py clean && pip install --no-deps . && pip check

########################################################################

FROM sdp-docker-registry.kat.ac.za:5000/docker-base-runtime
MAINTAINER Bruce Merry "bmerry@ska.ac.za"

COPY --from=build --chown=kat:kat /home/kat/ve /home/kat/ve
ENV PATH="$PATH_PYTHON2" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON2"

# Expose katcp port
EXPOSE 2047
