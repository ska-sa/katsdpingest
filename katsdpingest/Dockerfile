FROM sdp-docker-registry.kat.ac.za:5000/docker-base-gpu-build as build
MAINTAINER Bruce Merry "bmerry@ska.ac.za"

# Enable Python 3 venv
ENV PATH="$PATH_PYTHON3" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON3"

# Install Python dependencies
COPY --chown=kat:kat requirements.txt /tmp/install/requirements.txt
RUN install-requirements.py -d ~/docker-base/base-requirements.txt -d ~/docker-base/gpu-requirements.txt -r /tmp/install/requirements.txt

# Install the current package
COPY --chown=kat:kat . /tmp/install/katsdpingest
RUN cd /tmp/install/katsdpingest && \
    python ./setup.py clean && pip install --no-deps . && pip check

#######################################################################

FROM sdp-docker-registry.kat.ac.za:5000/docker-base-gpu-runtime
MAINTAINER Bruce Merry "bmerry@ska.ac.za"

COPY --chown=kat:kat --from=build /home/kat/ve3 /home/kat/ve3
ENV PATH="$PATH_PYTHON3" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON3"

EXPOSE 2040
EXPOSE 7148/udp
