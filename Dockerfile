ARG KATSDPDOCKERBASE_REGISTRY=harbor.sdp.kat.ac.za/dpp

FROM $KATSDPDOCKERBASE_REGISTRY/docker-base-gpu-build:production-tag28102026 AS build

# uv is installed in ~/.local/bin in the base image
# Activate the base image virtualenv and make uv available
ENV PATH="/home/kat/.local/bin:$PATH_PYTHON3" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON3"


# Install Python dependencies
COPY --chown=kat:kat requirements.txt /tmp/install/requirements.txt

# Replace custom install_pinned.py script with uv 
RUN uv pip compile /tmp/install/requirements.txt \
      -o /tmp/install/requirements.lock && \
    uv pip install --no-deps -c /tmp/install/requirements.lock numpy && \
    uv pip install --no-cache --no-deps --no-binary pycuda --no-build-isolation \
      -c /tmp/install/requirements.lock pycuda && \
    uv pip install --no-deps -r /tmp/install/requirements.lock && \
    uv pip check

# Install the current package
COPY --chown=kat:kat . /tmp/install/katsdpingest

RUN cd /tmp/install/katsdpingest && \
    python3 ./setup.py clean   && \
    uv pip install --no-deps . && \
    uv pip check


#######################################################################

FROM $KATSDPDOCKERBASE_REGISTRY/docker-base-gpu-runtime:production-tag28102026
LABEL maintainer="sdpdev+katsdpingest@ska.ac.za"

COPY --chown=kat:kat --from=build /home/kat/ve3 /home/kat/ve3
ENV PATH="$PATH_PYTHON3" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON3"

# Allow raw packets (for ibverbs raw QPs)
USER root
RUN setcap cap_net_raw+p /usr/local/bin/capambel
USER kat

EXPOSE 2040
EXPOSE 7148/udp
