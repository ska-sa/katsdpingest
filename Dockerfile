ARG KATSDPDOCKERBASE_REGISTRY=harbor.sdp.kat.ac.za/dpp

FROM $KATSDPDOCKERBASE_REGISTRY/docker-base-gpu-build AS build

# Install Python dependencies
COPY --chown=kat:kat requirements.txt /tmp/install/requirements.txt

# Replace custom install_pinned.py script with uv 
# numpy is installed first so pycuda can find its headers during
# compilation, then installs everything else from the lock file
RUN uv pip install --no-deps numpy && \
    uv pip install --no-cache --no-deps --no-binary pycuda \
        --no-build-isolation pycuda && \
    uv pip install --no-deps \
        -r /tmp/install/requirements.txt && \
    uv pip check

# Install the current package
COPY --chown=kat:kat . /tmp/install/katsdpingest

RUN cd /tmp/install/katsdpingest && \
    uv pip install --no-deps . && \
    uv pip check


#######################################################################

FROM $KATSDPDOCKERBASE_REGISTRY/docker-base-gpu-runtime
LABEL maintainer="sdpdev+katsdpingest@ska.ac.za"

COPY --chown=kat:kat --from=build /home/kat/ve3 /home/kat/ve3
ENV PATH="$PATH_PYTHON3" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON3"

# Allow raw packets (for ibverbs raw QPs)
USER root
RUN setcap cap_net_raw+p /usr/local/bin/capambel
USER kat

EXPOSE 2040
EXPOSE 7148/udp
