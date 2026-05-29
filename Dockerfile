#ARG KATSDPDOCKERBASE_REGISTRY=127.0.0.1:5000
ARG KATSDPDOCKERBASE_REGISTRY=harbor.sdp.kat.ac.za/dpp

FROM $KATSDPDOCKERBASE_REGISTRY/base-gpu-build:focaluvpip AS build

# Enable Python 3 venv
ENV PATH="$PATH_PYTHON3" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON3"

# Install Python dependencies
COPY --chown=kat:kat requirements.txt /tmp/install/requirements.txt
#RUN install_pinned.py -r /tmp/install/requirements.txt
RUN uv pip compile /tmp/install/requirements.txt \
      -o /tmp/install/requirements.lock && \
    uv pip sync /tmp/install/requirements.lock --strict

# Install the current package
COPY --chown=kat:kat . /tmp/install/katsdpingest
RUN cd /tmp/install/katsdpingest && \
    python ./setup.py clean && pip install --no-deps . && pip check

#######################################################################

FROM $KATSDPDOCKERBASE_REGISTRY/base-gpu-runtime:focaluvpip
LABEL maintainer="sdpdev+katsdpingest@ska.ac.za"

COPY --chown=kat:kat --from=build /home/kat/ve3 /home/kat/ve3
ENV PATH="$PATH_PYTHON3" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON3"

# Allow raw packets (for ibverbs raw QPs)
USER root
RUN setcap cap_net_raw+p /usr/local/bin/capambel
USER kat

EXPOSE 2040
EXPOSE 7148/udp
