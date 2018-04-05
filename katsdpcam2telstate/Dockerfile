FROM sdp-docker-registry.kat.ac.za:5000/docker-base

MAINTAINER Bruce Merry "bmerry@ska.ac.za"

# Install dependencies
COPY requirements.txt /tmp/install/requirements.txt
RUN install-requirements.py -d ~/docker-base/base-requirements.txt -r /tmp/install/requirements.txt

# Install current package
COPY . /tmp/install/katsdpcam2telstate
WORKDIR /tmp/install/katsdpcam2telstate
RUN python ./setup.py clean && pip install --no-deps . && pip check

# Expose katcp port
EXPOSE 2047
