FROM sdp-ingest5.kat.ac.za:5000/docker-base-gpu

MAINTAINER Bruce Merry "bmerry@ska.ac.za"

# Install dependencies.
COPY requirements.txt /tmp/install/requirements.txt
RUN install-requirements.py -d ~/docker-base/base-requirements.txt -d ~/docker-base/gpu-requirements.txt -r /tmp/install/requirements.txt

# Install the current package
COPY . /tmp/install/katsdpingest
WORKDIR /tmp/install/katsdpingest
RUN python ./setup.py clean && pip install --no-index .

EXPOSE 2040
EXPOSE 7147/udp
EXPOSE 7148/udp
