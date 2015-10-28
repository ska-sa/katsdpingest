FROM sdp-ingest5.kat.ac.za:5000/docker-base

MAINTAINER Bruce Merry "bmerry@ska.ac.za"

# Install system packages. Python packages are mostly installed here, but
# certain packages are handled by pip because they're not available.
RUN apt-get -y update && apt-get -y install \
    python-babel \
    python-jinja2 \
    python-markupsafe \
    python-six \
    python-tz

# Install Python dependencies. Versions are explicitly listed and pinned, so
# that the docker image is reproducible. There were all up-to-date versions
# at the time of writing i.e. there are no currently known reasons not to
# update to newer versions.
RUN pip install --no-deps \
        git+ssh://git@github.com/ska-sa/katversion && \
    pip install --no-deps \
        backports.ssl-match-hostname==3.4.0.2 \
        certifi==2015.9.6.2 \
        omnijson==0.1.2 \
        snowballstemmer==1.2.0 \
        tornado==4.2.1 \
        ujson==1.33 \
        wsgiref==0.1.2 \
        git+ssh://git@github.com/ska-sa/katsdptelstate \
        git+ssh://git@github.com/ska-sa/katportalclient

# Install the current package
COPY . /tmp/install/katsdpingest
WORKDIR /tmp/install/katsdpingest
RUN python ./setup.py clean && pip install --no-deps .

# Run as a non-root user
USER kat
WORKDIR /home/kat
