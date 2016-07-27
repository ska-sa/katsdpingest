#!/bin/bash
set -e
if [ "$#" -ne 1 ]; then
    echo "Usage: jenkins-autotune.sh <gpu-tag>" 1>&2
    exit 1
fi
GPU="$1"
LABEL="${GIT_BRANCH#origin/}"
if [ "$LABEL" = "master" ]; then
    LABEL=latest
fi
IMAGE="$DOCKER_REGISTRY/katsdpingest_$GPU:$LABEL"
BASE_IMAGE="$DOCKER_REGISTRY/katsdpingest:$LABEL"
COPY_FROM="$DOCKER_REGISTRY/katsdpingest_$GPU:latest"
pip install docker-py==1.7.0
docker pull "$BASE_IMAGE"
docker pull "$COPY_FROM"
scripts/autotune_mkimage.py -H $DOCKER_HOST --tls --copy --copy-from "$COPY_FROM" "$IMAGE" "$BASE_IMAGE"
docker push "$IMAGE"
