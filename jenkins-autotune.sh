#!/bin/bash
set -e
if [ "$#" -ne 1 ]; then
    echo "Usage: jenkins-autotune.sh <gpu-tag>" 1>&2
    exit 1
fi
GPU="$1"
LABEL="${BRANCH_NAME#origin/}"
if [ "$LABEL" = "master" ]; then
    LABEL=latest
fi
IMAGE="$DOCKER_REGISTRY/katsdpingest_$GPU:$LABEL"
BASE_IMAGE="$DOCKER_REGISTRY/katsdpingest:$LABEL"
COPY_FROM="$DOCKER_REGISTRY/katsdpingest_$GPU:latest"
install_pinned.py -r requirements-autotune.txt
docker pull "$BASE_IMAGE"
docker pull "$COPY_FROM"
trap "docker rmi $IMAGE" EXIT
scripts/autotune_mkimage.py -H $DOCKER_HOST --tls --copy --copy-from "$COPY_FROM" "$IMAGE" "$BASE_IMAGE"
docker push "$IMAGE"

if [ -n "$DOCKER_REGISTRY2" ]; then
    IMAGE2="$DOCKER_REGISTRY2/katsdpingest_$GPU:$LABEL"
    trap "docker rmi $IMAGE2" EXIT
    docker tag "$IMAGE" "$IMAGE2"
    docker push "$IMAGE2"
fi
