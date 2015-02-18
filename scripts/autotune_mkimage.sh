#!/bin/bash
# Create a derived image from the katsdpingest base image that contains
# autotuning results.
set -e

if [ $# -ne 2 ]; then
    echo "Usage: $0 <image-name> <base-image>" 1>&2
    exit 1
fi

args=()
for i in /dev/nvidia*; do
    args+=(--device $i:$i)
done
docker pull "$2"
container="$(docker run -d "${args[@]}" "$2" ingest_autotune.py)"
if [ -z "$container" ]; then
    echo "did not get container name" 1>&2
    exit 1
fi

echo "Started container $container, waiting..."
status="$(docker wait $container)"
if [ "$status" = "0" ]; then
    docker commit "$container" "$1"
else
    echo "autotuner failed with status $status" 1>&2
    exit "$status"
fi
