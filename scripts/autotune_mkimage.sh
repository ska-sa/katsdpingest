#!/bin/bash
# Create a derived image from the katsdpingest base image that contains
# autotuning results. The ska_sa/katsdpingest image must be available.
set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <image-name>" 1>&2
    exit 1
fi

args=()
for i in /dev/nvidia*; do
    args+=(--device $i:$i)
done
container="$(docker run -d "${args[@]}" ska_sa/katsdpingest ingest_autotune.py)"
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
