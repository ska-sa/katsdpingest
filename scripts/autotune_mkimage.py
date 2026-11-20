#!/usr/bin/env python3
"""Create a derived image from the katsdpingest base image that contains
autotuning results.
"""

import argparse
import sys
import os
import os.path
import tempfile
import tarfile
import io
from typing import Optional

import docker
from docker.types import DeviceRequest
from docker.models.containers import Container


CACHE_DIR = '/home/kat/.cache/katsdpsigproc/'
DB_FILE = 'tuning.db'

DOCKERFILE = """\
FROM {base}
""" + f"""\
COPY --chown=kat:kat {DB_FILE} {CACHE_DIR}
"""


def get_cache(container: Container) -> bytes:
    bits, _ = container.get_archive(CACHE_DIR + DB_FILE)
    tardata = b''.join(bits)
    return tardata


def untar_cache(tardata: bytes) -> bytes:
    with tarfile.open(fileobj=io.BytesIO(tardata)) as tar:
        f = tar.extractfile(DB_FILE)
        if f is None:
            raise RuntimeError(f"{DB_FILE} in archive is not a regular file")
        with f:
            return f.read()


def tune(
    client: docker.DockerClient,
    base_image: str,
    skip: bool,
    init_tar: Optional[bytes] = None,
) -> bytes:
    """Run a throwaway container to do the autotuning, and extract the result."""
    command = ['ingest_autotune.py'] if not skip else ['/bin/true']
    if init_tar is not None:
        command = ['sh', '-xc',
                   f'mkdir -p {CACHE_DIR} && '
                   f'cp /tmp/{DB_FILE} {CACHE_DIR} && {command[0]}']
    # Request every GPU on the Docker host that runs the container.
    # This is the API equivalent of `docker run --gpus all`.
    gpus = DeviceRequest(count=-1, capabilities=[['gpu']])
    container = client.containers.create(base_image, command, device_requests=[gpus])
    try:
        if init_tar is not None:
            container.put_archive('/tmp', init_tar)
        container.start()
        try:
            for chunk in container.logs(stdout=True, stderr=True, stream=True, follow=True):
                sys.stdout.buffer.write(chunk)
            sys.stdout.flush()
            result = container.wait()
        except (Exception, KeyboardInterrupt):
            container.stop(timeout=2)
            raise
        if wait_error := result.get('Error'):
            raise RuntimeError(f"Docker wait error: {wait_error.get('Message')}")
        exit_code = result.get('StatusCode')
        if exit_code != 0:
            raise RuntimeError(f'Autotuning failed with exit code {exit_code}')
        return get_cache(container)
    finally:
        container.remove(force=True)


def build(client: docker.DockerClient, image: str, base_image: str, tuning: bytes) -> None:
    """Build a new `image` that adds `tuning` database to `base_image`."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, 'Dockerfile'), 'w') as f:
            f.write(DOCKERFILE.format(base=base_image))
        with open(os.path.join(tmpdir, DB_FILE), 'wb') as f:
            f.write(tuning)
        for chunk in client.api.build(path=tmpdir, tag=image, rm=True, forcerm=True, decode=True):
            if 'stream' in chunk:
                sys.stdout.write(chunk['stream'])
                sys.stdout.flush()
            elif 'error' in chunk:
                raise RuntimeError(f"Docker build failed: {chunk['error'].strip()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('base_image')
    parser.add_argument(
        '--copy', action='store_true',
        help='Copy old autotuning results from existing image')
    parser.add_argument(
        '--copy-from', type=str, metavar='IMAGE',
        help='Specify alternative image from which to obtain existing results (implies --copy)')
    parser.add_argument(
        '--skip', action='store_true',
        help='Only copy, do not run tuning check afterwards')
    args = parser.parse_args()
    if args.skip and not args.copy and args.copy_from is None:
        parser.error('Cannot use --skip without --copy or --copy-from')

    if args.copy_from is not None:
        copy_base = args.copy_from
    elif args.copy:
        copy_base = args.image
    else:
        copy_base = None

    cli = docker.from_env()
    init_tar = tune(cli, copy_base, True) if copy_base is not None else None
    tuned = untar_cache(tune(cli, args.base_image, args.skip, init_tar))
    build(cli, args.image, args.base_image, tuned)


if __name__ == '__main__':
    sys.exit(main())
