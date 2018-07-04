#!/usr/bin/env python3
"""Create a derived image from the katsdpingest base image that contains
autotuning results.
"""

import argparse
import sys
import os
import os.path
import tempfile
import shutil
import tarfile
import io
import contextlib
from textwrap import dedent

import docker
from docker import APIClient


DOCKERFILE = dedent('''\
    FROM {base}
    COPY --chown=kat:kat tuning.db /home/kat/.cache/katsdpsigproc/
''')


def get_cache(cli, container_id):
    data, _ = cli.get_archive(container_id, '/home/kat/.cache/katsdpsigproc/tuning.db')
    tardata = b''.join(data)
    with tarfile.open(fileobj=io.BytesIO(tardata)) as tar:
        with contextlib.closing(tar.extractfile('tuning.db')) as f:
            return f.read()


def tune(cli, base_image, skip):
    """Run a throwaway container to do the autotuning, and extract the result."""
    command = ['ingest_autotune.py'] if not skip else ['/bin/true']
    # If we're running inside a Docker container, expose the same devices
    # to our child container.
    environment = {
        'NVIDIA_VISIBLE_DEVICES': os.environ.get('NVIDIA_VISIBLE_DEVICES', 'all')
    }
    container = cli.create_container(
        image=base_image,
        command=command,
        environment=environment,
        runtime='nvidia')
    try:
        if container['Warnings']:
            print(container['Warnings'], file=sys.stderr)
        container_id = container['Id']
        cli.start(container_id)
        try:
            for line in cli.logs(container_id, True, True, True):
                sys.stdout.buffer.write(line)
            result = cli.wait(container_id)
        except (Exception, KeyboardInterrupt):
            cli.stop(container_id, timeout=2)
            raise
        if result['StatusCode'] == 0:
            return get_cache(cli, container_id)
        else:
            msg = 'Autotuning failed with status {0[Error]} ({0[StatusCode]})'.format(result)
            raise RuntimeError(msg)
    finally:
        cli.remove_container(container_id)


def build(cli, image, base_image, tuning):
    tmpdir = tempfile.mkdtemp()
    try:
        with open(os.path.join(tmpdir, 'Dockerfile'), 'w') as f:
            f.write(DOCKERFILE.format(base=base_image))
        with open(os.path.join(tmpdir, 'tuning.db'), 'wb') as f:
            f.write(tuning)
        for line in cli.build(path=tmpdir, rm=True, tag=image, decode=True):
            if 'stream' in line:
                sys.stdout.write(line['stream'])
    finally:
        shutil.rmtree(tmpdir)


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
    parser.add_argument(
        '--host', '-H', type=str, default='unix:///var/run/docker.sock',
        help='Docker host')
    parser.add_argument(
        '--tls', action='store_true',
        help='Use TLS to connect to Docker daemon')
    args = parser.parse_args()
    if args.skip and not args.copy and args.copy_from is None:
        parser.error('Cannot use --skip without --copy or --copy-from')

    if args.tls:
        tls_config = docker.tls.TLSConfig(
            client_cert=(os.path.expanduser('~/.docker/cert.pem'),
                         os.path.expanduser('~/.docker/key.pem')),
            verify=os.path.expanduser('~/.docker/ca.pem'))
        cli = APIClient(args.host, tls=tls_config)
    else:
        cli = APIClient(args.host)

    if args.copy_from is not None:
        tune_base = args.copy_from
    elif args.copy:
        tune_base = args.image
    else:
        tune_base = args.base_image

    tuned = tune(cli, tune_base, args.skip)
    build(cli, args.image, args.base_image, tuned)


if __name__ == '__main__':
    sys.exit(main())
