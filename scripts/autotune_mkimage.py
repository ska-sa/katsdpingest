#!/usr/bin/env python
"""Create a derived image from the katsdpingest base image that contains
autotuning results.
"""

from __future__ import print_function
import argparse
import glob
import datetime
import sys
import os
import os.path
import subprocess
import docker
from docker import APIClient


def get_existing(cli, image):
    container = cli.create_container(image=image)
    container_id = container['Id']
    try:
        if container['Warnings']:
            print(container['Warnings'], file=sys.stderr)
        data, _ = cli.get_archive(container['Id'], '/home/kat/.cache/katsdpsigproc')
        return data.read()
    finally:
        cli.remove_container(container_id)


def split_tag(path):
    """Split an image or image:tag string into image and tag parts. Docker
    doesn't seem to have a spec for this, so we assume that anything after the
    last colon is a tag, *unless* it contains a slash. The exception is so that
    a port number on a registry isn't mistaken for a tag.
    """
    last_slash = path.rfind('/')
    last_colon = path.rfind(':')
    if last_colon > last_slash:
        return path[:last_colon], path[last_colon + 1:]
    else:
        # No tag
        return path, ''


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

    if args.tls:
        # docker-py doesn't support using tcp:// in the URL
        if args.host.startswith('tcp://'):
            args.host = 'https://' + args.host[6:]
        tls_config = docker.tls.TLSConfig(
            client_cert=(os.path.expanduser('~/.docker/cert.pem'),
                         os.path.expanduser('~/.docker/key.pem')),
            verify=os.path.expanduser('~/.docker/ca.pem'))
        cli = APIClient(args.host, tls=tls_config)
    else:
        cli = APIClient(args.host)
    old = None
    if args.copy or args.copy_from is not None:
        if args.copy_from is None:
            args.copy_from = args.image
        old = get_existing(cli, args.copy_from)

    # put_archive makes the cache owned by root. To deal with this, we run as
    # root, then fix up permissions afterwards.
    command = ['sh', '-c', 'HOME=/home/kat ingest_autotune.py && chown -R kat:kat /home/kat/.cache']
    # It's tempting to use the nvidia-docker REST API to get the appropriate
    # options, but that has two drawbacks:
    # 1. This script is run inside a Docker container as part of CI, and
    # there is no access from that container to the host's nvidia-docker-plugin
    # daemon.
    # 2. If that was fixed, the REST API would enumerate all GPUs in the
    # system, instead of just those assigned to the particular Jenkins slave.
    #
    # So instead, we fall back on working out the appropriate options
    # by hand.
    devices = glob.glob('/dev/nvidia*')
    driver_version = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=driver_version', '--id=0', '--format=csv,noheader'])
    driver_version = driver_version.strip()
    if not driver_version:
        raise RuntimeError('Could not determine NVIDIA driver version')
    binds = ['nvidia_driver_' + driver_version + ':/usr/local/nvidia:ro']
    volumes = ['/usr/local/nvidia']
    container = cli.create_container(
        image=args.base_image,
        command=command,
        user='root', volumes=volumes, volume_driver='nvidia-docker',
        host_config=cli.create_host_config(devices=devices, binds=binds))
    try:
        if container['Warnings']:
            print(container['Warnings'], file=sys.stderr)
        container_id = container['Id']
        if old is not None:
            cli.put_archive(container_id, '/home/kat/.cache', old)
        if not args.skip:
            cli.start(container_id)
            try:
                for line in cli.logs(container_id, True, True, True):
                    sys.stdout.write(line)
                result = cli.wait(container_id)
            except (Exception, KeyboardInterrupt):
                cli.stop(container_id, timeout=2)
                raise
        else:
            result = 0
        if result == 0:
            msg = 'Autotuning run at {}'.format(datetime.datetime.now().isoformat())
            image, tag = split_tag(args.image)
            # Passing 'USER kat' sets the user back to kat, since we
            # override it to root earlier.
            cli.commit(container_id, image, tag, msg, changes='USER kat')
            print('Committed to', args.image)
            return 0
        else:
            print('Autotuning failed with status', result)
            return 1
    finally:
        cli.remove_container(container_id)


if __name__ == '__main__':
    sys.exit(main())
