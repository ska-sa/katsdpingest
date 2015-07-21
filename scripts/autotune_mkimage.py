#!/usr/bin/env python
"""Create a derived image from the katsdpingest base image that contains
autotuning results.
"""

from __future__ import print_function
import argparse
import glob
import docker
import datetime
import sys
import tempfile
import os

def get_existing(cli, image):
    container = cli.create_container(image=image)
    container_id = container['Id']
    try:
        if container['Warnings']:
            print(container['Warnings'], file=sys.stderr)
        data = cli.copy(container['Id'], '/home/kat/.cache/katsdpsigproc/tuning.db')
        return data.read()
    finally:
        cli.remove_container(container_id)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('base_image')
    parser.add_argument('--copy', action='store_true', help='Copy old autotuning results from existing image')
    parser.add_argument('--copy-from', type=str, metavar='IMAGE', help='Specify alternative image from which to obtain existing results (implies --copy)')
    parser.add_argument('--skip', action='store_true', help='Only copy, do not run tuning check afterwards')
    parser.add_argument('--host', '-H', type=str, default='unix:///var/run/docker.sock', help='Docker host')
    args = parser.parse_args()

    cli = docker.Client(args.host)

    old = None
    if args.copy or args.copy_from is not None:
        if args.copy_from is None:
            args.copy_from = args.image
        old = get_existing(cli, args.copy_from)

    devices = []
    volumes = []
    binds = {}
    db_filename = None
    command = ['ingest_autotune.py']
    for device in glob.glob('/dev/nvidia*'):
        devices.append('{device}:{device}'.format(device=device))
    try:
        if old is not None:
            (handle, filename) = tempfile.mkstemp(suffix='.tar')
            with os.fdopen(handle, 'wb') as f:
                f.write(old)
            # User inside the container probably has different UID
            os.chmod(filename, 0o755)
            command = ['/bin/sh', '-c', 'mkdir -p ~/.cache/katsdpsigproc && tar -C ~/.cache/katsdpsigproc -xf /tuning.tar']
            if not args.skip:
                command[-1] += ' && exec ingest_autotune.py'
            volumes = ['/tuning.tar']
            binds = {filename: {'bind': '/tuning.tar', 'ro': True}}

        container = cli.create_container(
                image=args.base_image,
                volumes=volumes,
                command=command)
        if container['Warnings']:
            print(container['Warnings'], file=sys.stderr)
        try:
            container_id = container['Id']
            cli.start(container_id, devices=devices, binds=binds)
            try:
                for line in cli.logs(container_id, True, True, True):
                    sys.stdout.write(line)
                result = cli.wait(container_id)
                if result == 0:
                    msg = 'Autotuning run at {}'.format(datetime.datetime.now().isoformat())
                    cli.commit(container_id, args.image, '', msg)
                    print('Committed to', args.image)
                    return 0
                else:
                    print('Autotuning failed with status', result)
                    return 1
            except (Exception, KeyboardInterrupt):
                cli.stop(container_id, timeout=2)
                raise
        finally:
            cli.remove_container(container_id)
    finally:
        if db_filename is not None:
            os.remove(db_filename)

if __name__ == '__main__':
    sys.exit(main())
