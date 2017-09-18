#!/usr/bin/env python

"""
Makes autotuning happen for common configurations and for all discovered
devices, so that a subsequent run will not have to wait for autotuning.
"""
from __future__ import print_function
import logging
import sys
from katsdpingest import ingest_session
from katsdpsigproc import accel


def autotune_device(device):
    context = device.make_context()
    tune_channels = ingest_session.CBFIngest.tune_channels
    tune_percentile_sizes = ingest_session.CBFIngest.tune_percentile_sizes
    for channels in tune_channels:
        for excise in [False, True]:
            for continuum in [False, True]:
                ingest_session.CBFIngest.create_proc_template(
                    context, tune_percentile_sizes, channels, excise, continuum)


def main():
    logging.basicConfig(level='INFO')
    logging.getLogger('katsdpsigproc.tune').setLevel(logging.INFO)
    devices = accel.all_devices()
    if not devices:
        logging.error('No acceleration devices found')
        sys.exit(1)
    for device in devices:
        autotune_device(device)


if __name__ == '__main__':
    main()
