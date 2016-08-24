#!/usr/bin/env python

"""
Makes autotuning happen for common configurations and for all discovered
devices, so that a subsequent run will not have to wait for autotuning.
"""
import logging
from katsdpingest import ingest_session
from katsdpsigproc import accel


def autotune_device(device):
    context = device.make_context()
    tune_antennas = ingest_session.CBFIngest.tune_antennas
    tune_channels = ingest_session.CBFIngest.tune_channels

    # The tuning parameters are independent, so we just need to tune along a
    # line in each dimension, rather than the Cartesian product.
    for antennas in tune_antennas:
        ingest_session.CBFIngest.create_proc_template(context, antennas, tune_channels[0])
    for channels in tune_channels:
        ingest_session.CBFIngest.create_proc_template(context, tune_antennas[0], channels)


def main():
    logging.getLogger('katsdpsigproc.tune').setLevel(logging.INFO)
    for device in accel.all_devices():
        autotune_device(device)

if __name__ == '__main__':
    main()
