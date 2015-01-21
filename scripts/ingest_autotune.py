#!/usr/bin/env python

"""
Makes autotuning happen for common configurations and for all discovered
devices, so that a subsequent run will not have to wait for autotuning.
"""
from katsdpingest import ingest_threads
from katsdpsigproc import accel

def autotune_device(device):
    context = device.make_context()
    ingest_threads.CBFIngest.create_proc_template(context)

def main():
    for device in accel.all_devices():
        autotune_device(device)

if __name__ == '__main__':
    main()
