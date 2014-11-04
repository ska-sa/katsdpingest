#!/usr/bin/env python

"""Simple test and benchmark of the GPU ingest components"""

import katsdpsigproc.rfi.device as rfi
import katsdpsigproc.accel as accel
import katsdpingest.sigproc as sp
import numpy as np
import argparse

def generate_data(vis_in_device, channels, baselines):
    vis_in = vis_in_device.empty_like()
    vis_in[:] = np.random.normal(scale=32.0, size=(channels, baselines, 2)).astype(np.int32)
    return vis_in

def create_flagger(context, args):
    background = rfi.BackgroundMedianFilterDeviceTemplate(
            context, args.width)
    noise_est = rfi.NoiseEstMADTDeviceTemplate(
            context, 10240)
    threshold = rfi.ThresholdSumDeviceTemplate(
            context, args.sigmas)
    return rfi.FlaggerDeviceTemplate(background, noise_est, threshold)

def create_template(context, args):
    return sp.IngestTemplate(context, create_flagger(context, args), args.freq_avg)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument_group('Data selection')
    parser.add_argument('--antennas', '-a', type=int, default=7)
    parser.add_argument('--channels', '-c', type=int, default=1024)
    parser.add_argument('--border', '-B', type=int, default=0, help='extra overlap channels')
    parser.add_argument('--baselines', '-b', type=int, help='(overrides --antennas)')

    parser.add_argument_group('Parameters')
    parser.add_argument('--time-avg', '-T', type=int, default=4, help='number of input dumps per output dump')
    parser.add_argument('--freq-avg', '-F', type=int, default=16, help='number of input channels per continuum channel')
    parser.add_argument('--width', '-w', type=int, help='median filter kernel size (must be odd)', default=13)
    parser.add_argument('--sigmas', type=float, help='threshold for detecting RFI', default=11.0)

    args = parser.parse_args()
    channels = args.channels + args.border
    channel_range = (args.border // 2, args.channels + args.border // 2)
    if args.baselines is not None:
        baselines = args.baselines
    else:
        baselines = args.antennas * (args.antennas + 1) * 2

    context = accel.create_some_context(True)
    command_queue = context.create_command_queue(profile=True)
    template = create_template(context, args)
    proc = template.instantiate(command_queue, channels, channel_range, baselines)
    print "{0} bytes required".format(proc.required_bytes())
    proc.ensure_all_bound()
    permutation = np.random.permutation(baselines).astype(np.uint16)
    proc.buffer('permutation').set(command_queue, permutation)

    command_queue.finish()

    vis_in_device = proc.buffer('vis_in')
    output_names = ['spec_vis', 'spec_weights', 'spec_flags', 'cont_vis', 'cont_weights', 'cont_flags']
    output_buffers = [proc.buffer(name) for name in output_names]
    output_arrays = [buf.empty_like() for buf in output_buffers]

    dumps = [generate_data(vis_in_device, channels, baselines) for i in range(args.time_avg)]
    start_event = command_queue.enqueue_marker()
    proc.start_sum()
    for dump in dumps:
        proc.buffer('vis_in').set_async(command_queue, dump)
        proc()
    proc.end_sum()
    for buf, array in zip(output_buffers, output_arrays):
        buf.get_async(command_queue, array)
    end_event = command_queue.enqueue_marker()
    print "{0:.3f}ms".format(end_event.time_since(start_event) * 1000.0)

if __name__ == '__main__':
    main()
