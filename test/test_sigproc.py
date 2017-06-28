#!/usr/bin/env python

"""Simple test and benchmark of the GPU ingest components"""

import katsdpsigproc.rfi.device as rfi
import katsdpsigproc.accel as accel
import katsdpingest.sigproc as sp
from katsdpingest.utils import Range
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
            context, args.channels + args.border)
    threshold = rfi.ThresholdSumDeviceTemplate(context)
    return rfi.FlaggerDeviceTemplate(background, noise_est, threshold)


def create_percentile_ranges(antennas):
    n_cross = antennas * (antennas - 1) // 2
    sections = [
            antennas,          # autohh
            antennas,          # autovv
            2 * antennas,      # autohv (each appears as hv and vh)
            n_cross,           # crosshh
            n_cross,           # crossvv
            2 * n_cross        # crosshv
    ]
    cuts = np.cumsum([0] + sections)
    return [
            (cuts[0], cuts[2]),  # autohhvv
            (cuts[0], cuts[1]),  # autohh
            (cuts[1], cuts[2]),  # autovv
            (cuts[2], cuts[3]),  # autohv
            (cuts[3], cuts[5]),  # crosshhvv
            (cuts[3], cuts[4]),  # crosshh
            (cuts[4], cuts[5]),  # crossvv
            (cuts[5], cuts[6])   # crosshv
    ]


def create_template(context, args):
    percentile_ranges = create_percentile_ranges(args.mask_antennas)
    percentile_sizes = list(set([x[1] - x[0] for x in percentile_ranges]))
    return sp.IngestTemplate(context, create_flagger(context, args), percentile_sizes, args.excise)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument_group('Data selection')
    parser.add_argument('--antennas', '-a', type=int, help='total number of antennas', default=7)
    parser.add_argument('--mask-antennas', type=int, help='number of antennas in antenna mask', default=None)
    parser.add_argument('--channels', '-c', type=int, help='number of frequency channels', default=1024)
    parser.add_argument('--border', '-B', type=int, default=0, help='extra overlap channels')

    parser.add_argument_group('Parameters')
    parser.add_argument('--time-avg', '-T', type=int, default=4, help='number of input dumps per output dump')
    parser.add_argument('--freq-avg', '-F', type=int, default=16, help='number of input channels per continuum channel')
    parser.add_argument('--sd-time-avg', type=int, default=4, help='number of input dumps per signal display dump')
    parser.add_argument('--sd-freq-avg', type=int, default=128, help='number of input channels for signal display channel')
    parser.add_argument('--width', '-w', type=int, help='median filter kernel size (must be odd)', default=13)
    parser.add_argument('--sigmas', type=float, help='threshold for detecting RFI', default=11.0)
    parser.add_argument('--no-excise', dest='excise', action='store_false', help='disable excision of flagged data')
    parser.add_argument('--repeat', '-r', type=int, default=8, help='number of dumps to process')
    parser.add_argument('--no-transfer', '-N', action='store_true', help='skip data transfers')

    args = parser.parse_args()
    channels = args.channels + args.border
    channel_range = Range(args.border // 2, args.channels + args.border // 2)
    cbf_baselines = args.antennas * (args.antennas + 1) * 2
    if args.mask_antennas is None:
        args.mask_antennas = args.antennas
    baselines = args.mask_antennas * (args.mask_antennas + 1) * 2

    context = accel.create_some_context(True)
    command_queue = context.create_command_queue(profile=True)
    template = create_template(context, args)
    proc = template.instantiate(
            command_queue, channels, channel_range, 2 * args.mask_antennas,
            cbf_baselines, baselines,
            args.freq_avg, args.sd_freq_avg, create_percentile_ranges(args.mask_antennas),
            threshold_args={'n_sigma': args.sigmas})
    print "{0} bytes required".format(proc.required_bytes())
    proc.ensure_all_bound()

    permutation = np.random.permutation(baselines).astype(np.int16)
    permutation = np.r_[permutation, -np.ones(cbf_baselines - baselines, np.int16)]
    # The baseline_inputs and input_auto_baseline arrays aren't consistent
    # with the percentile ranges, but that doesn't really matter (although
    # it may impact memory access patterns and hence performance).
    baseline_inputs = []
    input_auto_baseline = np.zeros(2 * args.mask_antennas, np.uint16)
    for i in range(2 * args.mask_antennas):
        for j in range(i // 2 * 2, 2 * args.mask_antennas):
            if i == j:
                input_auto_baseline[i] = len(baseline_inputs)
            baseline_inputs.append((i, j))
    assert len(baseline_inputs) == baselines
    baseline_inputs = np.array(baseline_inputs, np.uint16)
    proc.buffer('permutation').set(command_queue, permutation)
    proc.buffer('input_auto_baseline').set(command_queue, input_auto_baseline)
    proc.buffer('baseline_inputs').set(command_queue, baseline_inputs)

    command_queue.finish()

    vis_in_device = proc.buffer('vis_in')
    output_names = ['spec_vis', 'spec_weights', 'spec_weights_channel', 'spec_flags',
                    'cont_vis', 'cont_weights', 'cont_weights_channel', 'cont_flags']
    output_buffers = [proc.buffer(name) for name in output_names]
    output_arrays = [buf.empty_like() for buf in output_buffers]
    sd_names = ['sd_cont_vis', 'sd_cont_flags', 'sd_cont_weights', 'timeseries', 'timeseriesabs']
    for i in range(5):
        sd_names.append('percentile{0}'.format(i))
    sd_buffers = [proc.buffer(name) for name in sd_names]
    sd_arrays = [buf.empty_like() for buf in sd_buffers]

    dumps = [generate_data(vis_in_device, channels, cbf_baselines) for i in range(max(args.sd_time_avg, args.time_avg))]
    # Push data before we start timing, to ensure everything is allocated
    for dump in dumps:
        proc.buffer('vis_in').set(command_queue, dump)

    start_event = command_queue.enqueue_marker()
    proc.start_sum()
    for pass_ in range(args.repeat):
        if not args.no_transfer:
            proc.buffer('vis_in').set_async(command_queue, dumps[pass_ % len(dumps)])
        proc()
        if (pass_ + 1) % args.time_avg == 0:
            proc.end_sum()
            if not args.no_transfer:
                for buf, array in zip(output_buffers, output_arrays):
                    buf.get_async(command_queue, array)
            proc.start_sum()
        if (pass_ + 1) % args.sd_time_avg == 0:
            proc.end_sd_sum()
            if not args.no_transfer:
                for buf, array in zip(sd_buffers, sd_arrays):
                    buf.get_async(command_queue, array)
            proc.start_sd_sum()
    end_event = command_queue.enqueue_marker()
    elapsed_ms = end_event.time_since(start_event) * 1000.0
    dump_ms = elapsed_ms / args.repeat
    print "{0:.3f}ms ({1:.3f}ms per dump)".format(elapsed_ms, dump_ms)

if __name__ == '__main__':
    main()
