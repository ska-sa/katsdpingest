#!/usr/bin/env python

"""Simple test and benchmark of the GPU ingest components"""

import argparse

import katsdpsigproc.accel as accel
import katsdpsigproc.rfi.device as rfi
import numpy as np
from katdal.flags import CAM, STATIC

import katsdpingest.sigproc as sp
from katsdpingest.utils import Range


def add_random_flags(flags, flag_value, probability=0.1):
    few_random_indices = np.random.choice(
        flags.size, int(np.ceil(probability * flags.size)), replace=False
    )
    flags.flat[few_random_indices] |= np.uint8(flag_value)


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
    # XXX Ensure that output has plain ints and not np.int64, as
    # these numbers eventually pass via katsdpsigproc into pycuda,
    # which cannot digest NumPy types.
    cuts = np.cumsum([0] + sections).tolist()
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
    return sp.IngestTemplate(
        context,
        create_flagger(context, args),
        percentile_sizes,
        args.excise,
        args.continuum,
    )


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument_group('Data selection')
    parser.add_argument('--antennas', '-a', type=int, default=7,
                        help='total number of antennas',)
    parser.add_argument('--mask-antennas', type=int, default=None,
                        help='number of antennas in antenna mask')
    parser.add_argument('--channels', '-c', type=int, default=1024,
                        help='number of frequency channels')
    parser.add_argument('--border', '-B', type=int, default=0, help='extra overlap channels')

    parser.add_argument_group('Parameters')
    parser.add_argument('--time-avg', '-T', type=int, default=4,
                        help='number of input dumps per output dump')
    parser.add_argument('--freq-avg', '-F', type=int, default=16,
                        help='number of input channels per continuum channel')
    parser.add_argument('--sd-time-avg', type=int, default=4,
                        help='number of input dumps per signal display dump')
    parser.add_argument('--sd-freq-avg', type=int, default=128,
                        help='number of input channels for signal display channel')
    parser.add_argument('--width', '-w', type=int, default=13,
                        help='median filter kernel size (must be odd)')
    parser.add_argument('--sigmas', type=float, default=11.0,
                        help='threshold for detecting RFI')
    parser.add_argument('--excise', action='store_true',
                        help='enable excision of flagged data')
    parser.add_argument('--continuum', action='store_true',
                        help='enable continuum output')
    parser.add_argument('--repeat', '-r', type=int, default=8,
                        help='number of dumps to process')
    parser.add_argument('--no-transfer', '-N', action='store_true',
                        help='skip data transfers')

    args = parser.parse_args()
    channels = args.channels + args.border
    channel_range = Range(args.border // 2, args.channels + args.border // 2)
    kept_channels = len(channel_range)
    count_flags_channel_range = Range(
        channel_range.start - args.border // 4,
        channel_range.stop + args.border // 4,
    )
    cbf_baselines = args.antennas * (args.antennas + 1) * 2
    if args.mask_antennas is None:
        args.mask_antennas = args.antennas
    baselines = args.mask_antennas * (args.mask_antennas + 1) * 2
    masks = 3

    context = accel.create_some_context(True)
    command_queue = context.create_command_queue(profile=True)
    template = create_template(context, args)
    proc = template.instantiate(
        command_queue,
        channels,
        channel_range,
        count_flags_channel_range,
        cbf_baselines,
        baselines,
        masks,
        cont_factor=args.freq_avg,
        sd_cont_factor=args.sd_freq_avg,
        percentile_ranges=create_percentile_ranges(args.mask_antennas),
        threshold_args={'n_sigma': args.sigmas}
    )
    print("{0} bytes required".format(proc.required_bytes()))
    proc.ensure_all_bound()

    permutation = np.random.permutation(baselines).astype(np.int16)
    permutation = np.r_[permutation, -np.ones(cbf_baselines - baselines, np.int16)]
    channel_mask = np.zeros((masks, channels), np.uint8)
    channel_mask[:, :args.border + 2] |= np.uint8(STATIC)
    channel_mask[:, -args.border - 2:] |= np.uint8(STATIC)
    add_random_flags(channel_mask, CAM)
    channel_mask_idx = np.random.randint(0, masks, baselines).astype(np.uint32)
    baseline_flags = np.zeros((baselines,), np.uint8)
    add_random_flags(baseline_flags, CAM, probability=0.01)
    timeseries_weights = np.random.choice(2, size=kept_channels, p=[0.1, 0.9])
    timeseries_weights = timeseries_weights.astype(np.float32)
    timeseries_weights /= np.sum(timeseries_weights)

    proc.buffer('permutation').set(command_queue, permutation)
    proc.buffer('channel_mask').set(command_queue, channel_mask)
    proc.buffer('channel_mask_idx').set(command_queue, channel_mask_idx)
    proc.buffer('baseline_flags').set(command_queue, baseline_flags)
    proc.buffer('timeseries_weights').set(command_queue, timeseries_weights)

    command_queue.finish()

    vis_in_device = proc.buffer('vis_in')
    output_names = ['spec_vis', 'spec_weights', 'spec_weights_channel', 'spec_flags']
    if args.continuum:
        output_names.extend(['cont_vis', 'cont_weights', 'cont_weights_channel', 'cont_flags'])
    output_buffers = [proc.buffer(name) for name in output_names]
    output_arrays = [buf.empty_like() for buf in output_buffers]
    sd_names = ['sd_spec_vis', 'sd_spec_flags', 'sd_spec_weights', 'sd_spec_weights_channel',
                'sd_flag_counts', 'sd_flag_any_counts', 'timeseries', 'timeseriesabs']
    if args.continuum:
        sd_names.extend(
            ['sd_cont_vis', 'sd_cont_flags', 'sd_cont_weights', 'sd_cont_weights_channel']
        )
    for i in range(5):
        sd_names.append('percentile{0}'.format(i))
    sd_buffers = [proc.buffer(name) for name in sd_names]
    sd_arrays = [buf.empty_like() for buf in sd_buffers]

    dumps = [generate_data(vis_in_device, channels, cbf_baselines)
             for i in range(max(args.sd_time_avg, args.time_avg))]
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
    print("{0:.3f}ms ({1:.3f}ms per dump)".format(elapsed_ms, dump_ms))


if __name__ == '__main__':
    main()
