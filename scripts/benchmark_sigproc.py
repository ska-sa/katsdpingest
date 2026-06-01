#!/usr/bin/env python

"""Simple test and benchmark of the GPU ingest components"""

import argparse

import katsdpsigproc.accel as accel
import katsdpsigproc.rfi.device as rfi
import numpy as np
from katdal.flags import CAM, STATIC

from katsdpingest.ingest_session import ChannelRanges
import katsdpingest.sigproc as sp
from katsdpingest.utils import Range


def create_channel_ranges(args):
    return ChannelRanges(
        servers=args.servers,
        server_id=args.server_id - 1,
        channels=args.cbf_channels,
        cont_factor=args.continuum_factor,
        sd_cont_factor=args.sd_continuum_factor,
        streams=args.servers,  # not used, so pretend there is one stream per server
        guard=args.guard_channels,
        all_output=Range(0, args.cbf_channels),
        all_sd_output=Range(0, args.cbf_channels),
    )


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
    channel_ranges = create_channel_ranges(args)
    input_channels = len(channel_ranges.input)
    background = rfi.BackgroundMedianFilterDeviceTemplate(
        context, args.flagger_width, use_flags=rfi.BackgroundFlags.FULL
    )
    noise_est = rfi.NoiseEstMADTDeviceTemplate(context, input_channels)
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
    percentile_ranges = create_percentile_ranges(args.output_antennas)
    percentile_sizes = list(set([x[1] - x[0] for x in percentile_ranges]))
    return sp.IngestTemplate(
        context,
        create_flagger(context, args),
        percentile_sizes,
        args.excise,
        args.continuum,
    )


def main():
    description = """
    Benchmark the ingest pipeline on generated correlator data that
    strips out the network layer. The **input** parameters are associated
    with the CBF SPEAD stream, while the **output** parameters are
    associated with the SDP L0 output products and signal display ("sd")
    streams.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument_group('Data size setup')
    parser.add_argument('--input-antennas', '-a', type=int, default=7,
                        help='total number of antennas in CBF input stream')
    parser.add_argument('--output-antennas', type=int, default=None,
                        help='number of antennas in SDP output products (keeps all by default)')
    parser.add_argument('--cbf-channels', '-c', type=int, default=1024,
                        help='number of frequency channels in CBF input stream')
    parser.add_argument('--servers', type=int, default=1,
                        help='number of intended parallel ingest servers (only simulate one)')

    parser.add_argument_group('Parameters')
    parser.add_argument('--server-id', type=int, default=None,
                        help='index of this server amongst parallel servers (1-based)')
    parser.add_argument('--input-int-time', default=0.5, type=float,
                        help='seconds between input CBF dumps')
    parser.add_argument('--output-int-time', default=2.0, type=float,
                        help='seconds between output SDP dumps')
    parser.add_argument('--sd-int-time', default=2.0, type=float,
                        help='seconds between signal display updates')
    parser.add_argument('--continuum-factor', '-F', type=int, default=16,
                        help='number of input CBF channels per continuum output SDP channel')
    parser.add_argument('--sd-continuum-factor', type=int, default=128,
                        help='number of input CBF channels per signal display channel')
    parser.add_argument('--guard-channels', '-B', type=int, default=64,
                        help='number of extra input channels on either side used by flagger')
    parser.add_argument('--flagger-width', '-w', type=int, default=13,
                        help='median filter kernel size in ingest RFI flagger (must be odd)')
    parser.add_argument('--flagger-sigma', type=float, default=11.0,
                        help='threshold for detecting RFI in ingest RFI flagger')
    parser.add_argument('--excise', action='store_true',
                        help='enable excision of flagged data')
    parser.add_argument('--continuum', action='store_true',
                        help='enable continuum output')
    parser.add_argument('--repeat', '-r', type=int, default=None,
                        help='number of input CBF dumps to process (default is 2 SDP dumps)')
    parser.add_argument('--no-transfer', '-N', action='store_true',
                        help='skip data transfers')

    args = parser.parse_args()
    if args.server_id is None:
        args.server_id = 3 if args.servers == 4 else 1
    if args.output_antennas is None:
        args.output_antennas = args.input_antennas
    dumps_per_output_dump = max(1, int(round(args.output_int_time / args.input_int_time)))
    dumps_per_sd_dump = max(1, int(round(args.sd_int_time / args.input_int_time)))
    if args.repeat is None:
        args.repeat = max(2 * dumps_per_output_dump, 2 * dumps_per_sd_dump)

    channel_ranges = create_channel_ranges(args)
    input_channels = len(channel_ranges.input)
    output_channel_range = channel_ranges.computed.relative_to(channel_ranges.input)
    sd_channel_range = channel_ranges.sd_output.relative_to(channel_ranges.input)
    output_channels = len(output_channel_range)
    input_baselines = args.input_antennas * (args.input_antennas + 1) * 2
    output_baselines = args.output_antennas * (args.output_antennas + 1) * 2
    masks = 2
    percentile_ranges = create_percentile_ranges(args.output_antennas)

    print(f"Total number of CBF frequency channels: {len(channel_ranges.cbf)}")
    print(f"This is meant to be split across {args.servers} ingest server(s)")
    print(f"This simulated server processes channels {channel_ranges.input}")
    print(f"Number of channels: input={input_channels}, output={output_channels}")
    print(f"Number of baselines: input={input_baselines}, output={output_baselines}")
    print(f"Number of input dumps per output dump: {dumps_per_output_dump}")
    print(f"Number of input dumps per signal display dump: {dumps_per_sd_dump}")

    context = accel.create_some_context(True)
    command_queue = context.create_command_queue(profile=True)
    template = create_template(context, args)
    proc = template.instantiate(
        command_queue,
        input_channels,
        output_channel_range,
        sd_channel_range,
        input_baselines,
        output_baselines,
        masks,
        channel_ranges.cont_factor,
        channel_ranges.sd_cont_factor,
        percentile_ranges,
        threshold_args={'n_sigma': args.flagger_sigma}
    )
    print(f"Total device buffer size = {proc.required_bytes()/1e6:.6f} MB")
    proc.ensure_all_bound()

    permutation = np.random.permutation(output_baselines).astype(np.int16)
    permutation = np.r_[permutation, -np.ones(input_baselines - output_baselines, np.int16)]
    # The channel mask combines multiple static masks and the channel_data_suspect sensor
    channel_mask = np.zeros((masks, input_channels), np.uint8)
    channel_mask[:, :input_channels // 10] |= np.uint8(STATIC)
    channel_mask[:, -input_channels // 10:] |= np.uint8(STATIC)
    add_random_flags(channel_mask, CAM)
    channel_mask_idx = np.random.randint(0, masks, output_baselines).astype(np.uint32)
    # The baseline flags combine the CAM per-dish data_suspect sensor and input_data_suspect
    baseline_flags = np.zeros((output_baselines,), np.uint8)
    add_random_flags(baseline_flags, CAM, probability=0.01)
    # The timeseries weights / mask are set by the signal display controls via telstate
    timeseries_weights = np.random.choice(2, size=output_channels, p=[0.1, 0.9])
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
    for i in range(len(percentile_ranges)):
        sd_names.append(f"percentile{i}")
    sd_buffers = [proc.buffer(name) for name in sd_names]
    sd_arrays = [buf.empty_like() for buf in sd_buffers]

    print("Generating input visibility data...")
    dumps = [generate_data(vis_in_device, input_channels, input_baselines)
             for _ in range(max(dumps_per_sd_dump, dumps_per_output_dump))]
    # Push data before we start timing, to ensure everything is allocated
    for dump in dumps:
        proc.buffer('vis_in').set(command_queue, dump)

    output_dumps = args.repeat // dumps_per_output_dump
    print(f"Processing {args.repeat} CBF dumps ({output_dumps} SDP dumps)...")
    start_event = command_queue.enqueue_marker()
    proc.start_sum()
    proc.start_sd_sum()
    # Iterate over input CBF dumps
    for pass_ in range(args.repeat):
        if not args.no_transfer:
            proc.buffer('vis_in').set_async(command_queue, dumps[pass_ % len(dumps)])
        proc()
        if (pass_ + 1) % dumps_per_output_dump == 0:
            proc.end_sum()
            if not args.no_transfer:
                for buf, array in zip(output_buffers, output_arrays):
                    buf.get_async(command_queue, array)
            proc.start_sum()
        if (pass_ + 1) % dumps_per_sd_dump == 0:
            proc.end_sd_sum()
            if not args.no_transfer:
                for buf, array in zip(sd_buffers, sd_arrays):
                    buf.get_async(command_queue, array)
            proc.start_sd_sum()
    end_event = command_queue.enqueue_marker()
    elapsed_ms = end_event.time_since(start_event) * 1000.0
    dump_ms = elapsed_ms / args.repeat
    realtime_perc = dump_ms / 1000.0 / args.input_int_time * 100.0
    print(f"Total runtime = {elapsed_ms:.3f} ms ({dump_ms:.3f} ms per input CBF dump)")
    print(f"Processing takes {realtime_perc:.1f}% of real time")


if __name__ == '__main__':
    main()
