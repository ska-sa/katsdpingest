#!/usr/bin/env python3

# Capture utility for a relatively generic packetised correlator data output stream.

import logging
import sys
import signal
import asyncio

import katsdpservices
import manhole

from katsdpingest.ingest_session import ChannelRanges, get_cbf_attr
from katsdpingest.utils import Range
from katsdpsigproc import accel
from katsdpingest.ingest_server import IngestDeviceServer
from katsdptelstate import endpoint


logger = logging.getLogger("katsdpingest.ingest")


def comma_list(type_):
    """Return a function which splits a string on commas and converts each element to
    `type_`."""

    def convert(arg):
        return [type_(x) for x in arg.split(',')]
    return convert


def parse_args():
    parser = katsdpservices.ArgumentParser()
    parser.add_argument(
        '--sdisp-spead', type=endpoint.endpoint_list_parser(7149),
        default='127.0.0.1:7149', metavar='ENDPOINT',
        help=('signal display destination. Either single ip or comma-separated list. '
              '[default=%(default)s]'))
    parser.add_argument(
        '--cbf-spead', type=endpoint.endpoint_list_parser(7148),
        default=':7148', metavar='ENDPOINTS',
        help=('endpoints to listen for CBF SPEAD stream (including multicast IPs). '
              '[<ip>[+<count>]][:port]. [default=%(default)s]'))
    parser.add_argument(
        '--cbf-interface', metavar='INTERFACE',
        help='interface to subscribe to for CBF SPEAD data. [default=auto]')
    parser.add_argument(
        '--cbf-ibv', action='store_true',
        help='use ibverbs acceleration for CBF SPEAD data [default=no].')
    parser.add_argument(
        '--cbf-name',
        help='name of the baseline correlation products stream')
    parser.add_argument(
        '--l0-spectral-spead', type=endpoint.endpoint_list_parser(7200), metavar='ENDPOINTS',
        help='destination for spectral L0 output. [default=do not send]')
    parser.add_argument(
        '--l0-spectral-interface', metavar='INTERFACE',
        help='interface on which to send spectral L0 output. [default=auto]')
    parser.add_argument(
        '--l0-spectral-name', default='sdp_l0', metavar='NAME',
        help='telstate name of the spectral output stream')
    parser.add_argument(
        '--l0-continuum-spead', type=endpoint.endpoint_list_parser(7201), metavar='ENDPOINTS',
        help='destination for continuum L0 output. [default=do not send]')
    parser.add_argument(
        '--l0-continuum-interface', metavar='INTERFACE',
        help='interface on which to send continuum L0 output. [default=auto]')
    parser.add_argument(
        '--l0-continuum-name', default='sdp_l0_continuum', metavar='NAME',
        help='telstate name of the continuum output stream')
    parser.add_argument(
        '--output-int-time', default=2.0, type=float,
        help='seconds between output dumps (will be quantised). [default=%(default)s]')
    parser.add_argument(
        '--sd-int-time', default=2.0, type=float,
        help='seconds between signal display updates (will be quantised). [default=%(default)s]')
    parser.add_argument(
        '--antenna-mask', default=None, type=comma_list(str),
        help='comma-separated list of antennas to keep. [default=all]')
    parser.add_argument(
        '--output-channels', type=Range.parse,
        help='output spectral channels, in format A:B [default=all]')
    parser.add_argument(
        '--sd-output-channels', type=Range.parse,
        help='signal display channels, in format A:B [default=all]')
    parser.add_argument(
        '--continuum-factor', default=16, type=int,
        help='factor by which to reduce number of channels. [default=%(default)s]')
    parser.add_argument(
        '--sd-continuum-factor', default=128, type=int,
        help=('factor by which to reduce number of channels for signal display. '
              '[default=%(default)s]'))
    parser.add_argument(
        '--guard-channels', default=64, type=int,
        help='extra channels to use for RFI detection. [default=%(default)s]')
    parser.add_argument(
        '--input-streams', default=1, type=int,
        help='maximum separate streams for receive. [default=%(default)s]')
    parser.add_argument(
        '--input-max-packet-size', default=4608, type=int,
        help='maximum packet size to receive. [default=[%(default)s]')
    parser.add_argument(
        '--input-buffer', default=64 * 1024**2, type=int,
        help='network buffer size ofr input. [default=%(default)s]')
    parser.add_argument(
        '--sd-spead-rate', type=float, default=1000000000,
        help='rate (bits per second) to transmit signal display output. [default=%(default)s]')
    parser.add_argument(
        '--no-excise', dest='excise', action='store_false',
        help='disable excision of flagged data [default=no]')
    parser.add_argument(
        '--servers', type=int, default=1,
        help='number of parallel servers producing the output [default=%(default)s]')
    parser.add_argument(
        '--server-id', type=int, default=1,
        help='index of this server amongst parallel servers (1-based) [default=%(default)s]')
    parser.add_argument(
        '-p', '--port', type=int, default=2040, metavar='N',
        help='katcp host port. [default=%(default)s]')
    parser.add_argument(
        '-a', '--host', type=str, default="", metavar='HOST',
        help='katcp host address. [default=all hosts]')
    parser.add_argument(
        '-l', '--log-level', type=str, default=None, metavar='LEVEL',
        help='log level to use')
    args = parser.parse_args()
    if args.telstate is None:
        parser.error('argument --telstate is required')
    if args.cbf_ibv and args.cbf_interface is None:
        parser.error('--cbf-ibv requires --cbf-interface')
    if args.cbf_name is None:
        parser.error('--cbf-name is required')
    if not 1 <= args.server_id <= args.servers:
        parser.error('--server-id is out of range')
    if args.l0_spectral_spead is None and args.l0_continuum_spead is None:
        parser.error('at least one of --l0-spectral-spead and --l0-continuum-spead must be given')
    return args


async def on_shutdown(server):
    # Disable the signal handlers, to avoid being unable to kill if there
    # is an exception in the shutdown path.
    for sig in [signal.SIGINT, signal.SIGTERM]:
        asyncio.get_event_loop().remove_signal_handler(sig)
    logger.info("Shutting down katsdpingest server...")
    await server.handle_interrupt()
    server.halt()


def main():
    katsdpservices.setup_logging()
    katsdpservices.setup_restart()
    args = parse_args()
    if args.log_level is not None:
        logging.root.setLevel(args.log_level.upper())

    loop = asyncio.get_event_loop()
    try:
        cbf_attr = get_cbf_attr(args.telstate, args.cbf_name)
    except KeyError as error:
        logger.error('Terminating due to catastrophic failure: %s', error.message)
        sys.exit(1)
    cbf_channels = cbf_attr['n_chans']
    if args.output_channels is None:
        args.output_channels = Range(0, cbf_channels)
    if args.sd_output_channels is None:
        args.sd_output_channels = Range(0, cbf_channels)
    # If no continuum product is selected, set continuum factor to 1 since
    # that effectively disables the alignment checks.
    continuum_factor = args.continuum_factor if args.l0_continuum_spead else 1
    # TODO: determine an appropriate value for guard
    channel_ranges = ChannelRanges(
        args.servers, args.server_id - 1,
        cbf_channels, continuum_factor, args.sd_continuum_factor,
        len(args.cbf_spead), args.guard_channels, args.output_channels, args.sd_output_channels)
    context = accel.create_some_context(interactive=False)
    server = IngestDeviceServer(args, channel_ranges, cbf_attr, context, args.host, args.port)
    # allow remote debug connections and expose server and args
    manhole.install(oneshot_on='USR1', locals={'server': server, 'args': args})

    loop.add_signal_handler(signal.SIGINT, lambda: loop.create_task(on_shutdown(server)))
    loop.add_signal_handler(signal.SIGTERM, lambda: loop.create_task(on_shutdown(server)))
    loop.run_until_complete(server.start())
    logger.info("Started katsdpingest server.")
    loop.run_until_complete(server.join())
    logger.info("Shutdown complete")
    loop.close()


if __name__ == '__main__':
    main()
