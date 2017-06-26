#!/usr/bin/env python

# Capture utility for a relatively generic packetised correlator data output stream.

import logging
import threading
import time
import sys
import katsdpservices


# This must be as early as possible to intercept all logger registrations
class Logger(logging.getLoggerClass()):
    # Have to include the root logger explicitly, because it is created before
    # we get to call setLoggerClass
    _loggers = {'': logging.getLogger()}
    _lock = threading.RLock()

    def __init__(self, name, level=logging.NOTSET):
        super(Logger, self).__init__(name, level)
        with Logger._lock:
            Logger._loggers[name] = self

    @classmethod
    def get_loggers(cls):
        with cls._lock:
            # Make a copy, which is safe to use outside the lock
            return dict(cls._loggers)

logging.setLoggerClass(Logger)


import manhole
import signal
import trollius
from trollius import From
import tornado.gen
from tornado.platform.asyncio import AsyncIOMainLoop, to_asyncio_future, to_tornado_future

from katcp import DeviceServer, Sensor
from katcp.kattypes import request, return_reply, Str, Float

import katsdpingest
from katsdpingest.ingest_session import CBFIngest, ChannelRanges, BaselineOrdering, get_cbf_attr
from katsdpingest.utils import Range
from katsdpsigproc import accel
from katsdptelstate import endpoint


logger = logging.getLogger("katsdpingest.ingest")


def comma_list(type_):
    """Return a function which splits a string on commas and converts each element to
    `type_`."""

    def convert(arg):
        return [type_(x) for x in arg.split(',')]
    return convert


def range_str(value):
    """Convert a string of the form 'A:B' to a :class:`~katsdpingest.utils.Range`,
    where A and B are integers.
    """
    fields = value.split(':', 1)
    if len(fields) != 2:
        raise ValueError('Invalid range format {}'.format(value))
    else:
        return Range(int(fields[0]), int(fields[1]))


def parse_args():
    parser = katsdpservices.ArgumentParser()
    parser.add_argument('--sdisp-spead', type=endpoint.endpoint_list_parser(7149), default='127.0.0.1:7149', help='signal display destination. Either single ip or comma separated list. [default=%(default)s]', metavar='ENDPOINT')
    parser.add_argument('--cbf-spead', type=endpoint.endpoint_list_parser(7148), default=':7148', help='endpoints to listen for CBF SPEAD stream (including multicast IPs). [<ip>[+<count>]][:port]. [default=%(default)s]', metavar='ENDPOINTS')
    parser.add_argument('--cbf-interface', help='interface to subscribe to for CBF SPEAD data. [default=auto]', metavar='INTERFACE')
    parser.add_argument('--cbf-ibv', action='store_true', help='use ibverbs acceleration for CBF SPEAD data [default=no].')
    parser.add_argument('--l0-spectral-spead', type=endpoint.endpoint_list_parser(7200), default='127.0.0.1:7200', help='destination for spectral L0 output. [default=%(default)s]', metavar='ENDPOINTS')
    parser.add_argument('--l0-spectral-interface', help='interface on which to send spectral L0 output. [default=auto]', metavar='INTERFACE')
    parser.add_argument('--l0-continuum-spead', type=endpoint.endpoint_list_parser(7201), default='127.0.0.1:7201', help='destination for continuum L0 output. [default=%(default)s]', metavar='ENDPOINTS')
    parser.add_argument('--l0-continuum-interface', help='interface on which to send continuum L0 output. [default=auto]', metavar='INTERFACE')
    parser.add_argument('--output-int-time', default=2.0, type=float, help='seconds between output dumps (will be quantised). [default=%(default)s]')
    parser.add_argument('--sd-int-time', default=2.0, type=float, help='seconds between signal display updates (will be quantised). [default=%(default)s]')
    parser.add_argument('--antenna-mask', default=None, type=comma_list(str), help='comma-separated list of antennas to keep. [default=all]')
    parser.add_argument('--output-channels', type=range_str, help='output spectral channels, in format A:B [default=all]')
    parser.add_argument('--sd-output-channels', type=range_str, help='signal display channels, in format A:B [default=all]')
    parser.add_argument('--continuum-factor', default=16, type=int, help='factor by which to reduce number of channels. [default=%(default)s]')
    parser.add_argument('--sd-continuum-factor', default=128, type=int, help='factor by which to reduce number of channels for signal display. [default=%(default)s]')
    parser.add_argument('--sd-spead-rate', type=float, default=1000000000, help='rate (bits per second) to transmit signal display output. [default=%(default)s]')
    parser.add_argument('-p', '--port', type=int, default=2040, metavar='N', help='katcp host port. [default=%(default)s]')
    parser.add_argument('-a', '--host', type=str, default="", metavar='HOST', help='katcp host address. [default=all hosts]')
    parser.add_argument('-l', '--log-level', type=str, default=None, metavar='LEVEL',
                        help='log level to use')
    args = parser.parse_args()
    if args.telstate is None:
        parser.error('argument --telstate is required')
    if args.cbf_ibv and args.cbf_interface is None:
        parser.error('--cbf-ibv requires --cbf-interface')
    return args


class IngestDeviceServer(DeviceServer):
    """Serves the ingest katcp interface.
    Top level holder of the ingest session.

    Parameters
    ----------
    user_args : :class:`argparse.Namespace`
        Command-line arguments
    channel_ranges : :class:`katsdpingest.ingest_session.ChannelRanges`
        Ranges of channels for various parts of the pipeline
    cbf_attr : dict
        CBF stream configuration, as returned by
        :func:`katsdpingest.ingest_session.get_cbf_attr`.
    args, kwargs
        Passed to :class:`katcp.DeviceServer`
    """

    VERSION_INFO = ("sdp-ingest", 0, 1)
    BUILD_INFO = ('katsdpingest',) + tuple(katsdpingest.__version__.split('.', 1)) + ('',)

    def __init__(self, user_args, channel_ranges, cbf_attr, *args, **kwargs):
        self._stopping = False
        # reference to the CBF ingest session
        self.cbf_session = None
        self.sdisp_ips = {}
        self.channel_ranges = channel_ranges
        self.cbf_attr = cbf_attr
        self.user_args = user_args
        # add default or user specified endpoints
        for sdisp_endpoint in user_args.sdisp_spead:
            self.sdisp_ips[sdisp_endpoint.host] = sdisp_endpoint.port
        # compile the device code
        baselines = BaselineOrdering(cbf_attr['bls_ordering'], user_args.antenna_mask)
        percentile_sizes = set(r[1] - r[0] for r in baselines.percentile_ranges)
        context = accel.create_some_context(interactive=False)
        self.proc_template = CBFIngest.create_proc_template(
            context, percentile_sizes, len(channel_ranges.input))

        sensors = [
            Sensor(
                Sensor.INTEGER, "capture-active",
                "Is there a currently active capture session.",
                "", default=0, params=[0, 1]),
            Sensor.string("status", "The current status of the capture session.", ""),
            Sensor(
                Sensor.FLOAT, "last-dump-timestamp",
                "Timestamp of most recently received correlator dump in Unix seconds",
                "", default=0, params=[0, 2**63]),
            Sensor.discrete("device-status", "Health status", "", ["ok", "degraded", "fail"]),
            Sensor(
                Sensor.INTEGER, "input-bytes-total",
                "Number of payload bytes received from CBF in this session",
                "", default=0),
            Sensor(
                Sensor.INTEGER, "input-heaps-total",
                "Number of payload heaps received from CBF in this session",
                "", default=0),
            Sensor(
                Sensor.INTEGER, "input-dumps-total",
                "Number of CBF dumps received in this session",
                "", default=0),
            Sensor(
                Sensor.INTEGER, "output-bytes-total",
                "Number of payload bytes sent on L0 in this session",
                "", default=0),
            Sensor(
                Sensor.INTEGER, "output-heaps-total",
                "Number of payload heaps sent on L0 in this session",
                "", default=0),
            Sensor(
                Sensor.INTEGER, "output-dumps-total",
                "Number of payload dumps sent on L0 in this session",
                "", default=0)
        ]
        self._my_sensors = {sensor.name: sensor for sensor in sensors}

        super(IngestDeviceServer, self).__init__(*args, **kwargs)

    def setup_sensors(self):
        for sensor in self._my_sensors:
            self.add_sensor(self._my_sensors[sensor])
            # take care of basic defaults to ensure sensor status is 'nominal'
            if self._my_sensors[sensor]._sensor_type == Sensor.STRING:
                self._my_sensors[sensor].set_value("")
            if self._my_sensors[sensor]._sensor_type == Sensor.INTEGER:
                self._my_sensors[sensor].set_value(0)
        self._my_sensors["status"].set_value("init")
        self._my_sensors["device-status"].set_value("ok")

    @return_reply(Str())
    def request_enable_debug(self, req, msg):
        """Enable debugging of the ingest process."""
        self._enable_debug(True)
        return("ok", "Debug logging enabled.")

    @return_reply(Str())
    def request_disable_debug(self, req, msg):
        """Disable debugging of the ingest process."""
        self._enable_debug(False)
        return ("ok", "Debug logging disabled.")

    def _enable_debug(self, debug):
        if self.cbf_session is not None:
            self.cbf_session.enable_debug(debug)

    @request(Str(optional=True), Str(optional=True))
    @return_reply(Str())
    def request_internal_log_level(self, req, component, level):
        """
        Set the log level of an internal component to the specified value.
        ?internal-log-level <component> <level>

        If <level> is omitted, the current log level is shown. If component
        is also omitted, the current log levels of all loggers are shown.
        """
        if component is None:
            loggers = Logger.get_loggers()
            for name, logger in sorted(loggers.iteritems()):
                req.inform(name, logging.getLevelName(logger.level))
            return ('ok', '{} logger(s) reported'.format(len(loggers)))
        elif level is None:
            logger = Logger.get_loggers().get(component)
            if logger is None:
                return ('fail', 'Unknown logger component {}'.format(component))
            else:
                req.inform(component, logging.getLevelName(logger.level))
                return ('ok', '1 logger reported')
        else:
            level = level.upper()
            logger = logging.getLogger(component)
            try:
                logger.setLevel(level)
            except ValueError:
                return ("fail", "Unknown log level specified {}".format(level))
            return ("ok", "Log level set to {}".format(level))

    @return_reply(Str())
    @tornado.gen.coroutine
    def request_capture_init(self, req, msg):
        """Spawns ingest session to capture suitable data to produce
        the L0 output stream."""
        if self.cbf_session is not None:
            raise tornado.gen.Return((
                "fail", "Existing capture session found. If you really want to init, stop the current capture using capture_stop."))
        if self._stopping:
            raise tornado.gen.Return((
                "fail", "Cannot start a capture session while ingest is shutting down"))

        self.cbf_session = CBFIngest(
                self.user_args, self.cbf_attr, self.channel_ranges, self.proc_template,
                self._my_sensors, self.user_args.telstate)
        # add in existing signal display recipients...
        for (ip, port) in self.sdisp_ips.iteritems():
            self.cbf_session.add_sdisp_ip(ip, port)
        self.cbf_session.start()

        self._my_sensors["capture-active"].set_value(1)
        smsg = "Capture initialised at %s" % time.ctime()
        logger.info(smsg)
        raise tornado.gen.Return(("ok", smsg))

    @request(Float())
    @return_reply(Str())
    def request_set_center_freq(self, req, center_freq_hz):
        """Set the center freq for use in the signal displays.

        Parameters
        ----------
        center_freq_hz : int
            The current system center frequency in hz
        """
        if self.cbf_session is None:
            return ("fail", "No active capture session. Please start one using capture_init")
        self.cbf_session.set_center_freq(center_freq_hz)
        logger.info("Center frequency set to %f Hz", center_freq_hz)
        return ("ok", "set")

    @request(Str())
    @return_reply(Str())
    @tornado.gen.coroutine
    def request_drop_sdisp_ip(self, req, ip):
        """Drop an IP address from the internal list of signal display data recipients."""
        try:
            del self.sdisp_ips[ip]
        except KeyError:
            raise tornado.gen.Return(
                    ("fail", "The IP address specified (%s) does not exist in the current list of recipients." % (ip)))
        if self.cbf_session is not None:
            # drop_sdisp_ip can in theory raise KeyError, but the check against
            # our own list prevents that.
            yield to_tornado_future(trollius.async(self.cbf_session.drop_sdisp_ip(ip)))
        raise tornado.gen.Return(
                ("ok", "The IP address has been dropped as a signal display recipient"))

    @request(Str())
    @return_reply(Str())
    def request_add_sdisp_ip(self, req, ip):
        """Add the supplied ip and port (ip[:port]) to the list of signal
        display data recipients.If not port is supplied default of 7149 is
        used."""
        ipp = ip.split(":")
        ip = ipp[0]
        if len(ipp) > 1:
            port = int(ipp[1])
        else:
            port = 7149
        if ip in self.sdisp_ips:
            return ("ok", "The supplied IP is already in the active list of recipients.")
        self.sdisp_ips[ip] = port
        if self.cbf_session is not None:
            # add_sdisp_ip can in theory raise KeyError, but the check against
            # our own list prevents that.
            self.cbf_session.add_sdisp_ip(ip, port)
        return ("ok", "Added IP address %s (port: %i) to list of signal display data recipients." % (ip, port))

    @tornado.gen.coroutine
    def handle_interrupt(self):
        """Used to attempt a graceful resolution to external
        interrupts. Basically calls capture done."""
        self._stopping = True   # Prevent a capture-init during shutdown
        if self.cbf_session is not None:
            yield self.request_capture_done("", "")

    @return_reply(Str())
    @tornado.gen.coroutine
    def request_capture_done(self, req, msg):
        """Stops the current capture."""
        session = self.cbf_session
        if session is None:
            raise tornado.gen.Return(("fail", "No existing capture session."))

        yield to_tornado_future(trollius.async(session.stop()))

        # We need to check that cbf_session hasn't changed, because while
        # yielding, another connection may have manipulated it. This check
        # ensures that any particular session is only reported shut down
        # once.
        if self.cbf_session is session:
            self.cbf_session = None
            self._my_sensors["capture-active"].set_value(0)
            # Error states were associated with the session, which is now dead.
            self._my_sensors["device-status"].set_value("ok")
            logger.info("capture complete")
        raise tornado.gen.Return(("ok", "capture complete"))


def on_shutdown(server):
    # Disable the signal handlers, to avoid being unable to kill if there
    # is an exception in the shutdown path.
    for sig in [signal.SIGINT, signal.SIGTERM]:
        trollius.get_event_loop().remove_signal_handler(sig)
    logger.info("Shutting down katsdpingest server...")
    yield From(to_asyncio_future(server.handle_interrupt()))
    yield From(to_asyncio_future(server.stop()))
    trollius.get_event_loop().stop()


def main():
    logging.setLoggerClass(Logger)
    katsdpservices.setup_logging()
    katsdpservices.setup_restart()
    args = parse_args()
    if args.log_level is not None:
        logging.root.setLevel(args.log_level.upper())

    ioloop = AsyncIOMainLoop()
    ioloop.install()
    try:
        cbf_attr = get_cbf_attr(args.telstate, 'cbf')
    except KeyError as error:
        logger.error('Terminating due to catastrophic failure: %s', error.message)
        sys.exit(1)
    cbf_channels = cbf_attr['n_chans']
    if args.output_channels is None:
        args.output_channels = Range(0, cbf_channels)
    if args.sd_output_channels is None:
        args.sd_output_channels = Range(0, cbf_channels)
    # TODO: determine an appropriate value for guard
    channel_ranges = ChannelRanges(
        cbf_channels, args.continuum_factor, args.sd_continuum_factor,
        len(args.cbf_spead), 64, args.output_channels, args.sd_output_channels)
    server = IngestDeviceServer(args, channel_ranges, cbf_attr, args.host, args.port)
    server.set_concurrency_options(thread_safe=False, handler_thread=False)
    server.set_ioloop(ioloop)
    # allow remote debug connections and expose server and args
    manhole.install(oneshot_on='USR1', locals={'server': server, 'args': args})

    trollius.get_event_loop().add_signal_handler(
        signal.SIGINT, lambda: trollius.async(on_shutdown(server)))
    trollius.get_event_loop().add_signal_handler(
        signal.SIGTERM, lambda: trollius.async(on_shutdown(server)))
    ioloop.add_callback(server.start)
    logger.info("Started katsdpingest server.")
    trollius.get_event_loop().run_forever()
    logger.info("Shutdown complete")


if __name__ == '__main__':
    main()
