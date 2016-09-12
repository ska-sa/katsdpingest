#!/usr/bin/env python

# Capture utility for a relatively generic packetised correlator data output stream.

import time
import logging
import manhole
import signal
import trollius
from trollius import From
import tornado.gen
from tornado.platform.asyncio import AsyncIOMainLoop, to_asyncio_future, to_tornado_future

from katcp import DeviceServer, Sensor
from katcp.kattypes import request, return_reply, Str, Float

import katsdpingest
from katsdpingest.ingest_session import CBFIngest, ChannelRanges
from katsdpingest.utils import Range
from katsdpsigproc import accel
from katsdptelstate import endpoint
import katsdptelstate


logger = logging.getLogger("katsdpingest.ingest")
opts = None


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


def parse_opts():
    parser = katsdptelstate.ArgumentParser()
    parser.add_argument('--sdisp-spead', type=endpoint.endpoint_list_parser(7149), default='127.0.0.1:7149', help='signal display destination. Either single ip or comma separated list. [default=%(default)s]', metavar='ENDPOINT')
    parser.add_argument('--cbf-spead', type=endpoint.endpoint_list_parser(7148), default=':7148', help='endpoints to listen for CBF SPEAD stream (including multicast IPs). [<ip>[+<count>]][:port]. [default=%(default)s]', metavar='ENDPOINTS')
    parser.add_argument('--l0-spectral-spead', type=endpoint.endpoint_list_parser(7200), default='127.0.0.1:7200', help='destination for spectral L0 output. [default=%(default)s]', metavar='ENDPOINTS')
    parser.add_argument('--l0-continuum-spead', type=endpoint.endpoint_list_parser(7201), default='127.0.0.1:7201', help='destination for continuum L0 output. [default=%(default)s]', metavar='ENDPOINTS')
    parser.add_argument('--output-int-time', default=2.0, type=float, help='seconds between output dumps (will be quantised). [default=%(default)s]')
    parser.add_argument('--sd-int-time', default=2.0, type=float, help='seconds between signal display updates (will be quantised). [default=%(default)s]')
    parser.add_argument('--antennas', default=2, type=int, help='number of antennas (prior to masking). [default=%(default)s]')
    parser.add_argument('--antenna-mask', default=None, type=comma_list(str), help='comma-separated list of antennas to keep. [default=all]')
    parser.add_argument('--cbf-channels', default=32768, type=int, help='number of channels. [default=%(default)s]')
    parser.add_argument('--output-channels', type=range_str, help='output spectral channels, in format A:B [default=all]')
    parser.add_argument('--sd-output-channels', type=range_str, help='signal display channels, in format A:B [default=all]')
    parser.add_argument('--continuum-factor', default=16, type=int, help='factor by which to reduce number of channels. [default=%(default)s]')
    parser.add_argument('--sd-continuum-factor', default=128, type=int, help='factor by which to reduce number of channels for signal display. [default=%(default)s]')
    parser.add_argument('--sd-spead-rate', type=float, default=1000000000, help='rate (bits per second) to transmit signal display output. [default=%(default)s]')
    parser.add_argument('-p', '--port', type=int, default=2040, metavar='N', help='katcp host port. [default=%(default)s]')
    parser.add_argument('-a', '--host', type=str, default="", metavar='HOST', help='katcp host address. [default=all hosts]')
    parser.add_argument('-l', '--log-level', type=str, default='INFO', metavar='LEVEL',
                        help='log level to use [default=%(default)s]')
    opts = parser.parse_args()
    if opts.output_channels is None:
        opts.output_channels = Range(0, opts.cbf_channels)
    if opts.sd_output_channels is None:
        opts.sd_output_channels = Range(0, opts.cbf_channels)
    return opts


class IngestDeviceServer(DeviceServer):
    """Serves the ingest katcp interface.
    Top level holder of the ingest session.

    Parameters
    ----------
    sdisp_endpoints : list of `katsdptelstate.endpoint.Endpoint`
        Endpoints for signal display data
    antennas : int
        Number of antennas in output
    channel_ranges : :class:`katsdpingest.ingest_session.ChannelRanges`
        Ranges of channels for various parts of the pipeline
    args, kwargs
        Passed to :class:`katcp.DeviceServer`
    """

    VERSION_INFO = ("sdp-ingest", 0, 1)
    BUILD_INFO = ('katsdpingest',) + tuple(katsdpingest.__version__.split('.', 1)) + ('',)

    def __init__(self, sdisp_endpoints, antennas, channel_ranges, *args, **kwargs):
        # reference to the CBF ingest session
        self.cbf_session = None
        self.sdisp_ips = {}
        self.channel_ranges = channel_ranges
        # add default or user specified endpoints
        for sdisp_endpoint in sdisp_endpoints:
            self.sdisp_ips[sdisp_endpoint.host] = sdisp_endpoint.port
        # compile the device code
        context = accel.create_some_context(interactive=False)
        self.proc_template = CBFIngest.create_proc_template(context, antennas, len(channel_ranges.input))

        self._my_sensors = {}
        self._my_sensors["capture-active"] = Sensor(
                Sensor.INTEGER, "capture_active",
                "Is there a currently active capture session.",
                "", default=0, params=[0, 1])
        self._my_sensors["packets-captured"] = Sensor(
                Sensor.INTEGER, "packets_captured",
                "The number of packets captured so far by the current session.",
                "", default=0, params=[0, 2**63])
        self._my_sensors["status"] = Sensor.string(
                "status", "The current status of the capture session.", "")
        self._my_sensors["last-dump-timestamp"] = Sensor(
                Sensor.FLOAT, "last_dump_timestamp",
                "Timestamp of most recently received correlator dump in Unix seconds",
                "", default=0, params=[0, 2**63])
        self._my_sensors["input-rate"] = Sensor(
                Sensor.INTEGER, "input-rate",
                "Input data rate in Bps averaged over the last 10 dumps",
                "Bps", default=0)
        self._my_sensors["output-rate"] = Sensor(
                Sensor.INTEGER, "output-rate",
                "Output data rate in Bps averaged over the last 10 dumps",
                "Bps", default=0)
        self._my_sensors["device-status"] = Sensor.discrete(
                "device-status", "Health status", "", ["ok", "degraded", "fail"])

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

    @request(Str(), Str())
    @return_reply(Str())
    def request_internal_log_level(self, req, component, level):
        """Set the log level of an internal component to the specified value.
           ?internal-log-level <component> <level>
        """
        level = level.upper()
        logger = logging.getLogger(component)
        try:
            logger.setLevel(level)
        except ValueError:
            return ("fail", "Unknown log level specified {}".format(level))
        return ("ok", "Log level set to {}".format(level))

    @return_reply(Str())
    def request_capture_init(self, req, msg):
        """Spawns ingest session to capture suitable data to produce
        the L0 output stream."""
        if self.cbf_session is not None:
            return ("fail", "Existing capture session found. If you really want to init, stop the current capture using capture_stop.")

        self.cbf_session = CBFIngest(
                opts, self.channel_ranges, self.proc_template,
                self._my_sensors, opts.telstate, 'cbf')
        # add in existing signal display recipients...
        for (ip, port) in self.sdisp_ips.iteritems():
            self.cbf_session.add_sdisp_ip(ip, port)
        self.cbf_session.start()

        self._my_sensors["capture-active"].set_value(1)
        smsg = "Capture initialised at %s" % time.ctime()
        logger.info(smsg)
        return ("ok", smsg)

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
        logger.warning("External interrupt called - attempting graceful shutdown.")
        yield self.request_capture_done("", "")

    @return_reply(Str())
    @tornado.gen.coroutine
    def request_capture_done(self, req, msg):
        """Closes the current capture file and renames it for use by augment."""
        if self.cbf_session is None:
            raise tornado.gen.Return(("fail", "No existing capture session."))

        yield to_tornado_future(trollius.async(self.cbf_session.stop()))
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
    global opts
    opts = parse_opts()

    if len(logging.root.handlers) > 0:
        logging.root.removeHandler(logging.root.handlers[0])
    formatter = logging.Formatter("%(asctime)s.%(msecs)dZ - %(filename)s:%(lineno)s - %(levelname)s - %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logging.root.addHandler(sh)
    logging.root.setLevel(opts.log_level)

    ioloop = AsyncIOMainLoop()
    ioloop.install()
    antennas = len(opts.antenna_mask) if opts.antenna_mask else opts.antennas
    # TODO: determine an appropriate value for guard
    channel_ranges = ChannelRanges(
        opts.cbf_channels, opts.continuum_factor, opts.sd_continuum_factor,
        len(opts.cbf_spead), 64, opts.output_channels, opts.sd_output_channels)
    server = IngestDeviceServer(opts.sdisp_spead, antennas,
                                channel_ranges,
                                opts.host, opts.port)
    server.set_concurrency_options(thread_safe=False, handler_thread=False)
    server.set_ioloop(ioloop)
    # allow remote debug connections and expose server and opts
    manhole.install(oneshot_on='USR1', locals={'server': server, 'opts': opts})

    trollius.get_event_loop().add_signal_handler(
        signal.SIGINT, lambda: trollius.async(on_shutdown(server)))
    trollius.get_event_loop().add_signal_handler(
        signal.SIGTERM, lambda: trollius.async(on_shutdown(server)))
    ioloop.add_callback(server.start)
    logger.info("Started katsdpingest server.")
    trollius.get_event_loop().run_forever()
    server.start()
    logger.info("Shutdown complete")


if __name__ == '__main__':
    main()
