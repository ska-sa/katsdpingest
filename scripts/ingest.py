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
from katsdpingest.ingest_session import CBFIngest
from katsdpsigproc import accel
from katsdptelstate import endpoint
import katsdptelstate


logger = logging.getLogger("katsdpingest.ingest")
cbf_logger = logging.getLogger("katsdpingest.cbf_ingest")
opts = None


def comma_list(type_):
    """Return a function which splits a string on commas and converts each element to
    `type_`."""

    def convert(arg):
        return [type_(x) for x in arg.split(',')]
    return convert


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
    parser.add_argument('--continuum-factor', default=16, type=int, help='factor by which to reduce number of channels. [default=%(default)s]')
    parser.add_argument('--sd-continuum-factor', default=128, type=int, help='factor by which to reduce number of channels for signal display. [default=%(default)s]')
    parser.add_argument('--sd-spead-rate', type=float, default=1000000000, help='rate (bits per second) to transmit signal display output. [default=%(default)s]')
    parser.add_argument('-p', '--port', dest='port', type=int, default=2040, metavar='N', help='katcp host port. [default=%(default)s]')
    parser.add_argument('-a', '--host', dest='host', type=str, default="", metavar='HOST', help='katcp host address. [default=all hosts]')
    parser.add_argument('-l', '--logging', dest='logging', type=str, default=None, metavar='LOGGING',
                        help='level to use for basic logging or name of logging configuration file; '
                        'default is /log/log.<SITENAME>.conf')
    return parser.parse_args()


class IngestDeviceServer(DeviceServer):
    """Serves the ingest katcp interface.
    Top level holder of the ingest session and the owner of any output files."""

    VERSION_INFO = ("sdp-ingest", 0, 1)
    BUILD_INFO = ('katsdpingest',) + tuple(katsdpingest.__version__.split('.', 1)) + ('',)

    def __init__(self, logger, sdisp_endpoints, antennas, channels, *args, **kwargs):
        self.logger = logger
        # reference to the CBF ingest session
        self.cbf_session = None
        self.sdisp_ips = {}
        # add default or user specified endpoints
        for sdisp_endpoint in sdisp_endpoints:
            self.sdisp_ips[sdisp_endpoint.host] = sdisp_endpoint.port
        # compile the device code
        context = accel.create_some_context(interactive=False)
        self.proc_template = CBFIngest.create_proc_template(context, antennas, channels)

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
                opts, self.proc_template,
                self._my_sensors, opts.telstate, 'cbf', cbf_logger)
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
        trollius.get_event_loop().remove_signal_handler(sig)
    logger = logging.getLogger("katsdpingest.ingest")
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

    logger.setLevel(logging.INFO)

    # configure SPEAD to display warnings about dropped packets etc...
    logging.getLogger('spead2').setLevel(logging.WARNING)

    cbf_logger.setLevel(logging.INFO)
    cbf_logger.info("CBF ingest logging started")

    ioloop = AsyncIOMainLoop()
    ioloop.install()
    antennas = len(opts.antenna_mask) if opts.antenna_mask else opts.antennas
    server = IngestDeviceServer(logger, opts.sdisp_spead, antennas, opts.cbf_channels,
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
