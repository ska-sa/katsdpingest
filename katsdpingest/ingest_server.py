"""katcp server for ingest."""

import time
import logging

import trollius
import tornado.gen
from katsdpservices.asyncio import to_tornado_future
from katcp import AsyncDeviceServer, Sensor
from katcp.kattypes import request, return_reply, Str, Float
from katsdptelstate.endpoint import endpoint_parser
from katsdpsigproc import accel

import katsdpingest
from .ingest_session import CBFIngest
from . import receiver


logger = logging.getLogger(__name__)


class IngestDeviceServer(AsyncDeviceServer):
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
    context : :class:`katsdpsigproc.cuda.Context` or :class:`katsdpsigproc.opencl.Context`
        Context in which to compile device code and allocate resources
    args, kwargs
        Passed to :class:`katcp.DeviceServer`
    """

    VERSION_INFO = ("sdp-ingest", 0, 2)
    BUILD_INFO = ('katsdpingest',) + tuple(katsdpingest.__version__.split('.', 1)) + ('',)

    def __init__(self, user_args, channel_ranges, cbf_attr, context, *args, **kwargs):
        self._stopping = False

        sensors = [
            Sensor(Sensor.INTEGER, "output-n-ants", "Number of antennas in L0 stream"),
            Sensor(Sensor.INTEGER, "output-n-inputs", "Number of single-pol signals in L0 stream"),
            Sensor(Sensor.INTEGER, "output-n-bls", "Number of baseline products in L0 stream"),
            Sensor(Sensor.INTEGER, "output-n-chans",
                   "Number of channels this server contributes to L0 spectral stream"),
            Sensor(Sensor.FLOAT, "output-int-time", "Integration time of L0 stream", "s"),
            Sensor(Sensor.BOOLEAN, "capture-active",
                   "Is there a currently active capture session.",
                   default=False, initial_status=Sensor.NOMINAL),
            Sensor(Sensor.DISCRETE, "status", "The current status of the capture session.", "",
                   ["init", "wait-data", "capturing", "complete"],
                   default="init", initial_status=Sensor.NOMINAL),
            Sensor(Sensor.FLOAT, "last-dump-timestamp",
                   "Timestamp of most recently received correlator dump in Unix seconds", "s",
                   default=0.0, initial_status=Sensor.NOMINAL),
            Sensor(Sensor.DISCRETE,
                   "device-status", "Health status", "", ["ok", "degraded", "fail"],
                   default="ok", initial_status=Sensor.NOMINAL),
            Sensor(Sensor.INTEGER, "input-bytes-total",
                   "Number of payload bytes received from CBF in this session",
                   initial_status=Sensor.NOMINAL),
            Sensor(Sensor.INTEGER, "input-heaps-total",
                   "Number of payload heaps received from CBF in this session",
                   initial_status=Sensor.NOMINAL),
            Sensor(Sensor.INTEGER, "input-dumps-total",
                   "Number of CBF dumps received in this session",
                   initial_status=Sensor.NOMINAL),
            Sensor(Sensor.INTEGER, "output-bytes-total",
                   "Number of payload bytes sent on L0 in this session",
                   initial_status=Sensor.NOMINAL),
            Sensor(Sensor.INTEGER, "output-heaps-total",
                   "Number of payload heaps sent on L0 in this session",
                   initial_status=Sensor.NOMINAL),
            Sensor(Sensor.INTEGER, "output-dumps-total",
                   "Number of payload dumps sent on L0 in this session",
                   initial_status=Sensor.NOMINAL),
            Sensor(Sensor.BOOLEAN, "descriptors-received",
                   "Whether the SPEAD descriptors have been received",
                   initial_status=Sensor.NOMINAL)
        ]
        for key, value in receiver.REJECT_HEAP_TYPES.items():
            sensors.append(Sensor(
                Sensor.INTEGER, "input-" + key + "-heaps-total",
                "Number of heaps rejected because " + value,
                initial_status=Sensor.NOMINAL))
        self._my_sensors = {sensor.name: sensor for sensor in sensors}

        # create the device resources
        self.cbf_ingest = CBFIngest(
            user_args, cbf_attr, channel_ranges, context,
            self._my_sensors, user_args.telstate)
        # add default or user specified endpoints
        for sdisp_endpoint in user_args.sdisp_spead:
            self.cbf_ingest.add_sdisp_ip(sdisp_endpoint)

        super(IngestDeviceServer, self).__init__(*args, **kwargs)

    def setup_sensors(self):
        for sensor in self._my_sensors:
            self.add_sensor(self._my_sensors[sensor])

    @return_reply(Str())
    def request_enable_debug(self, req, msg):
        """Enable debugging of the ingest process."""
        self.cbf_ingest.enable_debug(True)
        return("ok", "Debug logging enabled.")

    @return_reply(Str())
    def request_disable_debug(self, req, msg):
        """Disable debugging of the ingest process."""
        self.cbf_ingest.enable_debug(False)
        return ("ok", "Debug logging disabled.")

    @request(Str(optional=True), Str(optional=True))
    @return_reply(Str())
    def request_internal_log_level(self, req, component, level):
        """
        Set the log level of an internal component to the specified value.
        ?internal-log-level <component> <level>

        If <level> is omitted, the current log level is shown. If component
        is also omitted, the current log levels of all loggers are shown.
        """
        # Filter out placeholders
        loggers = {name: logger for (name, logger) in logging.Logger.manager.loggerDict.items()
                   if isinstance(logger, logging.Logger)}
        loggers[''] = logging.getLogger()   # Not kept in loggerDict
        if component is None:
            for name, logger in sorted(loggers.iteritems()):
                req.inform(name, logging.getLevelName(logger.level))
            return ('ok', '{} logger(s) reported'.format(len(loggers)))
        elif level is None:
            logger = loggers.get(component)
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

    @request(Str())
    @return_reply(Str())
    @tornado.gen.coroutine
    def request_capture_init(self, req, capture_block_id):
        """Spawns ingest session to capture suitable data to produce
        the L0 output stream."""
        if self.cbf_ingest.capturing:
            raise tornado.gen.Return((
                "fail", "Existing capture session found. If you really want to init, stop the current capture using capture-done."))
        if self._stopping:
            raise tornado.gen.Return((
                "fail", "Cannot start a capture session while ingest is shutting down"))

        self.cbf_ingest.start(capture_block_id)

        self._my_sensors["capture-active"].set_value(1)
        smsg = "Capture initialised at %s" % time.ctime()
        logger.info(smsg)
        raise tornado.gen.Return(("ok", smsg))

    @request(Str())
    @return_reply(Str())
    @tornado.gen.coroutine
    def request_drop_sdisp_ip(self, req, ip):
        """Drop an IP address from the internal list of signal display data recipients."""
        try:
            yield to_tornado_future(trollius.async(self.cbf_ingest.drop_sdisp_ip(ip)))
        except KeyError:
            raise tornado.gen.Return(
                    ("fail", "The IP address specified (%s) does not exist in the current list of recipients." % (ip)))
        else:
            raise tornado.gen.Return(
                    ("ok", "The IP address has been dropped as a signal display recipient"))

    @request(Str())
    @return_reply(Str())
    def request_add_sdisp_ip(self, req, ip):
        """Add the supplied ip and port (ip[:port]) to the list of signal
        display data recipients.If not port is supplied default of 7149 is
        used."""
        endpoint = endpoint_parser(7149)(ip)
        try:
            self.cbf_ingest.add_sdisp_ip(endpoint)
        except ValueError:
            return ("ok", "The supplied IP is already in the active list of recipients.")
        else:
            return ("ok", "Added {} to list of signal display data recipients.".format(endpoint))

    @tornado.gen.coroutine
    def handle_interrupt(self):
        """Used to attempt a graceful resolution to external
        interrupts. Basically calls capture done."""
        self._stopping = True   # Prevent a capture-init during shutdown
        if self.cbf_ingest.capturing:
            yield self.request_capture_done("", "")
        self.cbf_ingest.close()

    @return_reply(Str())
    @tornado.gen.coroutine
    def request_capture_done(self, req, msg):
        """Stops the current capture."""
        if not self.cbf_ingest.capturing:
            raise tornado.gen.Return(("fail", "No existing capture session."))

        stopped = yield to_tornado_future(trollius.async(self.cbf_ingest.stop()))

        # In the case of concurrent connections, we need to ensure that we
        # were the one that actually did the stop, as another connection may
        # have raced us to stop and then started a new session.
        if stopped:
            self._my_sensors["capture-active"].set_value(0)
            # Error states were associated with the session, which is now dead.
            self._my_sensors["device-status"].set_value("ok")
            logger.info("capture complete")
        raise tornado.gen.Return(("ok", "capture complete"))
