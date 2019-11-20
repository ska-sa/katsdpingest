"""katcp server for ingest."""

import time
import logging
import argparse
from typing import List, Dict, Mapping, Any, cast   # noqa: F401

import aiokatcp
from aiokatcp import FailReply, SensorSampler
from katsdptelstate.endpoint import endpoint_parser

import katsdpingest
from .ingest_session import CBFIngest, Status, DeviceStatus, ChannelRanges
from . import receiver
from .utils import Sensor


logger = logging.getLogger(__name__)


def _warn_if_positive(value: float) -> aiokatcp.Sensor.Status:
    """Status function for sensors that count problems"""
    return Sensor.Status.WARN if value > 0 else Sensor.Status.NOMINAL


def _device_status_status(value: DeviceStatus) -> aiokatcp.Sensor.Status:
    """Sets katcp status for device-status sensor from value"""
    if value == DeviceStatus.OK:
        return Sensor.Status.NOMINAL
    elif value == DeviceStatus.DEGRADED:
        return Sensor.Status.WARN
    else:
        return Sensor.Status.ERROR


class IngestDeviceServer(aiokatcp.DeviceServer):
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
        Passed to :class:`aiokatcp.DeviceServer`
    """

    VERSION = "sdp-ingest-0.2"
    BUILD_STATE = 'katsdpingest-' + katsdpingest.__version__

    def __init__(
            self,
            user_args: argparse.Namespace,
            channel_ranges: ChannelRanges,
            cbf_attr: Dict[str, Any],
            context, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._stopping = False

        def counter(name: str, description: str, *,
                    event_rate: bool = False,
                    warn_if_positive: bool = False,
                    **kwargs: Any) -> Sensor:
            if event_rate:
                kwargs['auto_strategy'] = SensorSampler.Strategy.EVENT_RATE
                kwargs['auto_strategy_parameters'] = (0.05, 10.0)
            if warn_if_positive:
                kwargs['status_func'] = _warn_if_positive
            return Sensor(int, name, description + ' (prometheus: counter)',
                          initial_status=Sensor.Status.NOMINAL, **kwargs)

        sensors = [
            Sensor(int, "output-n-ants",
                   "Number of antennas in L0 stream (prometheus: gauge)"),
            Sensor(int, "output-n-inputs",
                   "Number of single-pol signals in L0 stream (prometheus: gauge)"),
            Sensor(int, "output-n-bls",
                   "Number of baseline products in L0 stream (prometheus: gauge)"),
            Sensor(int, "output-n-chans",
                   "Number of channels this server contributes to L0 spectral stream "
                   "(prometheus: gauge)"),
            Sensor(float, "output-int-time",
                   "Integration time of L0 stream (prometheus: gauge)", "s"),
            Sensor(bool, "capture-active",
                   "Is there a currently active capture session (prometheus: gauge)",
                   default=False, initial_status=Sensor.Status.NOMINAL),
            Sensor(Status, "status",
                   "The current status of the capture session.",
                   default=Status.INIT, initial_status=Sensor.Status.NOMINAL),
            Sensor(float, "last-dump-timestamp",
                   "Timestamp of most recently received correlator dump in Unix seconds "
                   "(prometheus: gauge)", "s",
                   default=0.0, initial_status=Sensor.Status.NOMINAL),
            Sensor(DeviceStatus, "device-status",
                   "Health status",
                   default=DeviceStatus.OK, initial_status=Sensor.Status.NOMINAL,
                   status_func=_device_status_status),
            counter("input-bytes-total",
                    "Number of payload bytes received from CBF in this session",
                    event_rate=True),
            counter("input-heaps-total",
                    "Number of payload heaps received from CBF in this session",
                    event_rate=True),
            counter("input-dumps-total",
                    "Number of CBF dumps received in this session",
                    event_rate=True),
            counter("input-metadata-heaps-total",
                    "Number of heaps that do not contain payload in this session",
                    event_rate=True),
            counter("output-bytes-total",
                    "Number of payload bytes sent on L0 in this session"),
            counter("output-heaps-total",
                    "Number of payload heaps sent on L0 in this session"),
            counter("output-dumps-total",
                    "Number of payload dumps sent on L0 in this session"),
            counter("output-vis-total",
                    "Number of spectral visibilities computed for signal displays in this session"),
            counter("output-flagged-total",
                    "Number of flagged visibilities (out of output-vis-total)"),
            Sensor(bool, "descriptors-received",
                   "Whether the SPEAD descriptors have been received "
                   " (prometheus: gauge)",
                   initial_status=Sensor.Status.NOMINAL)
        ]   # type: List[Sensor]
        for key, value in receiver.REJECT_HEAP_TYPES.items():
            sensors.append(counter(
                "input-" + key + "-heaps-total",
                "Number of heaps rejected because {}".format(value),
                event_rate=True, warn_if_positive=True))
        for sensor in sensors:
            self.sensors.add(sensor)

        # create the device resources
        self.cbf_ingest = CBFIngest(
            user_args, cbf_attr, channel_ranges, context,
            cast(Mapping[str, Sensor], self.sensors), user_args.telstate)
        # add default or user specified endpoints
        for sdisp_endpoint in user_args.sdisp_spead:
            self.cbf_ingest.add_sdisp_ip(sdisp_endpoint)

    async def request_enable_debug(self, ctx) -> str:
        """Enable debugging of the ingest process."""
        self.cbf_ingest.enable_debug(True)
        return "Debug logging enabled."

    async def request_disable_debug(self, ctx) -> str:
        """Disable debugging of the ingest process."""
        self.cbf_ingest.enable_debug(False)
        return "Debug logging disabled."

    async def request_internal_log_level(self, ctx,
                                         component: str = None, level: str = None) -> None:
        """
        Set the log level of an internal component to the specified value.
        ?internal-log-level <component> <level>

        If <level> is omitted, the current log level is shown. If component
        is also omitted, the current log levels of all loggers are shown.
        """
        # manager isn't part of the documented API, so mypy/typeshed doesn't know about it
        manager = logging.Logger.manager       # type: ignore
        # Filter out placeholders
        loggers = {name: logger for (name, logger) in manager.loggerDict.items()
                   if isinstance(logger, logging.Logger)}
        loggers[''] = logging.getLogger()   # Not kept in loggerDict
        if component is None:
            ctx.informs((name, logging.getLevelName(logger.level))
                        for name, logger in sorted(loggers.items()))
        elif level is None:
            logger = loggers.get(component)
            if logger is None:
                raise FailReply('Unknown logger component {}'.format(component))
            else:
                ctx.informs([(component, logging.getLevelName(logger.level))])
        else:
            level = level.upper()
            logger = logging.getLogger(component)
            try:
                logger.setLevel(level)
            except ValueError:
                raise FailReply("Unknown log level specified {}".format(level))

    async def request_capture_init(self, ctx, capture_block_id: str) -> str:
        """Spawns ingest session to capture suitable data to produce
        the L0 output stream."""
        if self.cbf_ingest.capturing:
            raise FailReply(
                "Existing capture session found. "
                "If you really want to init, stop the current capture using capture-done.")
        if self._stopping:
            raise FailReply("Cannot start a capture session while ingest is shutting down")

        self.cbf_ingest.start(capture_block_id)

        self.sensors["capture-active"].value = True
        smsg = "Capture initialised at %s" % time.ctime()
        logger.info(smsg)
        return smsg

    async def request_drop_sdisp_ip(self, ctx, ip: str) -> str:
        """Drop an IP address from the internal list of signal display data recipients."""
        try:
            await self.cbf_ingest.drop_sdisp_ip(ip)
        except KeyError:
            raise FailReply(
                "The IP address specified ({}) does not exist "
                "in the current list of recipients.".format(ip))
        else:
            return "The IP address has been dropped as a signal display recipient"

    async def request_add_sdisp_ip(self, ctx, ip: str) -> str:
        """Add the supplied ip and port (ip[:port]) to the list of signal
        display data recipients.If not port is supplied default of 7149 is
        used."""
        endpoint = endpoint_parser(7149)(ip)
        try:
            self.cbf_ingest.add_sdisp_ip(endpoint)
        except ValueError:
            return "The supplied IP is already in the active list of recipients."
        else:
            return "Added {} to list of signal display data recipients.".format(endpoint)

    async def handle_interrupt(self) -> None:
        """Used to attempt a graceful resolution to external
        interrupts. Basically calls capture done."""
        self._stopping = True   # Prevent a capture-init during shutdown
        if self.cbf_ingest.capturing:
            await self.request_capture_done(None)
        self.cbf_ingest.close()

    async def request_capture_done(self, ctx) -> str:
        """Stops the current capture."""
        if not self.cbf_ingest.capturing:
            raise FailReply("fail", "No existing capture session.")

        stopped = await self.cbf_ingest.stop()

        # In the case of concurrent connections, we need to ensure that we
        # were the one that actually did the stop, as another connection may
        # have raced us to stop and then started a new session.
        if stopped:
            self.sensors["capture-active"].value = False
            # Error states were associated with the session, which is now dead.
            self.sensors["device-status"].value = DeviceStatus.OK
            logger.info("capture complete")
        return "capture complete"
