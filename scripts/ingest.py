#!/usr/bin/env python

# Capture utility for a relatively generic packetised correlator data output stream.

import spead2
import time
import argparse
import Queue
import logging
import manhole
import signal
import os

from katcp import DeviceServer, Sensor
from katcp.kattypes import request, return_reply, Str, Float

import katsdpingest.sigproc as sp
from katsdpingest.ingest_threads import CAMIngest, CBFIngest
from katsdpsigproc import accel
from katsdptelstate import endpoint
import katsdptelstate

# import katconf

def comma_list(type_):
    """Return a function which splits a string on commas and converts each element to
    `type_`."""

    def convert(arg):
        return [type_(x) for x in arg.split(',')]
    return convert

def parse_opts():
    parser = katsdptelstate.ArgumentParser()
    parser.add_argument('--sdisp-spead', type=endpoint.endpoint_list_parser(7149), default='127.0.0.1:7149', help='signal display destination. Either single ip or comma separated list. [default=%(default)s]', metavar='ENDPOINT')
    parser.add_argument('--cbf-spead', type=endpoint.endpoint_list_parser(7148, single_port=True), default=':7148', help='endpoints to listen for CBF SPEAD stream (including multicast IPs). [<ip>[+<count>]][:port]. [default=%(default)s]', metavar='ENDPOINTS')
    parser.add_argument('--cam-spead', type=endpoint.endpoint_list_parser(7147, single_port=True), default=':7147', help='endpoints to listen for CAM SPEAD stream (including multicast IPs). [<ip>[+<count>]][:port]. [default=%(default)s]', metavar='ENDPOINTS')
    parser.add_argument('--l0-spectral-spead', type=endpoint.endpoint_parser(7200), default='127.0.0.1:7200', help='destination for spectral L0 output. [default=%(default)s]', metavar='ENDPOINT')
    parser.add_argument('--l0-spectral-spead-rate', type=float, default=1000000000, help='rate (bits per second) to transmit spectral L0 output. [default=%(default)s]', metavar='RATE')
    parser.add_argument('--l0-continuum-spead', type=endpoint.endpoint_parser(7201), default='127.0.0.1:7201', help='destination for continuum L0 output. [default=%(default)s]', metavar='ENDPOINT')
    parser.add_argument('--l0-continuum-spead-rate', type=float, default=1000000000, help='rate (bits per second) to transmit continuum L0 output. [default=%(default)s]', metavar='RATE')
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
    Top level holder of the ingest threads and the owner of any output files."""

    VERSION_INFO = ("sdp-ingest", 0, 1)
    BUILD_INFO = ("sdp-ingest", 0, 1, "rc1")

    def __init__(self, logger, sdisp_endpoints, antennas, channels, *args, **kwargs):
        self.logger = logger
        self.cbf_thread = None
         # reference to the CBF ingest thread
        self.cam_thread = None
         # reference to the Telescope Manager thread
        self.obs = None
         # the observation component for holding observation attributes
        self.sdisp_ips = {}
        for endpoint in sdisp_endpoints:
            self.sdisp_ips[endpoint.host] = endpoint.port
         # add default or user specified endpoints
        # compile the device code
        context = accel.create_some_context(interactive=False)
        self.proc_template = CBFIngest.create_proc_template(context, antennas, channels)

        self._my_sensors = {}
        self._my_sensors["capture-active"] = Sensor(Sensor.INTEGER, "capture_active", "Is there a currently active capture thread.","",default=0, params=[0,1])
        self._my_sensors["packets-captured"] = Sensor(Sensor.INTEGER, "packets_captured", "The number of packets captured so far by the current session.","",default=0, params=[0,2**63])
        self._my_sensors["status"] = Sensor.string("status", "The current status of the capture thread.","")
        self._my_sensors["last-dump-timestamp"] = Sensor(Sensor.FLOAT, "last_dump_timestamp","Timestamp of most recently received correlator dump in Unix seconds","",default=0,params=[0,2**63])
        self._my_sensors["input-rate"] = Sensor(Sensor.INTEGER, "input-rate","Input data rate in Bps averaged over the last 10 dumps","Bps",default=0)
        self._my_sensors["output-rate"] = Sensor(Sensor.INTEGER, "output-rate","Output data rate in Bps averaged over the last 10 dumps","Bps",default=0)
        self._my_sensors["device-status"] = Sensor.discrete("device-status", "Health status", "", ["ok", "degraded", "fail"])

        super(IngestDeviceServer, self).__init__(*args, **kwargs)

    def setup_sensors(self):
        for sensor in self._my_sensors:
            self.add_sensor(self._my_sensors[sensor])
            if self._my_sensors[sensor]._sensor_type == Sensor.STRING:
                self._my_sensors[sensor].set_value("")
            if self._my_sensors[sensor]._sensor_type == Sensor.INTEGER:
                self._my_sensors[sensor].set_value(0)
             # take care of basic defaults to ensure sensor status is 'nominal'
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
        if self.cbf_thread is not None: self.cbf_thread.enable_debug(debug)
        if self.cam_thread is not None: self.cam_thread.enable_debug(debug)

    @request(Str(),Str())
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
    def request_capture_start(self, req, msg):
        """Dummy capture start command - calls capture init."""
        self.request_capture_init(req, msg)
        smsg = "Capture initialised at %s" % time.ctime()
        logger.info(smsg)
        return ("ok", smsg)

    @return_reply(Str())
    def request_capture_init(self, req, msg):
        """Spawns ingest threads to capture suitable data and meta-data to produce
        the L0 output stream."""
        if self.cbf_thread is not None:
            return ("fail", "Existing capture session found. If you really want to init, stop the current capture using capture_stop.")
             # this should be enough of an indicator as to session activity, but it 
             # may be worth expanding the scope to checking the CAM thread as well

        self.cbf_thread = CBFIngest(opts, self.proc_template,
                self._my_sensors, opts.telstate, 'cbf', cbf_logger)
        # add in existing signal display recipients...
        for (ip,port) in self.sdisp_ips.iteritems():
            self.cbf_thread.add_sdisp_ip(ip,port)
        self.cbf_thread.start()

        self.cam_thread = CAMIngest(opts.cam_spead, self._my_sensors, opts.telstate, cam_logger)
        self.cam_thread.start()

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
        if self.cbf_thread is None:
            return ("fail","No active capture thread. Please start one using capture_init")
        self.cbf_thread.set_center_freq(center_freq_hz)
        logger.info("Center frequency set to %f Hz", center_freq_hz)
        return ("ok", "set")

    @request(Str())
    @return_reply(Str())
    def request_drop_sdisp_ip(self, req, ip):
        """Drop an IP address from the internal list of signal display data recipients."""
        try:
            del self.sdisp_ips[ip]
        except KeyError:
            return ("fail", "The IP address specified (%s) does not exist in the current list of recipients." % (ip))
        if self.cbf_thread is not None:
            # drop_sdisp_ip can in theory raise KeyError, but the check against
            # our own list prevents that.
            self.cbf_thread.drop_sdisp_ip(ip)
        return ("ok","The IP address has been dropped as a signal display recipient")

    @request(Str())
    @return_reply(Str())
    def request_add_sdisp_ip(self, req, ip):
        """Add the supplied ip and port (ip[:port]) to the list of signal display data recipients.If not port is supplied default of 7149 is used."""
        ipp = ip.split(":")
        ip = ipp[0]
        if len(ipp) > 1: port = int(ipp[1])
        else: port = 7149
        if self.sdisp_ips.has_key(ip):
            return ("ok", "The supplied IP is already in the active list of recipients.")
        self.sdisp_ips[ip] = port
        if self.cbf_thread is not None:
            # add_sdisp_ip can in theory raise KeyError, but the check against
            # our own list prevents that.
            self.cbf_thread.add_sdisp_ip(ip, port)
        return ("ok", "Added IP address %s (port: %i) to list of signal display data recipients." % (ip, port))

    def handle_interrupt(self):
        """Used to attempt a graceful resolution to external
        interrupts. Basically calls capture done."""
        logger.warning("External interrupt called - attempting graceful shutdown.")
        self.request_capture_done("","")

    @return_reply(Str())
    def request_capture_done(self, req, msg):
        """Closes the current capture file and renames it for use by augment."""
        if self.cbf_thread is None:
            return ("fail", "No existing capture session.")

         # if the observation framework is behaving correctly
         # then these threads will be dead before capture_done
         # is called. If not, then we take more drastic action.
        if self.cbf_thread.is_alive():
            # This doesn't take the lock on cbf_thread, but it's safe because
            # the stop method is itself thread-safe.
            self.cbf_thread.rx.stop()
            time.sleep(1)

        if self.cam_thread.is_alive():
            self.cam_thread.rx.stop()
            time.sleep(1)

        self.cbf_thread.join()
        self.cam_thread.join()
         # we really dont want these lurking around
        self.cbf_thread = None
         # we are done with the capture thread
        self.cam_thread = None

        self._my_sensors["capture-active"].set_value(0)
        # Error states were associated with the threads, which are now dead.
        self._my_sensors["device-status"].set_value("ok")
        logger.info("capture complete")
        return ("ok", "capture complete")


if __name__ == '__main__':
    opts = parse_opts()

    # Setup configuration source
    #katconf.set_config(katconf.environ(opts.sysconfig))
    # set up Python logging
    #katconf.configure_logging(opts.logging)

    if len(logging.root.handlers) > 0: logging.root.removeHandler(logging.root.handlers[0])
    formatter = logging.Formatter("%(asctime)s.%(msecs)dZ - %(filename)s:%(lineno)s - %(levelname)s - %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logging.root.addHandler(sh)

    logger = logging.getLogger("katsdpingest.ingest")
    logger.setLevel(logging.INFO)

    logging.getLogger('spead2').setLevel(logging.WARNING)
     # configure SPEAD to display warnings about dropped packets etc...

    cbf_logger = logging.getLogger("katsdpingest.cbf_ingest")
    cbf_logger.setLevel(logging.INFO)
    cbf_logger.info("CBF ingest logging started")

    cam_logger = logging.getLogger("katsdpingest.cam_ingest")
    cam_logger.setLevel(logging.INFO)
    cam_logger.info("CAM ingest logging started")

    restart_queue = Queue.Queue()
    antennas = len(opts.antenna_mask) if opts.antenna_mask else opts.antennas
    server = IngestDeviceServer(logger, opts.sdisp_spead, antennas, opts.cbf_channels,
            opts.host, opts.port)
    server.set_restart_queue(restart_queue)
    server.start()
    logger.info("Started katsdpingest server.")

    manhole.install(oneshot_on='USR1', locals={'server':server, 'opts':opts})
     # allow remote debug connections and expose server and opts

    def graceful_exit(_signo=None, _stack_frame=None):
        logger.info("Exiting ingest on SIGTERM")
        os.kill(os.getpid(), signal.SIGINT)
         # rely on the interrupt handler around the katcp device server
         # to peform graceful shutdown. this preserves the command
         # line Ctrl-C shutdown.

    signal.signal(signal.SIGTERM, graceful_exit)
     # mostly needed for Docker use since this process runs as PID 1
     # and does not get passed sigterm unless it has a custom listener

    try:
        while True:
            try:
                device = restart_queue.get(timeout=0.5)
            except Queue.Empty:
                device = None
            if device is not None:
                logger.info("Stopping")
                device.stop()
                device.join()
                logger.info("Restarting")
                device.start()
                logger.info("Started")
    except KeyboardInterrupt:
        logger.info("Shutting down katsdpingest server...")
        logger.info("Activity logging stopped")
        server.handle_interrupt()
        server.stop()
        server.join()
