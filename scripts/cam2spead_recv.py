#!/usr/bin/env python

"""Capture sensor data sent by katstream and store it in telescope state."""

import spead2
import spead2.recv
import time
import Queue
import logging
import manhole
import signal
import os
import numpy as np
import threading
import socket

from katcp import DeviceServer, Sensor
from katcp.kattypes import request, return_reply, Str

from katsdptelstate import endpoint
import katsdptelstate


def parse_opts():
    parser = katsdptelstate.ArgumentParser()
    parser.add_argument('--cam-spead', type=endpoint.endpoint_list_parser(7147, single_port=True), default=':7147', help='endpoints to listen for CAM SPEAD stream (including multicast IPs). [<ip>[+<count>]][:port]. [default=%(default)s]', metavar='ENDPOINTS')
    parser.add_argument('-p', '--port', dest='port', type=int, default=2041, metavar='N', help='katcp host port. [default=%(default)s]')
    parser.add_argument('-a', '--host', dest='host', type=str, default="", metavar='HOST', help='katcp host address. [default=all hosts]')
    parser.add_argument('-l', '--logging', dest='logging', type=str, default=None, metavar='LOGGING',
                      help='level to use for basic logging or name of logging configuration file; '
                           'default is /log/log.<SITENAME>.conf')
    return parser.parse_args()


class CAMIngest(threading.Thread):
    """The CAM Ingest class receives meta-data updates in the form
    of sensor information from the CAM via SPEAD. It uses these to
    update the telescope state."""
    def __init__(self, spead_endpoints, my_sensors, telstate, logger):
        self.logger = logger
        self.spead_endpoints = spead_endpoints
        self.telstate = telstate
        self._my_sensors = my_sensors
        self.ig = None
        threading.Thread.__init__(self)

    def enable_debug(self, debug):
        """Enable/disable debugging in the internal logger.

        This function is thread-safe (because the logging module is).
        """
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)

    def _update_telstate(self, updated):
        for item_name, item in updated.iteritems():
            self.logger.debug("Sensor update: {} == {}".format(item_name, item.value))
            try:
                (value_time, status, sensor_value) = item.value.split(" ", 2)
                value_time = float(value_time)
                sensor_value = np.safe_eval(sensor_value)
            except ValueError:
                # our update is not a classical sensor triplet of time / status / value
                # fake up a realistic looking sensor
                sensor_value = item.value
                value_time = time.time()
                status = "nominal"
                if sensor_value == '':
                    # TODO: once fixed in numpy remove this check
                    self.logger.error("Not inserting empty string into sensor {} due to existing numpy/pickle bug"
                                      .format(item_name))
                    continue
            if status == 'unknown':
                self.logger.debug("Sensor {0} received update '{1}' with status 'unknown' (ignored)"
                                  .format(item_name, item.value))
            elif self.telstate is not None:
                # XXX Nasty hack to get SDP onto cbf name for AR1 integration
                item_name = item_name.replace('data_1_', 'cbf_')
                self.telstate.add(item_name, sensor_value, value_time)

    def run(self):
        """Thin wrapper around :meth:`_run` which handles exceptions."""
        try:
            self._run()
        except Exception:
            self.logger.error('CAMIngest thread threw an exception', exc_info=True)
            status_sensor = self._my_sensors['device-status']
            if status_sensor.value() != 'fail':
                # If we were already in fail state, then don't raise the level to
                # warning.
                self._my_sensors['device-status'].set_value('degraded', Sensor.WARN)

    def _run(self):
        self.ig = spead2.ItemGroup()
        self.logger.debug("Initalising SPEAD transports at %f" % time.time())
        self.logger.info("CAM SPEAD stream reception on {0}".format(
            [str(x) for x in self.spead_endpoints]))
        # Socket only used for multicast subscription
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        for endpoint in self.spead_endpoints:
            if endpoint.multicast_subscribe(sock):
                self.logger.info("Subscribing to multicast address {0}".format(endpoint.host))
            elif endpoint.host != '':
                self.logger.warning("Ignoring non-multicast address {0}".format(endpoint.host))
        rx_md = spead2.recv.Stream(spead2.ThreadPool(), bug_compat=spead2.BUG_COMPAT_PYSPEAD_0_5_2)
        rx_md.add_udp_reader(self.spead_endpoints[0].port)
        self.rx = rx_md

        for heap in rx_md:
            updated = self.ig.update(heap)
            self._update_telstate(updated)

        self.logger.info("CAM ingest thread complete at %f" % time.time())


class IngestDeviceServer(DeviceServer):
    """Serves the ingest katcp interface.
    Top level holder of the cam2spead thread."""

    VERSION_INFO = ("sdp-cam2spead-recv", 0, 1)
    BUILD_INFO = ("sdp-cam2spead-recv", 0, 1, "rc1")

    def __init__(self, logger, *args, **kwargs):
        self.logger = logger
        self.cam_thread = None
         # reference to the Telescope Manager thread

        self._my_sensors = {}
        self._my_sensors["capture-active"] = Sensor(Sensor.INTEGER, "capture_active", "Is there a currently active capture thread.","",default=0, params=[0,1])
        self._my_sensors["status"] = Sensor.string("status", "The current status of the capture thread.","")
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
        """Spawns cam2spead thread."""
        if self.cam_thread is not None:
            return ("fail", "Existing capture session found. If you really want to init, stop the current capture using capture_stop.")

        self.cam_thread = CAMIngest(opts.cam_spead, self._my_sensors, opts.telstate, cam_logger)
        self.cam_thread.start()

        self._my_sensors["capture-active"].set_value(1)
        smsg = "Capture initialised at %s" % time.ctime()
        logger.info(smsg)
        return ("ok", smsg)

    def handle_interrupt(self):
        """Used to attempt a graceful resolution to external
        interrupts. Basically calls capture done."""
        logger.warning("External interrupt called - attempting graceful shutdown.")
        self.request_capture_done("","")

    @return_reply(Str())
    def request_capture_done(self, req, msg):
        """Shuts down the capture."""
        if self.cam_thread is None:
            return ("fail", "No existing capture session.")

        if self.cam_thread.is_alive():
            self.cam_thread.rx.stop()
            time.sleep(1)

        self.cam_thread.join()
         # we really dont want these lurking around
        self.cam_thread = None

        self._my_sensors["capture-active"].set_value(0)
        # Error states were associated with the threads, which are now dead.
        self._my_sensors["device-status"].set_value("ok")
        logger.info("capture complete")
        return ("ok", "capture complete")


if __name__ == '__main__':
    opts = parse_opts()

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

    cam_logger = logging.getLogger("katsdpingest.cam_ingest")
    cam_logger.setLevel(logging.INFO)
    cam_logger.info("CAM ingest logging started")

    restart_queue = Queue.Queue()
    server = IngestDeviceServer(logger, opts.host, opts.port)
    server.set_restart_queue(restart_queue)
    server.start()
    logger.info("Started cam2spead_recv server.")

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
        logger.info("Shutting down cam2spead_recv server...")
        logger.info("Activity logging stopped")
        server.handle_interrupt()
        server.stop()
        server.join()
