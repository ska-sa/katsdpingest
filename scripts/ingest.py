#!/usr/bin/python

# Capture utility for a relatively generic packetised correlator data output stream.

# The script performs two primary roles:
#
# Storage of stream data on disk in hdf5 format. This includes merging incoming meta-data with the correlator data
# stream to produce a complete, packaged hdf5 file.
#
# Regeneration of a SPEAD stream suitable for use in the online signal displays. At the moment this is basically
# just an aggregate of the incoming streams from the multiple x engines scaled with n_accumulations (if set)

import spead64_40 as spead40
import spead64_48 as spead
import sys
import time
import optparse
import Queue
import logging
import logging.handlers
import os

from katcp import DeviceServer, Sensor
from katcp.kattypes import request, return_reply, Str, Float

import katsdpingest.sigproc as sp
from katsdpingest.ingest_threads import CAMIngest, CBFIngest

 # import model components. In the future this may be done by the sdp_proxy and the 
 # complete model passed in.
from katsdpingest.telescope_model import AntennaPositioner, CorrelatorBeamformer, Enviro, TelescopeModel, Observation

# import katconf


def parse_opts(argv):
    parser = optparse.OptionParser()
    parser.add_option('-w', '--working-folder', dest='workpath', default=os.path.join("/", "var", "kat", "sdpcontroller"), metavar='WORKING_PATH', help='folder to write process standard out logs into (default=%default)')
    parser.add_option('--sdisp-ips', default='127.0.0.1', help='default signal display destination ip addresses. Either single ip or comma separated list. [default=%default]')
    parser.add_option('--sdisp-port', default='7149',type=int, help='port on which to send signal display data. [default=%default]')
    parser.add_option('--cbf-spead-port', default=7148, type=int, help='default port to receive CBF SPEAD stream on')
    parser.add_option('--cbf-spead-host', default='127.0.0.1', help='default host to receive CBF SPEAD stream from, may be multicast or unicast. <ip>[+<count>]. [default=%default]')
    parser.add_option('--cam-spead-port', default=7147, type=int, help='port to receive CAM SPEAD stream on')
    parser.add_option('--cam-spead-host', default='127.0.0.1', help='default host to receive CAM SPEAD stream from, may be multicast or unicast. <ip>[+<count>]. [default=%default]')
    parser.add_option('--file-base', default='/var/kat/data/staging', help='base directory into which to write HDF5 files. [default=%default]')
    parser.add_option('--antennas', default='m063,m062', help='Comma seperated list of antennas to record to file. Other antennas visibility data is discarded. [default=%default]')
    parser.add_option('-p', '--port', dest='port', type=long, default=2040, metavar='N', help='katcp host port. [default=%default]')
    parser.add_option('-a', '--host', dest='host', type="string", default="", metavar='HOST', help='katcp host address. [default="" (all hosts)]')
    parser.add_option('-l', '--loglevel', dest='loglevel', type="string", default="INFO", metavar='LOGLEVEL', help='set the Python logging level (default=%default)')
    return parser.parse_args(argv)


class IngestDeviceServer(DeviceServer):
    """Serves the ingest katcp interface.
    Top level holder of the ingest threads and the owner of any output files."""

    VERSION_INFO = ("sdp-ingest", 0, 1)
    BUILD_INFO = ("sdp-ingest", 0, 1, "rc1")

    def __init__(self, logger, sdisp_ips, sdisp_port, antennas, *args, **kwargs):
        self.logger = logger
        self.antennas = antennas
        self.cbf_thread = None
         # reference to the CBF ingest thread
        self.cam_thread = None
         # reference to the Telescope Manager thread
        self.h5_file = None
         # the current hdf5 file in use by the ingest threads
        self.model = None
         # the current telescope model for use in this ingest session
        self.obs = None
         # the observation component for holding observation attributes
        self.sdisp_ips = {}
        self.sdisp_ips['127.0.0.1'] = sdisp_port
         # add default signal display destination
        for ip in sdisp_ips.split(","):
            self.sdisp_ips[ip] = sdisp_port
         # add additional user specified ip
        self._my_sensors = {}
        self._my_sensors["capture-active"] = Sensor(Sensor.INTEGER, "capture_active", "Is there a currently active capture thread.","",default=0, params=[0,1])
        self._my_sensors["packets-captured"] = Sensor(Sensor.INTEGER, "packets_captured", "The number of packets captured so far by the current session.","",default=0, params=[0,2**63])
        self._my_sensors["status"] = Sensor.string("status", "The current status of the capture thread.","")

        self._my_sensors["obs-ants"] = Sensor.string("script-ants","The antennas specified by the user for use by the executed script.","")
        self._my_sensors["obs-script-name"] = Sensor.string("script-name", "Current script name", "")
        self._my_sensors["obs-experiment-id"] = Sensor.string("script-experiment-id", "Current experiment id", "")
        self._my_sensors["obs-observer"] = Sensor.string("script-observer", "Current experiment observer", "")
        self._my_sensors["obs-description"] = Sensor.string("script-description", "Current experiment description", "")
        self._my_sensors["obs-script-arguments"] = Sensor.string("script-arguments", "Options and parameters of script - from sys.argv", "")
        self._my_sensors["obs-status"] = Sensor.string("script-status", "Current status reported by running script", "")
        self._my_sensors["obs-starttime"] = Sensor.string("script-starttime", "Start time of current script", "")
        self._my_sensors["obs-endtime"] = Sensor.string("script-endtime", "End time of current script", "")

        self._my_sensors["spead-num-chans"] = Sensor(Sensor.INTEGER, "spead_num_chans","Number of channels reported via SPEAD header from the DBE","",default=0,params=[0,2**63])
        self._my_sensors["spead-num-bls"] = Sensor(Sensor.INTEGER, "spead_num_bls","Number of baselines reported via SPEAD header from the DBE","",default=0,params=[0,2**63])
        self._my_sensors["spead-dump-period"] = Sensor(Sensor.FLOAT, "spead_dump_period","Dump period reported via SPEAD header from the DBE","",default=0,params=[0,2**31])
        self._my_sensors["spead-accum-per-dump"] = Sensor(Sensor.INTEGER, "spead_accum_per_dump","Accumulations per dump reported via SPEAD header from the DBE","",default=0,params=[0,2**63])
        self._my_sensors["spead-center-freq"] = Sensor(Sensor.FLOAT, "spead_center_freq","Center frequency of correlator reported via SPEAD header","",default=0,params=[0,2**31])

        self._my_sensors["last-dump-timestamp"] = Sensor(Sensor.FLOAT, "last_dump_timestamp","Timestamp of most recently received correlator dump in Unix seconds","",default=0,params=[0,2**63])

        super(IngestDeviceServer, self).__init__(*args, **kwargs)

    def setup_sensors(self):
        for sensor in self._my_sensors:
            self.add_sensor(self._my_sensors[sensor])
            if sensor.startswith("spead"):
                self._my_sensors[sensor].set_value(0,status=Sensor.UNKNOWN)
                 # set all SPEAD sensors to unknown at start
                continue
            if self._my_sensors[sensor]._sensor_type == Sensor.STRING:
                self._my_sensors[sensor].set_value("")
            if self._my_sensors[sensor]._sensor_type == Sensor.INTEGER:
                self._my_sensors[sensor].set_value(0)
             # take care of basic defaults to ensure sensor status is 'nominal'
        self._my_sensors["status"].set_value("init")

    @return_reply(Str())
    def request_sd_metadata_issue(self, req, msg):
        """Resend the signal display metadata packets..."""
        if self.cbf_thread is None: return ("fail","No active capture thread. Please start one using capture_init or via a schedule block.")
        self.cbf_thread.send_sd_metadata()
        smsg = "SD Metadata resent"
        logger.info(smsg)
        return ("ok", smsg)

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
        if self.model is not None: self.model.enable_debug(debug)
        if self.cbf_thread is not None: self.cbf_thread.enable_debug(debug)
        if self.cam_thread is not None: self.cam_thread.enable_debug(debug)

    @return_reply(Str())
    def request_capture_start(self, req, msg):
        """Dummy capture start command - calls capture init."""
        self.request_capture_init(req, msg)
        smsg = "Capture initialised at %s" % time.ctime()
        logger.info(smsg)
        return ("ok", smsg)

    @return_reply(Str())
    def request_capture_init(self, req, msg):
        """Creates a telesacope model suitable for use in the current subarray.
        Opens an HDF5 file for use by the ingest threads.
        Then spawns ingest threads to capture suitable data and meta-data to produce
        an archive ready HDF5 file."""
        if self.cbf_thread is not None:
            return ("fail", "Existing capture session found. If you really want to init, stop the current capture using capture_stop.")
             # this should be enough of an indicator as to session activity, but it 
             # may be worth expanding the scope to checking the file and the CAM thread as well

        # for RTS we build a standard model. Normally this would be provided by the sdp_proxy
        cbf = CorrelatorBeamformer(name='data_rts')
        env = Enviro(name='anc_asc')
        self.obs = Observation(name='obs')
        self.model = TelescopeModel()
        self.model.add_components([cbf,env,self.obs])

        for a in self.antennas.split(","):
            self.model.add_components([AntennaPositioner(name=a)])

        self.model.build_index()

        fname = "{0}/{1}.writing.h5".format(opts.file_base, str(int(time.time())))
        self.h5_file = self.model.create_h5_file(fname)
         # open a new HDF5 file
        if self.h5_file is None:
            return ("fail","Failed to create HDF5 file. Init failed.")

        self.cbf_thread = CBFIngest(opts.cbf_spead_host, opts.cbf_spead_port, self.h5_file, self._my_sensors, self.model, cbf.name, self.antennas)
        self.cbf_thread.start()

        self.cam_thread = CAMIngest(opts.cam_spead_host, opts.cam_spead_port, self.h5_file, self.model)
        self.cam_thread.start()

        self._my_sensors["capture-active"].set_value(1)
         # add in existing signal display recipients...
        for (ip,port) in self.sdisp_ips.items():
            self.cbf_thread.add_sdisp_ip(ip,port)
        smsg =  "Capture initialised at %s" % time.ctime()
        logger.info(smsg)
        return ("ok", smsg)

    @request()
    @return_reply(Str())
    def request_enable_van_vleck(self, req):
        """Enable Van Vleck correction of the auto-correlated visibilities."""

    @request(Str(), Str())
    @return_reply(Str())
    def request_set_obs_param(self, req, key_string, value_string):
        """Write a key/value observation parameter to the output file.

        Parameters
        ----------
        key_string : str
            The name of the observation parameter.
        value_string : str
            A string containing the value of the observation parameter.

        Returns
        -------
        success : {'ok', 'fail'}
            Whether storing the key/value pair succeeded.

        Examples
        --------
        ?set_obs_param script-name Test
        !set_obs_param ok script-name=Test

        """
        if self.cbf_thread is None: return ("fail","No active capture thread. Please start one using capture_init")
        try:
            self.obs.set_attribute(key_string, value_string)
        except ValueError, e:
            return ("fail", "Could not set attribute '%s=%s': %s" % (key_string, value_string, e))
        smsg = "%s=%s" % (key_string, value_string)
        logger.info("Set obs param %s" % smsg)
        return ("ok", smsg)

    @request(Str())
    @return_reply(Str())
    def request_drop_sdisp_ip(self, req, ip):
        """Drop an IP address from the internal list of signal display data recipients."""
        if self.cbf_thread is not None:
            if not self.sdisp_ips.has_key(ip):
                return ("fail","The IP address specified (%s) does not exist in the current list of recipients." % (ip))
            del self.sdisp_ips[ip]
            self.cbf_thread.drop_sdisp_ip(ip)
            return ("ok","The IP address has been dropped as a signal display recipient")
        return ("fail","No active capture thread.")

    @request(Str())
    @return_reply(Str())
    def request_add_sdisp_ip(self, req, ip):
        """Add the supplied ip and port (ip[:port]) to the list of signal display data recipients.If not port is supplied default of 7149 is used."""
        ipp = ip.split(":")
        ip = ipp[0]
        if len(ipp) > 1: port = int(ipp[1])
        else: port = 7149
        if self.sdisp_ips.has_key(ip): return ("ok","The supplied IP is already in the active list of recipients.")
        self.sdisp_ips[ip] = port
        if self.cbf_thread is not None:
            self.cbf_thread.add_sdisp_ip(ip, port)
        return ("ok","Added IP address %s (port: %i) to list of signal display data recipients." % (ip, port))

    @return_reply(Str())
    def request_get_current_file(self, req, msg):
        """Return the name of the current (or most recent) capture file."""
        if self.h5_file is None:
            return ("fail", "No currently active file.")
        return ("ok", self.h5_file.filename)

    def handle_interrupt(self):
        """Used to attempt a graceful resolution to external
        interrupts. Basically calls capture done."""
        logger.warning("External interrupt called - attempting graceful shutdown.")
        self.request_capture_done("","")

    @return_reply(Str())
    def request_capture_done(self, req, msg):
        """Closes the current capture file and renames it for use by augment."""
        if self.cbf_thread is None:
            return ("fail","No existing capture session.")
        logger.debug("Capture done called. Thread status is ({},{})".format(self.cbf_thread.is_alive(), self.cam_thread.is_alive()))

         # if the observation framework is behaving correctly
         # then these threads will be dead before capture_done
         # is called. If not, then we take more drastic action.
        if self.cbf_thread.is_alive():
            logger.info("Sending forced stop to CBF thread on port {}".format(opts.cbf_spead_port))
            tx = spead.Transmitter(spead.TransportUDPtx('localhost',opts.cbf_spead_port))
            tx.end()
            time.sleep(1)

        if self.cam_thread.is_alive():
            logger.info("Sending forced stop to CAM thread on port {}".format(opts.cam_spead_port))
            tx = spead40.Transmitter(spead.TransportUDPtx('localhost',opts.cam_spead_port))
            tx.end()
            time.sleep(1)

        self.cbf_thread.join()
        self.cam_thread.join()
         # threads are now safe for finalisation
        logger.debug("All ingest threads now terminated.")
        self.cbf_thread.finalise()
         # no further correspondence will be entered into
        self.cbf_thread = None
         # we are done with the capture thread
        self.cam_thread = None

         # now we make sure to sync the model to the output file
        valid = self.model.is_valid()
         # check to see if we are valid up until the last 5 seconds
        if not valid: logger.warning("Model is not valid (for RTS this is expected). Writing to disk anyway.")
        self.model.finalise_h5_file(self.h5_file)
        smsg = self.model.close_h5_file(self.h5_file)
         # close file and rename if appropriate

        self.h5_file = None
        self.model = None
        self._my_sensors["capture-active"].set_value(0)
        for sensor in self._my_sensors:
            if sensor.startswith("spead"):
                self._my_sensors[sensor].set_value(0,status=Sensor.UNKNOWN)
                 # set all SPEAD sensors to unknown when thread has stopped
        logger.info(smsg)
        return ("ok", smsg)


if __name__ == '__main__':
    opts, args = parse_opts(sys.argv)

    logger = logging.getLogger('ingest')

    if isinstance(opts.loglevel, basestring):
        opts.loglevel = getattr(logging, opts.loglevel.upper())
    logger.setLevel(opts.loglevel)
    try:
        fh = logging.handlers.RotatingFileHandler(os.path.join(opts.workpath, 'ingest.log'), maxBytes=1e6, backupCount=10)
        formatter = logging.Formatter(("%(asctime)s.%(msecs)dZ - %(filename)s:%(lineno)s - %(levelname)s - %(message)s"),
                                      datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
         # we assume this is the SDP ur process and so we setup logging in a fairly manual fashion
    except IOError:
        logging.basicConfig()
        (logger.warn("Failed to create log file so reverting to console output. Most likely issue is that {0} does not exist or is not writeable"
         .format(os.path.join(opts.workpath))))

    if len(logger.root.handlers) > 0: logger.root.removeHandler(logger.root.handlers[0])
     # purge the root handler if it exists to avoid console duplicates

    spead.logger.setLevel(logging.WARNING)
     # configure SPEAD to display warnings about dropped packets etc...

    sp.ProcBlock.logger = logger
     # logger ref for use in the signal processing routines

    restart_queue = Queue.Queue()
    server = IngestDeviceServer(logger, opts.sdisp_ips, opts.sdisp_port, opts.antennas, opts.host, opts.port)
    server.set_restart_queue(restart_queue)
    server.start()
    logger.info("Started k7_capture server.")
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
        logger.info("Shutting down k7_capture server...")
        logger.info("Activity logging stopped")
        server.handle_interrupt()
        server.stop()
        server.join()
