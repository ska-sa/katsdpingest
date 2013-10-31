#!/usr/bin/python

# Capture utility for a relatively generic packetised correlator data output stream.

# The script performs two primary roles:
#
# Storage of stream data on disk in hdf5 format. This includes merging incoming meta-data with the correlator data
# stream to produce a complete, packaged hdf5 file.
#
# Regeneration of a SPEAD stream suitable for use in the online signal displays. At the moment this is basically
# just an aggregate of the incoming streams from the multiple x engines scaled with n_accumulations (if set)

import numpy as np
import spead
import h5py
import sys
import time
import optparse
import threading
import Queue
import copy
import os
import logging

from katcp import DeviceServer, Sensor, Message
from katcp.kattypes import request, return_reply, Str, Int, Float

import katsdpingest.sigproc as sp
from katsdpingest.ingest_threads import TMIngest, CBFIngest

 # import model components. In the future this may be done by the sdp_proxy and the 
 # complete model passed in.
from katsdpingest.telescope_model import AntennaPositioner, CorrelatorBeamformer, Enviro, TelescopeModel

import katconf

hdf5_version = "2.0"
 # initial version describing indicating compatibility with our HDF5v2 spec.
 # minor revision may be bumped when telescope model is written to file

def parse_opts(argv):
    parser = optparse.OptionParser()
    parser.add_option('--include_ctl', action='store_true', default=False, help='pull configuration information via katcp from the kat controller')
    parser.add_option('--sdisp-ips', default='127.0.0.1', help='default signal display destination ip addresses. Either single ip or comma separated list. [default=%default]')
    parser.add_option('--sdisp-port', default='7149',type=int, help='port on which to send signal display data. [default=%default]')
    parser.add_option('--data-port', default=7148, type=int, help='port to receive SPEAD data and meta-data from CBF on')
    parser.add_option('--meta-data-port', default=7147, type=int, help='port to receive SPEAD meta-data from CAM on')
    parser.add_option('--file-base', default='/var/kat/data/staging', help='base directory into which to write HDF5 files. [default=%default]')
    parser.add_option('-s', '--system', default='systems/local.conf', help='system configuration file to use. [default=%default]')
    parser.add_option('-p', '--port', dest='port', type=long, default=2040, metavar='N', help='katcp host port. [default=%default]')
    parser.add_option('-a', '--host', dest='host', type="string", default="", metavar='HOST', help='katcp host address. [default="" (all hosts)]')
    parser.add_option('-l', '--logging', dest='logging', type='string', default=None, metavar='LOGGING',
            help='level to use for basic logging or name of logging configuration file; ' \
            'default is /log/log.<SITENAME>.conf')
    return parser.parse_args(argv)

class IngestDeviceServer(DeviceServer):
    """Serves the ingest katcp interface. 
    Top level holder of the ingest threads and the owner of any output files."""

    VERSION_INFO = ("sdp-ingest", 0, 1)
    BUILD_INFO = ("sdp-ingest", 0, 1, "rc1")

    def __init__(self, logger, sdisp_ips, sdisp_port, system_config_path, *args, **kwargs):
        self.logger = logger
        self.cbf_thread = None
         # reference to the CBF ingest thread
        self.tm_thread = None
         # reference to the Telescope Manager thread
        self.h5_file = None
         # the current hdf5 file in use by the ingest threads
        self.system_config_path = system_config_path
        self.sdisp_ips = {}
        self.sdisp_ips['127.0.0.1'] = sdisp_port
         # add default signal display destination
        for ip in sdisp_ips.split(","):
            self.sdisp_ips[ip] = sdisp_port
         # add additional user specified ip
        self._my_sensors = {}
        self._my_sensors["capture-active"] = Sensor(Sensor.INTEGER, "capture_active", "Is there a currently active capture thread.","",default=0, params = [0,1])
        self._my_sensors["packets-captured"] = Sensor(Sensor.INTEGER, "packets_captured", "The number of packets captured so far by the current session.","",default=0, params=[0,2**63])
        self._my_sensors["status"] = Sensor.string("status", "The current status of the capture thread.","")
        self._my_sensors["label"] = Sensor.string("label", "The label applied to the data as currently captured.","")

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

    def create_h5_file(self):
        if self.h5_file is not None: return None

        fname = "{0}/{1}.writing.h5".format(opts.file_base, str(int(time.time())))
        f = h5py.File(fname, mode="w")
        f['/'].attrs['version'] = hdf5_version
        f['/'].create_group('Data')
        f['/'].create_group('MetaData')
        f['/'].create_group('MetaData/Configuration')
        f['/'].create_group('MetaData/Configuration/Observation')
        f['/'].create_group('MetaData/Configuration/Correlator')
        f['/'].create_group('Markup')
        f['/Markup'].create_dataset('labels', [1], maxshape=[None], dtype=np.dtype([('timestamp', np.float64), ('label', h5py.new_vlen(str))]))
         # create a label storage of variable length strings
        f['/'].create_group('History')
        f['/History'].create_dataset('script_log', [1], maxshape=[None], dtype=np.dtype([('timestamp', np.float64), ('log', h5py.new_vlen(str))]))
        f['/History'].create_dataset('process_log',[1], maxshape=[None], dtype=np.dtype([('process', h5py.new_vlen(str)), ('arguments', h5py.new_vlen(str)), ('revision', np.int32)]))
        return f

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
        self._my_sensors["label"].set_value("no_thread")
        self._my_sensors["status"].set_value("init")

    @return_reply(Str())
    def request_sd_metadata_issue(self, req, msg):
        """Resend the signal display metadata packets..."""
        if self.cbf_thread is None: return ("fail","No active capture thread. Please start one using capture_init or via a schedule block.")
        self.cbf_thread.send_sd_metadata()
        smsg = "SD Metadata resent"
        activitylogger.info(smsg)
        return ("ok", smsg)

    @return_reply(Str())
    def request_capture_start(self, req, msg):
        """Dummy capture start command - calls capture init."""
        self.request_capture_init(req, msg)
        smsg = "Capture initialised at %s" % time.ctime()
        activitylogger.info(smsg)
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
             # may be worth expanding the scope to checking the file and the TM thread as well
        self.h5_file = self.create_h5_file()
         # open a new HDF5 file
        if self.h5_file is None:
            return ("fail","Failed to create HDF5 file. Init failed.")

        # for RTS we build a standard model. Normally this would be provided by the sdp_proxy
        m063 = AntennaPositioner(name='m063')
        m062 = AntennaPositioner(name='m062')
        cbf = CorrelatorBeamformer(name='DBE')
        env = Enviro(name='asc')
        model = TelescopeModel()
        model.add_components([m063,m062,cbf,env])
        model.build_index()

        self.cbf_thread = CBFIngest(opts.data_port, self.h5_file, self._my_sensors, model, cbf.name, logger)
        self.cbf_thread.setDaemon(True)
        self.cbf_thread.start()

        self.tm_thread = TMIngest(opts.meta_data_port, self.h5_file, model, logger)
        self.tm_thread.setDaemon(True)
        self.tm_thread.start()

        self._my_sensors["capture-active"].set_value(1)
         # add in existing signal display recipients...
        for (ip,port) in self.sdisp_ips.iteritems():
            self.cbf_thread.add_sdisp_ip(ip,port)
        smsg =  "Capture initialised at %s" % time.ctime()
        activitylogger.info(smsg)
        return ("ok", smsg)

    @request()
    @return_reply(Str())
    def request_enable_van_vleck(self, req):
        """Enable Van Vleck correction of the auto-correlated visibilities."""

    @request(Float())
    @return_reply(Str())
    def request_set_center_freq(self, req, center_freq_hz):
        """Set the center freq for use in the signal displays.

        Parameters
        ----------
        center_freq_hz : int
            The current system center frequency in hz
        """
        if self.cbf_thread is None: return ("fail","No active capture thread. Please start one using capture_init")
        self.cbf_thread.center_freq = center_freq_hz
        self.cbf_thread.send_sd_metadata()
        smsg = "SD Metadata resent"
        activitylogger.info(smsg)
        return ("ok","set")

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
            if key_string in obs_sensors:
                self._my_sensors[key_string].set_value(value_string)
            self.cbf_thread.write_obs_param(key_string, value_string)
        except ValueError, e:
            return ("fail", "Could not parse sensor name or value string '%s=%s': %s" % (key_string, value_string, e))
        smsg = "%s=%s" % (key_string, value_string)
        activitylogger.info("Set script param %s" % smsg)
        return ("ok", smsg)

    @request(Str())
    @return_reply(Str())
    def request_script_log(self, req, log):
        """Add an entry to the script log."""
        if self.cbf_thread is None: return ("fail","No active capture thread. Please start one using capture_start")
        self._my_sensors["script-log"].set_value(log)
        self.cbf_thread.write_log(log)
        smsg = "Script log entry added (%s)" % log
        activitylogger.info(smsg)
        return ("ok", smsg)

    @request(Str())
    @return_reply(Str())
    def request_drop_sdisp_ip(self, req, ip):
        """Drop an IP address from the internal list of signal display data recipients."""
        if self.cbf_thread is not None:
            if not self.sdisp_ips.has_key(ip):
                return ("fail","The IP address specified (%s) does not exist in the current list of recipients." % (ip))
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

    @request(Str())
    @return_reply(Str())
    def request_set_label(self, req, label):
        """Set the current scan label to the supplied value."""
        if self.cbf_thread is None: return ("fail","No active capture thread. Please start one using capture_start")
        self._my_sensors["label"].set_value(label)
        self.cbf_thread.write_label(label)
        return ("ok","Label set to %s" % label)

    @return_reply(Str())
    def request_get_current_file(self, req, msg):
        """Return the name of the current (or most recent) capture file."""
        if self.h5_file is None:
            return ("fail", "No currently active file.")
        return ("ok", self.h5_file.filename)

    @return_reply(Str())
    def request_capture_done(self, req, msg):
        """Closes the current capture file and renames it for use by augment."""
        if self.cbf_thread is None:
            return ("fail","No existing capture session.")

        if self.cbf_thread.is_alive():
            # first try to shutdown the threads gracefully
            tx = spead.Transmitter(spead.TransportUDPtx('localhost',opts.data_port))
            tx.end()
            tx = spead.Transmitter(spead.TransportUDPtx('localhost',opts.meta_data_port))
            tx.end()
            time.sleep(2)

        self.cbf_thread.finalise()
         # no further correspondence will be entered into

        self.tm_thread.write_model()
         # at this point data writing is complete.
         # now we write the model to the file as well
        self.cbf_thread.join()
        self.tm_thread.join()
         # we really dont want these lurking around
        self.cbf_thread = None
         # we are done with the capture thread
        self.tm_thread = None

        if self.h5_file is None:
            logger.warning("File was already closed. Not renaming...")
            return ("ok","File was already closed.")
        filename = self.h5_file.filename
         # grab filename before we close
        self.h5_file.flush()
        self.h5_file.close()
         # make sure to close and flush gracefully
        output_file = filename[:filename.find(".writing.h5")] + ".unaugmented.h5"
        try:
            os.rename(filename, output_file)
        except Exception, e:
            logger.error("Failed to rename output file %s" % e)
            return ("fail","Failed to rename output file from %s to %s." % (filename, output_file))
        finally:
            self.h5_file = None
        smsg = "File renamed to %s" % (output_file)
        self._my_sensors["capture-active"].set_value(0)
        for sensor in self._my_sensors:
            if sensor.startswith("spead"):
                self._my_sensors[sensor].set_value(0,status=Sensor.UNKNOWN)
                 # set all SPEAD sensors to unknown when thread has stopped
        activitylogger.info(smsg)
        return ("ok", smsg)

if __name__ == '__main__':
    opts, args = parse_opts(sys.argv)

    # Setup configuration source
    #katconf.set_config(katconf.environ(opts.sysconfig))
    # set up Python logging
    #katconf.configure_logging(opts.logging)

    logger = logging.getLogger("kat.k7capture")
    logger.setLevel(logging.INFO)

    spead.logger.setLevel(logging.WARNING)
     # configure SPEAD to display warnings about dropped packets etc...

    sp.ProcBlock.logger = logger
     # logger ref for use in the signal processing routines
    activitylogger = logging.getLogger("activity")
    activitylogger.setLevel(logging.INFO)
    activitylogger.info("Activity logging started")

    restart_queue = Queue.Queue()
    server = IngestDeviceServer(logger, opts.sdisp_ips, opts.sdisp_port, opts.system,
                                 opts.host, opts.port)
    server.set_restart_queue(restart_queue)
    server.start()
    activitylogger.info("Started k7_capture server.")
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
        activitylogger.info("Shutting down k7_capture server...")
        activitylogger.info("Activity logging stopped")
        server.stop()
        server.join()
