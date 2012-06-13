#!/usr/bin/python

# Capture utility for a relatively generic packetised correlator data output stream.

# The script performs two primary roles:
#
# Storage of stream data on disk in hdf5 format. This includes placing meta data into the file as attributes.
#
# Regeneration of a SPEAD stream suitable for us in the online signal displays. At the moment this is basically
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
import katcapture.sigproc as sp

try:
    import katconf
    found_katconf = True
except ImportError:
    found_katconf = False

hdf5_version = "2.0"
 # initial version describing indicating compatibility with our HDF5v2 spec. Minor revision may be incremented by augment at a later stage.


mapping = {'xeng_raw':'/Data/correlator_data',
           'timestamp':'/Data/raw_timestamps'}
 # maps SPEAD element names to HDF5 paths
timestamps_dataset = '/Data/timestamps'
flags_dataset = '/Markup/flags'
correlator_map = '/MetaData/Configuration/Correlator/'
observation_map = '/MetaData/Configuration/Observation/'
 # default path for things that are not mentioned above
script_sensors = ['script_arguments','script_description','script_experiment_id','script_name','script_nd_params','script_observer','script_rf_params','script_starttime','script_status']
 # sensors to pull from the ctl katcp client
sdisp_ips = {}
 # dict storing the configured signal destination ip addresses

def small_build(system):
    print "Creating KAT connections..."
    katconfig = katcorelib.conf.KatuilibConfig(system)
    ctl_config = katconfig.clients['sys']
    ctl = katcorelib.build_client(ctl_config.name, ctl_config.ip, ctl_config.port)
    count=0
    while not ctl.is_connected() and count < 6:
        count+=1
        print "Waiting for ctl client to become available... (wait %i/5)" % count
        time.sleep(2)
        if not ctl.is_connected():
            print "Failed to connect to ctl client (ip: %s, port: %i)\n" % (ctl_config.ip, ctl_config.port)
            sys.exit(0)
        return ctl

def parse_opts(argv):
    parser = optparse.OptionParser()
    parser.add_option('--include_ctl', action='store_true', default=False, help='pull configuration information via katcp from the kat controller')
    parser.add_option('--sdisp-ips', default='127.0.0.1', help='default signal display destination ip addresses. Either single ip or comma separated list. [default=%default]')
    parser.add_option('--sdisp-port', default='7149',type=int, help='port on which to send signal display data. [default=%default]')
    parser.add_option('--data-port', default=7148, type=int, help='port to receive SPEAD data and metadata on')
    parser.add_option('-s', '--system', default='systems/local.conf', help='system configuration file to use. [default=%default]')
    parser.add_option('-p', '--port', dest='port', type=long, default=2040, metavar='N', help='katcp host port. [default=%default]')
    parser.add_option('-a', '--host', dest='host', type="string", default="", metavar='HOST', help='katcp host address. [default="" (all hosts)]')
    parser.add_option("-f", "--sysconfig", dest='sysconfig', default='/var/kat/katconfig', help='look for configuration files in folder CONF [default is KATCONF environment variable or /var/kat/conf]')
    parser.add_option('-l', '--logging', dest='logging', type='string', default=None, metavar='LOGGING',
            help='level to use for basic logging or name of logging configuration file; ' \
            'default is /log/log.<SITENAME>.conf')
    return parser.parse_args(argv)

class k7Capture(threading.Thread):
    def __init__(self, data_port, ctl, my_sensors):
        self.data_port = data_port
        self.data_scale_factor = 1.0
        self.acc_scale = True
        self.ctl = ctl
        self._label_idx = 0
        self._log_idx = 0
        self._current_hdf5 = None
        self._my_sensors = my_sensors
        self.pkt_sensor = self._my_sensors['packets-captured']
        self.status_sensor = self._my_sensors['status']
        self.status_sensor.set_value("init")
        self.fname = None
        self._sd_metadata = None
        self.sdisp_ips = {}
        self._sd_count = 0
        self.init_file()
        self.center_freq = 0
        self.meta = {}
        self.ig_sd = spead.ItemGroup()
        self.int_time = 1.0
         # default integration time in seconds. Updated by SPEAD metadata on stream initiation.
        self.sd_frame = None
        self.baseline_mask = None
         # by default record all baselines
        self._script_ants = None
         # a reference to the antennas requested from the current script
        #### Initialise processing blocks used
        self.scale = sp.Scale(self.data_scale_factor)
         # at this stage the scale factor is unknown
        self.rfi = sp.RFIThreshold2()
         # basic rfi thresholding flagger
        #### Done with blocks
        threading.Thread.__init__(self)

    def send_sd_data(self, data):
        #if self._sd_count % 10 == 0:
        #    logger.debug("Sending metadata heartbeat...")
        #    self.send_sd_metadata()

        for tx in self.sdisp_ips.itervalues():
            tx.send_heap(data)

        self._sd_count += 1

    def _update_sd_metadata(self):
        """Update the itemgroup for the signal display metadata to include any changes since last sent..."""
        self.ig_sd = spead.ItemGroup()
         # we need to clear the descriptor so as not to accidently send a signal display frame twice...
        self.ig_sd.add_item(name=('sd_data'),id=(0x3501), description="Combined raw data from all x engines.", ndarray=(self.sd_frame.dtype,self.sd_frame.shape))
        self.ig_sd.add_item(name=('sd_timestamp'), id=0x3502, description='Timestamp of this sd frame in centiseconds since epoch (40 bit limitation).',
                            shape=[], fmt=spead.mkfmt(('u',spead.ADDRSIZE)))
        self.ig_sd.add_item(name=('bls_ordering'), id=0x100C, description="Mapping of antenna/pol pairs to data output products.", init_val=self.meta['bls_ordering'])
        logger.debug("Update metadata. Bls ordering: %s" % self.meta['bls_ordering'])
        self.ig_sd.add_item(name="center_freq",id=0x1011, description="The center frequency of the DBE in Hz, 64-bit IEEE floating-point number.",
                            shape=[],fmt=spead.mkfmt(('f',64)), init_val=self.center_freq)
        self.ig_sd.add_item(name="bandwidth",id=0x1013, description="The analogue bandwidth of the digitally processed signal in Hz.",
                            shape=[],fmt=spead.mkfmt(('f',64)), init_val=self.meta['bandwidth'])
        self.ig_sd.add_item(name="n_chans",id=0x1009, description="The total number of frequency channels present in any integration.",
                            shape=[], fmt=spead.mkfmt(('u',spead.ADDRSIZE)), init_val=self.meta['n_chans'])
        return copy.deepcopy(self.ig_sd.get_heap())

    def set_baseline_mask(self):
        """Uses the _script_ants variable to set a baseline mask.
        This only works if script_ants has been set by an external process between capture_done and capture_start."""
        if self._script_ants is not None and self.meta.has_key('bls_ordering'):
            logger.info("Using script-ants (%s) as a custom baseline mask..." % self._script_ants)
            ants = self._script_ants.replace(" ","").split(",")
            if len(ants) > 0:
                b = self.meta['bls_ordering'].tolist()
                self.baseline_mask = [b.index(pair) for pair in b if pair[0][:-1] in ants and pair[1][:-1] in ants]
                self.meta['bls_ordering'] = np.array([b[idx] for idx in self.baseline_mask])
                 # we need to recalculate the bls ordering as well...
        if self.baseline_mask is None or len(self.baseline_mask) == 0:
            self.baseline_mask = range(self.meta['n_bls'])
             # by default we send all baselines...

    def send_sd_metadata(self):
        self._sd_metadata = self._update_sd_metadata()
        if self._sd_metadata is not None:
            for tx in self.sdisp_ips.itervalues():
                mdata = copy.deepcopy(self._sd_metadata)
                tx.send_heap(mdata)

    def remap(self, name):
        return name in mapping and mapping[name] or correlator_map + name

    def write_obs_param(self, sensor_string, value_string):
        f = (self._current_hdf5 is None and self.init_file() or self._current_hdf5)
        f['/MetaData/Configuration/Observation'].attrs[sensor_string.replace("-","_")] = value_string
        if sensor_string == "script-experiment-id":
            f.attrs['experiment_id'] = value_string
             # duplicated for easy use by the archiver. Note change of name from script-experiment-id

    def write_log(self, log):
        """Write a log value directly into the current hdf5 file."""
        f = (self._current_hdf5 is None and self.init_file() or self._current_hdf5)
        if self._log_idx > 0:
            f['/History/script_log'].resize(self._log_idx+1,axis=0)
        f['/History/script_log'][self._log_idx] = (time.time(), log)
        self._log_idx += 1

    def write_label(self, label):
        """Write a sensor value directly into the current hdf5 at the specified locations.
           Note that this will create a new HDF5 file if one does not already exist..."""
        f = (self._current_hdf5 is None and self.init_file() or self._current_hdf5)
        if self._label_idx > 0:
            f['/Markup/labels'].resize(self._label_idx+1,axis=0)
        f['/Markup/labels'][self._label_idx] = (time.time(), label)
        self._label_idx += 1

    def init_file(self):
        self.fname = "/var/kat/data/staging/" + str(int(time.time())) + ".writing.h5"
        f = h5py.File(self.fname, mode="w")
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
        self._current_hdf5 = f
        return f

    def close_file(self):
        """Close file handle reference and mark file is not current."""
        if self._current_hdf5 is not None:
            self._current_hdf5.flush()
            self._current_hdf5.close()
             # we may have ended capture before receiving any data packets and thus not have a current file
        self._current_hdf5 = None

    def add_sdisp_ip(self, ip, port):
        logger.info("Adding %s:%s to signal display list. Starting transport..." % (ip,port))
        self.sdisp_ips[ip] = spead.Transmitter(spead.TransportUDPtx(ip, port))
        if self._sd_metadata is not None:
            mdata = copy.deepcopy(self._sd_metadata)
            self.sdisp_ips[ip].send_heap(mdata)
             # new connection requires headers...

    def run(self):
        activitylogger.info("Capture started")
        logger.info("Initalising SPEAD transports at %f" % time.time())
        logger.info("Data reception on port %i" % self.data_port)
        rx = spead.TransportUDPrx(self.data_port, pkt_count=1024, buffer_size=51200000)
        #print "Sending Signal Display data to", self.sd_ip
        #tx_sd = spead.Transmitter(spead.TransportUDPtx(self.sd_ip, 7149))
        #self._tx_sd = tx_sd
        ig = spead.ItemGroup()
        idx = 0
        f = None
        self.status_sensor.set_value("idle")
        dump_size = 0
        datasets = {}
        datasets_index = {}
        meta_required = set(['n_chans','n_bls','bls_ordering','bandwidth'])
         # we need these bits of meta data before being able to assemble and transmit signal display data
        meta_desired = ['int_time', 'n_accs','center_freq']
         # if we find these, then what hey :)
        sd_slots = None
        sd_timestamp = None
        for heap in spead.iterheaps(rx):
            if idx == 0:
                f = (self._current_hdf5 is None and self.init_file() or self._current_hdf5)
                self.status_sensor.set_value("capturing")
            ig.update(heap)
            for name in ig.keys():
                item = ig.get_item(name)
                if not item._changed and datasets.has_key(name): continue
                 # the item is not marked as changed, and we have a record for it
                if name in meta_desired:
                    logger.info("Meta data received (desired) %s: %s => %s" % (time.ctime(), name, str(ig[name])))
                    self.meta[name] = ig[name]
                    if name == 'center_freq' and self.center_freq == 0:
                        self.center_freq = self.meta[name]
                        self._my_sensors["spead-center-freq"].set_value(self.meta[name])
                    if name == 'int_time' and self.int_time == 1:
                        self.int_time = self.meta[name]
                        self._my_sensors["spead-dump-period"].set_value(self.meta[name])
                    if name == 'n_accs':
                        self.data_scale_factor = np.float32(ig[name])
                        self.scale.scale_factor = self.data_scale_factor
                        self._my_sensors["spead-accum-per-dump"].set_value(self.meta[name])
                        logger.debug("Scale factor set to: %f\n" % self.data_scale_factor)
                if name in meta_required:
                    logger.info("Meta data received (required) %s: %s => %s" % (time.ctime(), name, str(ig[name])))
                    self.meta[name] = ig[name]
                    meta_required.remove(name)
                    if name == 'n_chans':
                        self._my_sensors["spead-num-chans"].set_value(self.meta[name])
                    if name == 'n_bls':
                        self._my_sensors["spead-num-bls"].set_value(self.meta[name])
                    if not meta_required:
                        self.set_baseline_mask()
                         # we first set the baseline mask in order to have the correct number of baselines
                        self.sd_frame = np.zeros((self.meta['n_chans'],len(self.baseline_mask),2),dtype=np.float32)
                        logger.debug("Initialised sd frame to shape %s" % str(self.sd_frame.shape))
                        meta_required = set(['n_chans','n_bls','bls_ordering','bandwidth'])
                        sd_slots = None
                if not name in datasets:
                 # check to see if we have encountered this type before
                    if name == 'xeng_raw' and self._script_ants is not None and self.baseline_mask is None:
                      # we are supposed to set a baseline mask, but this has not been done yet as the desired meta_data
                      # has not yet arrived (bls_ordering). Defer creation of xeng_raw dataset until this is available.
                        continue
                    shape = ig[name].shape if item.shape == -1 else item.shape
                    dtype = np.dtype(type(ig[name])) if shape == [] else item.dtype
                    if dtype is None:
                        dtype = ig[name].dtype
                     # if we can't get a dtype from the descriptor try and get one from the value
                    new_shape = list(shape)
                    if name == 'xeng_raw':
                        if self.baseline_mask is not None:
                            new_shape[-2] = len(self.baseline_mask)
                        dtype = np.dtype(np.float32)
                    logger.debug("Creating dataset for name: %s, shape: %s, dtype: %s" % (name, str(new_shape), dtype))
                    if new_shape == [1]:
                        new_shape = []
                    f.create_dataset(self.remap(name),[1] + new_shape, maxshape=[None] + new_shape, dtype=dtype)
                    dump_size += np.multiply.reduce(shape) * dtype.itemsize
                    datasets[name] = f[self.remap(name)]
                    datasets_index[name] = 0
                    if name == 'timestamp':
                        f.create_dataset(timestamps_dataset,[1] + new_shape, maxshape=[None] + new_shape, dtype=np.float64)
                        item._changed = False
                    if name == 'xeng_raw':
                        f.create_dataset(flags_dataset, [1] + new_shape[:-1], maxshape=[None] + new_shape[:-1], dtype=np.uint8)
                        if self.rfi is not None:
                            f[flags_dataset].attrs['flag_types'] = self.rfi.flag_types
                    if not item._changed:
                        continue
                     # if we built from an empty descriptor
                else:
                    if not item._changed:
                        continue
                    f[self.remap(name)].resize(datasets_index[name]+1, axis=0)
                    if name == 'timestamp':
                        f[timestamps_dataset].resize(datasets_index[name]+1, axis=0)
                    if name == 'xeng_raw':
                        f[flags_dataset].resize(datasets_index[name]+1, axis=0)
                if self.sd_frame is not None and name.startswith("xeng_raw"):
                    sd_timestamp = ig['sync_time'] + (ig['timestamp'] / ig['scale_factor_timestamp'])
                    if sd_slots is None:
                        self.sd_frame.dtype = np.dtype(np.float32) # if self.acc_scale else ig[name].dtype
                         # make sure we have the right dtype for the sd data
                        sd_slots = np.zeros(self.meta['n_chans']/ig[name].shape[0])
                        self.send_sd_metadata()
                        t_it = self.ig_sd.get_item('sd_data')
                        #print "Added SD frame dtype",t_it.dtype,"and shape",t_it.shape,". Metadata descriptors sent: %s" % self._sd_metadata
                    scaled_data = np.float32(ig[name]) / self.data_scale_factor
                    logger.info("Sending signal display frame with timestamp %i (local: %f). %s." % (sd_timestamp, time.time(), "Unscaled" if not self.acc_scale else "Scaled by %i" % self.data_scale_factor))
                    #logger.debug("Signal display frame: Max: %f, Min: %f, Mean: %f\n" % (np.max(scaled_data), np.min(scaled_data), np.mean(scaled_data)))
                    self.ig_sd['sd_data'] = scaled_data[...,self.baseline_mask,:]
                     # only send signal display data specified by baseline mask
                    self.ig_sd['sd_timestamp'] = int(sd_timestamp * 100)
                    self.send_sd_data(self.ig_sd.get_heap())
                if name.startswith("xeng_raw"):
                     # we have data...
                    sp.ProcBlock.current = ig[name][...,self.baseline_mask,:]
                     # update our data pointer. at this stage dtype is int32 and shape (channels, baselines, 2)
                    self.scale.proc()
                     # scale the data
                    sp.ProcBlock.current = sp.ProcBlock.current.copy()
                     # *** Dont delete this line ***
                     # since array uses advanced selection (baseline_mask not contiguous) we may
                     # have a changed the striding underneath with can prevent the array being viewed
                     # as complex64. A copy resets the stride to its natural form and fixes this.
                    sp.ProcBlock.current = sp.ProcBlock.current.view(np.complex64)[...,0]
                     # temporary reshape to complex, eventually this will be done by Scale
                     # (Once the final data format uses C64 instead of float32)
                    self.rfi.init_flags()
                     # begin flagging operations
                    self.rfi.proc()
                     # perform rfi thresholding
                    flags = self.rfi.finalise_flags()
                     # finalise flagging operations and return flag data to write
                    f[flags_dataset][datasets_index[name]] = flags
                     # write flags to file
                    f[self.remap(name)][datasets_index[name]] = sp.ProcBlock.current.view(np.float32).reshape(list(sp.ProcBlock.current.shape) + [2])[np.newaxis,...]
                     # write data to file (with temporary remap as mentioned above...)
                else:
                    f[self.remap(name)][datasets_index[name]] = ig[name]
                if name == 'timestamp':
                    try:
                        current_ts = ig['sync_time'] + (ig['timestamp'] / ig['scale_factor_timestamp'])
                        self._my_sensors["last-dump-timestamp"].set_value(current_ts)
                        f[timestamps_dataset][datasets_index[name]] = current_ts
                         # insert derived timestamps
                    except KeyError:
                        f[timestamps_dataset][datasets_index[name]] = -1.0
                datasets_index[name] += 1
                item._changed = False
                  # we have dealt with this item so continue...
            if idx==0 and self.ctl is not None:
                # add script metadata after receiving first frame. This should ensure that the values pulled are fresh.
                for s in script_sensors:
                    f[observation_map].attrs[s] = kat.sys.sensor.__getattribute__(s).get_value()
                logger.info("Added initial observation sensor values...\n")
            idx+=1
            f.flush()
            self.pkt_sensor.set_value(idx)

        if self.baseline_mask is not None:
            del f[self.remap('bls_ordering')]
            del datasets_index['bls_ordering']
            f[correlator_map].attrs['bls_ordering'] = self.meta['bls_ordering']
             # we have generated a new baseline mask, so fix bls_ordering attribute...

        logger.info("Capture complete at %f" % time.time())
        logger.debug("\nProcessing Blocks\n=================\n%s\n%s\n" % (self.scale,self.rfi))
        self.status_sensor.set_value("complete")
        activitylogger.info("Capture complete")

class CaptureDeviceServer(DeviceServer):

    VERSION_INFO = ("k7-capture", 0, 1)
    BUILD_INFO = ("k7-capture", 0, 1, "rc1")

    def __init__(self, sdisp_ips, sdisp_port, *args, **kwargs):
        self.rec_thread = None
        self.current_file = None
        self.sdisp_ips = {}
        self.sdisp_ips['127.0.0.1'] = sdisp_port
         # add default signal display destination
        for ip in sdisp_ips.split(","):
            self.sdisp_ips[ip] = sdisp_port
         # add additional user specified ip
        self._my_sensors = {}
        self._my_sensors["capture-active"] = Sensor(Sensor.INTEGER, "capture_active", "Is there a currently active capture thread.","",default=0, params = [0,1])
        self._my_sensors["packets-captured"] = Sensor(Sensor.INTEGER, "packets_captured", "The number of packets captured so far by the current session.","",default=0, params=[0,2**63])
        self._my_sensors["status"] = Sensor(Sensor.STRING, "status", "The current status of the capture thread.","")
        self._my_sensors["label"] = Sensor(Sensor.STRING, "label", "The label applied to the data as currently captured.","")
        self._my_sensors["script-ants"] = Sensor(Sensor.STRING, "script-ants","The antennas specified by the user for use by the executed script.","")
        self._my_sensors["script-log"] = Sensor(Sensor.STRING, "script-log", "The most recent script log entry.","")
        self._my_sensors["script-name"] = Sensor(Sensor.STRING, "script-name", "Current script name", "")
        self._my_sensors["script-experiment-id"] = Sensor(Sensor.STRING, "script-experiment-id", "Current experiment id", "")
        self._my_sensors["script-observer"] = Sensor(Sensor.STRING, "script-observer", "Current experiment observer", "")
        self._my_sensors["script-description"] = Sensor(Sensor.STRING, "script-description", "Current experiment description", "")
        self._my_sensors["script-rf-params"] = Sensor(Sensor.STRING, "script-rf-params", "Current experiment RF parameters", "")
        self._my_sensors["script-nd-params"] = Sensor(Sensor.STRING, "script-nd-params", "Current experiment Noise Diode parameters", "")
        self._my_sensors["script-arguments"] = Sensor(Sensor.STRING, "script-arguments", "Options and parameters of script - from sys.argv", "")
        self._my_sensors["script-status"] = Sensor(Sensor.STRING, "script-status", "Current status reported by running script", "")
        self._my_sensors["script-starttime"] = Sensor(Sensor.STRING, "script-starttime", "Start time of current script", "")
        self._my_sensors["script-endtime"] = Sensor(Sensor.STRING, "script-endtime", "End time of current script", "")
        self._my_sensors["spead-num-chans"] = Sensor(Sensor.INTEGER, "spead_num_chans","Number of channels reported via SPEAD header from the DBE","",default=0,params=[0,2**63])
        self._my_sensors["spead-num-bls"] = Sensor(Sensor.INTEGER, "spead_num_bls","Number of baselines reported via SPEAD header from the DBE","",default=0,params=[0,2**63])
        self._my_sensors["spead-dump-period"] = Sensor(Sensor.FLOAT, "spead_dump_period","Dump period reported via SPEAD header from the DBE","",default=0,params=[0,2**31])
        self._my_sensors["spead-accum-per-dump"] = Sensor(Sensor.INTEGER, "spead_accum_per_dump","Accumulations per dump reported via SPEAD header from the DBE","",default=0,params=[0,2**63])
        self._my_sensors["spead-center-freq"] = Sensor(Sensor.FLOAT, "spead_center_freq","Center frequency of correlator reported via SPEAD header","",default=0,params=[0,2**31])
        self._my_sensors["last-dump-timestamp"] = Sensor(Sensor.FLOAT, "last_dump_timestamp","Timestamp of most recently received correlator dump in Unix seconds","",default=0,params=[0,2**63])

        super(CaptureDeviceServer, self).__init__(*args, **kwargs)

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
    def request_sd_metadata_issue(self, sock, msg):
        """Resend the signal display metadata packets..."""
        self.rec_thread.send_sd_metadata()
        smsg = "SD Metadata resent"
        activitylogger.info(smsg)
        return ("ok", smsg)

    @return_reply(Str())
    def request_capture_start(self, sock, msg):
        """Dummy capture start command - calls capture init."""
        self.request_capture_init(sock, msg)
        smsg = "Capture initialised at %s" % time.ctime()
        activitylogger.info(smsg)
        return ("ok", smsg)

    @return_reply(Str())
    def request_capture_init(self, sock, msg):
        """Spawns a new capture thread that waits for a SPEAD start stream packet."""
        if self.rec_thread is not None:
            return ("fail", "Existing capture session found. If you really want to init, stop the current capture using capture_stop.")
        self.rec_thread = k7Capture(opts.data_port, ctl, self._my_sensors)
        self.rec_thread.setDaemon(True)
        self.rec_thread.start()
        self._my_sensors["capture-active"].set_value(1)
         # add in existing signal display recipients...
        for (ip,port) in self.sdisp_ips.iteritems():
            self.rec_thread.add_sdisp_ip(ip,port)
        self.current_file = self.rec_thread.fname
        smsg =  "Capture initialised at %s" % time.ctime()
        activitylogger.info(smsg)
        return ("ok", smsg)

    @request(Int())
    @return_reply(Str())
    def request_set_center_freq(self, sock, center_freq_hz):
        """Set the center freq for use in the signal displays.
    
        Parameters
        ----------
        center_freq_hz : int
            The current system center frequency in hz
        """
        if self.rec_thread is None: return ("fail","No active capture thread. Please start one using capture_init")
        self.rec_thread.center_freq = center_freq_hz
        return ("ok","set")

    @request(Str())
    @return_reply(Str())
    def request_set_antenna_mask(self, sock, ant_mask):
        """Set the antennas to be used to select which baseline are recorded (and sent to signal display targets).
        This must be done after a capture_init and before data starts being recorded.

        Note: This is not a persistent setting and must be reset for each new capture session.

        Parameters
        ----------
        ant_mask : str
            Comma delimited string of antenna names. e.g. 'ant1,ant2'. An empty string clears the mask.
        """
        if self.rec_thread is None: return ("fail","This mask must be set after issuing capture_init and before data transmission begins...")
        if ant_mask == '': return ("ok","Antenna mask cleared.")
        self.rec_thread._script_ants = ant_mask
        return ("ok", "Only data for the following antennas will be produced: %s" % ant_mask)

    @request(Str(), Str())
    @return_reply(Str())
    def request_set_script_param(self, sock, sensor_string, value_string):
        """Set the desired script parameter.

        Parameters
        ----------
        sensor_string : str
            The script parameter to be set. [script-ants, script-name, script-experiment-id, script-observer, script-description, script-rf-params, script-nd-params, script-arguments, script-status, script-starttime, script-endtime]
        value_string : str
            A string containing the value to be set
            
        Returns
        -------
        success : {'ok', 'fail'}
            Whether setting the sensor succeeded.
        sensor_string : str
            Name of sensor that was set
        value_string : str
            A string containing the sensor value it was set to
            
        Examples
        --------
        ?set_script_param script-name Test
        !set_script_param ok script-name
        
        """
        if self.rec_thread is None: return ("fail","No active capture thread. Please start one using capture_init")
        try:
            self._my_sensors[sensor_string].set_value(value_string)
            self.rec_thread.write_obs_param(sensor_string, value_string)
            if sensor_string == 'script-ants': self.rec_thread._script_ants = value_string
        except ValueError, e:
            return ("fail", "Could not parse sensor name or value string '%s=%s': %s" % (sensor_string, value_string, e))
        smsg = "%s=%s" % (sensor_string, value_string)
        activitylogger.info("Set script param %s" % smsg)
        return ("ok", smsg)

    @request(Str())
    @return_reply(Str())
    def request_script_log(self, sock, log):
        """Add an entry to the script log."""
        if self.rec_thread is None: return ("fail","No active capture thread. Please start one using capture_start")
        self._my_sensors["script-log"].set_value(log)
        self.rec_thread.write_log(log)
        smsg = "Script log entry added (%s)" % log
        activitylogger.info(smsg)
        return ("ok", smsg)

    @request(Str())
    @return_reply(Str())
    def request_add_sdisp_ip(self, sock, ip):
        """Add the supplied ip and port (ip[:port]) to the list of signal display data recipients.If not port is supplied default of 7149 is used."""
        ipp = ip.split(":")
        ip = ipp[0]
        if len(ipp) > 1: port = int(ipp[1])
        else: port = 7149
        if self.sdisp_ips.has_key(ip): return ("ok","The supplied IP is already in the active list of recipients.")
        self.sdisp_ips[ip] = port
        if self.rec_thread is not None:
            self.rec_thread.add_sdisp_ip(ip, port)
        return ("ok","Added IP address %s (port: %i) to list of signal display data recipients." % (ip, port))

    @request(Str())
    @return_reply(Str())
    def request_set_label(self, sock, label):
        """Set the current scan label to the supplied value."""
        if self.rec_thread is None: return ("fail","No active capture thread. Please start one using capture_start")
        self._my_sensors["label"].set_value(label)
        self.rec_thread.write_label(label)
        return ("ok","Label set to %s" % label)

    @return_reply(Str())
    def request_get_current_file(self, sock, msg):
        """Return the name of the current (or most recent) capture file."""
        if self.rec_thread is not None: self.current_file = self.rec_thread.fname
        if self.current_file is None:
            return ("fail", "No currently active file.")
        return ("ok", self.current_file)

    @return_reply(Str())
    def request_capture_stop(self, sock, msg):
        """Attempts to gracefully shut down current capture thread by sending a SPEAD stop packet to local receiver."""
        logger.warning("Forceable capture stop called (%f)" % (time.time()))
        if self.rec_thread is None:
            return ("ok","Thread was already stopped.")
        tx = spead.Transmitter(spead.TransportUDPtx('localhost',7148))
        tx.end()
        time.sleep(2)
         # wait for thread to settle...
        self.rec_thread.join()
        self._my_sensors["capture-active"].set_value(0)
        smsg = "Capture stoppped at %s" % time.ctime()
        activitylogger.info(smsg)
        return ("ok", smsg)

    @return_reply(Str())
    def request_capture_done(self, sock, msg):
        """Closes the current capture file and renames it for use by augment."""
        logger.info("Capture done called at %f (%s)" % (time.time(), time.ctime()))
        if self.rec_thread is None:
            return ("fail","No existing capture session.")
        self.current_file = self.rec_thread.fname
        if self.rec_thread.is_alive():
            time.sleep(2 * self.rec_thread.int_time)
             # the stop packet will only be sent at the end of the next dump
            if self.rec_thread.is_alive():
                 # capture thread is persistent...
                self.request_capture_stop(sock, msg)
                 # perform a hard stop if we have not stopped yet.
        self.rec_thread.close_file()
         # no further correspondence will be entered into
        self.rec_thread = None
         # we are done with the capture thread
        if self.current_file is None:
            logger.warning("File was already closed. Not renaming...")
            return ("ok","File was already closed.")
        output_file = self.current_file[:self.current_file.find(".writing.h5")] + ".unaugmented.h5"
        try:
            os.rename(self.current_file, output_file)
        except Exception, e:
            logger.error("Failed to rename output file %s" % e)
            return ("fail","Failed to rename output file from %s to %s." % (self.current_file, output_file))
        finally:
            self.current_file = None
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
    ctl = None
    if opts.include_ctl:
        try:
            import katcorelib
        except ImportError:
            print "katcorelib is not available on this host. please run script using --include_ctl=false"
            sys.exit(0)
        ctl = small_build(opts.system)

    found_katconf = False
    if found_katconf:
        # Setup configuration source
        katconf.set_config(katconf.environ(opts.sysconfig))
        # set up Python logging
        katconf.configure_logging(opts.logging)

    logger = logging.getLogger("kat.k7capture")
    if not found_katconf:
        if opts.logging is None: logger.warning("Defaulting to log level INFO. To override use the -l [DEBUG | INFO | WARNING | ERROR] switch.")
        logger.setLevel(logging.INFO if opts.logging is None else opts.logging)

    spead.logger.setLevel(logging.WARNING)
     # configure SPEAD to display warnings about dropped packets etc...

    sp.ProcBlock.logger = logger
     # logger ref for use in the signal processing routines
    activitylogger = logging.getLogger("activity")
    activitylogger.setLevel(logging.INFO)
    activitylogger.info("Activity logging started")

    restart_queue = Queue.Queue()
    server = CaptureDeviceServer(opts.sdisp_ips, opts.sdisp_port, opts.host, opts.port)
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

