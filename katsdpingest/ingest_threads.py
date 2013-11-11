#!/usr/bin/python

# Threads for ingesting data and meta-data in order to produce a complete HDF5 file for further
# processing.
#
# Currently has a CBFIngest and CAMIngest class
#
# Details on these are provided in the class documentation
import numpy as np
import threading
import spead
import time
import copy
import katsdpingest.sigproc as sp
import logging

timestamps_dataset = '/Data/timestamps'
flags_dataset = '/Markup/flags'
cbf_data_dataset = '/Data/correlator_data'
correlator_map = '/MetaData/Configuration/Correlator/'
observation_map = '/MetaData/Configuration/Observation/'
 # default path for things that are not mentioned above
obs_sensors = ['obs_ants','obs_script_arguments','obs_description','obs_experiment_id','obs_script_name','obs_observer','obs_starttime','obs_endtime','obs_status']
 # sensors to update based on observation parameters set by the executing script
sdisp_ips = {}
 # dict storing the configured signal destination ip addresses

class CAMIngest(threading.Thread):
    """The CAM Ingest class receives meta-data updates in the form
    of sensor information from the CAM via SPEAD. It uses these to
    update a model of the telescope that is specific to the current
    ingest configuration (subarray)."""
    def __init__(self, meta_data_port, h5_file, model, logger):
        self.logger = logger
        self.meta_data_port = meta_data_port
        self.h5_file = h5_file
        self.model = model
        self.ig = None
        threading.Thread.__init__(self)

    def enable_debug(self, debug):
        if debug: self.logger.setLevel(logging.DEBUG)
        else: self.logger.setLevel(logging.INFO)

    def run(self):
        self.logger.info("Meta-data reception on port %i" % self.meta_data_port)
        self.ig = spead.ItemGroup()
        rx_md = spead.TransportUDPrx(self.meta_data_port)

        for heap in spead.iterheaps(rx_md):
            self.ig.update(heap)
            self.model.update_from_ig(self.ig)

        self.logger.info("CAM ingest thread complete at %f" % time.time())

class CBFIngest(threading.Thread):
    def __init__(self, data_port, h5_file, my_sensors, model, cbf_name, logger):
        ## TODO: remove my_sensors and rather use the model to drive local sensor updates
        self.logger = logger
        self.data_port = data_port
        self.h5_file = h5_file
        self.model = model
        self.cbf_name = cbf_name
        self.cbf_component = self.model.components[self.cbf_name]
        self.cbf_attr = self.cbf_component.attributes

        self.data_scale_factor = 1.0
        self.acc_scale = True
        self._label_idx = 0
        self._log_idx = 0
        self._process_log_idx = 0
        self._my_sensors = my_sensors
        self.pkt_sensor = self._my_sensors['packets-captured']
        self.status_sensor = self._my_sensors['status']
        self.status_sensor.set_value("init")
        self._sd_metadata = None
        self.sdisp_ips = {}
        self._sd_count = 0
        self.center_freq = 0
        self.ig_sd = spead.ItemGroup()
        self.timestamps = []
         # temporary timestamp store
        self.int_time = 1.0
         # default integration time in seconds. Updated by SPEAD metadata on stream initiation.
        self.sd_frame = None
        self.baseline_mask = None
         # by default record all baselines
        self._script_ants = None
         # a reference to the antennas requested from the current script
        #### Initialise processing blocks used
        self.scale = sp.Scale(1.0)
         # at this stage the scale factor is unknown
        self.rfi = sp.RFIThreshold2()
         # basic rfi thresholding flagger
        self.flags_description = [[nm,self.rfi.flag_descriptions[i]] for (i,nm) in enumerate(self.rfi.flag_names)]
         # an array describing the flags produced by the rfi flagger
        self.h5_file['/Markup'].create_dataset('flags_description',data=self.flags_description)
         # insert flags descriptions into output file
        self.van_vleck = self.ant_gains = None
         # defer creation until we know baseline ordering
        #### Done with blocks
        self.write_process_log(*self.rfi.description())
        threading.Thread.__init__(self)

    def enable_debug(self, debug):
        if debug: self.logger.setLevel(logging.DEBUG)
        else: self.logger.setLevel(logging.INFO)

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
        if self.sd_frame is not None:
            self.ig_sd.add_item(name=('sd_data'),id=(0x3501), description="Combined raw data from all x engines.", ndarray=(self.sd_frame.dtype,self.sd_frame.shape))
            self.ig_sd.add_item(name=('sd_flags'),id=(0x3503), description="8bit packed flags for each data point.", ndarray=(np.dtype(np.uint8), self.sd_frame.shape[:-1]))
        self.ig_sd.add_item(name="center_freq",id=0x1011, description="The center frequency of the DBE in Hz, 64-bit IEEE floating-point number.",
                            shape=[],fmt=spead.mkfmt(('f',64)), init_val=self.center_freq)
        self.ig_sd.add_item(name=('sd_timestamp'), id=0x3502, description='Timestamp of this sd frame in centiseconds since epoch (40 bit limitation).',
                            shape=[], fmt=spead.mkfmt(('u',spead.ADDRSIZE)))
        self.ig_sd.add_item(name=('bls_ordering'), id=0x100C, description="Mapping of antenna/pol pairs to data output products.", init_val=self.cbf_attr['bls_ordering'].value)
        self.ig_sd.add_item(name="bandwidth",id=0x1013, description="The analogue bandwidth of the digitally processed signal in Hz.",
                            shape=[],fmt=spead.mkfmt(('f',64)), init_val=self.cbf_attr['bandwidth'].value)
        self.ig_sd.add_item(name="n_chans",id=0x1009, description="The total number of frequency channels present in any integration.",
                            shape=[], fmt=spead.mkfmt(('u',spead.ADDRSIZE)), init_val=self.cbf_attr['n_chans'].value)
        return copy.deepcopy(self.ig_sd.get_heap())

    def set_baseline_mask(self, bls_ordering):
        """Uses the _script_ants variable to set a baseline mask.
        This only works if script_ants has been set by an external process between capture_done and capture_start."""
        new_bls = bls_ordering
        if self._script_ants is not None:
            logger.info("Using script-ants (%s) as a custom baseline mask..." % self._script_ants)
            ants = self._script_ants.replace(" ","").split(",")
            if len(ants) > 0:
                b = bls_ordering.tolist()
                self.baseline_mask = [b.index(pair) for pair in b if pair[0][:-1] in ants and pair[1][:-1] in ants]
                new_bls = np.array([b[idx] for idx in self.baseline_mask])
                 # we need to recalculate the bls ordering as well...
        return new_bls

    def send_sd_metadata(self):
        self._sd_metadata = self._update_sd_metadata()
        if self._sd_metadata is not None:
            for tx in self.sdisp_ips.itervalues():
                mdata = copy.deepcopy(self._sd_metadata)
                tx.send_heap(mdata)

    def remap(self, name):
        return name in mapping and mapping[name] or correlator_map + name

    def write_log(self, log):
        """Write a log value directly into the current hdf5 file."""
        if self._log_idx > 0:
            self.h5_file['/History/script_log'].resize(self._log_idx+1,axis=0)
        self.h5_file['/History/script_log'][self._log_idx] = (time.time(), log)
        self._log_idx += 1

    def write_process_log(self, process, args, revision):
        """Write an entry into the process log."""
        if self._process_log_idx > 0:
            self.h5_file['/History/process_log'].resize(self._process_log_idx+1, axis=0)
        self.h5_file['/History/process_log'][self._process_log_idx] = (process, args, revision)
        self._process_log_idx += 1

    def write_timestamps(self):
        """Write the accumulated timestamps into a dataset.
        Previously these timestamps were written alongside each received data frame, but this
        results in a highly fragmented timestamp array. This in turns leads to exceptionally long load
        times for this dataset, even though it contains very little data. By deferring writing, we can
        instead locate the timestamp data contiguously on disk and thus obviate the read overhead.

        As this MUST be called before the file is closed, it may get called multiple times as security to
        ensure that it is done - it is therefore safe to call multiple times."""
        if self.h5_file is not None:
            if timestamps_dataset not in self.h5_file:
            # explicit check for existence of timestamp dataset - we could rely on h5py exceptions, but these change
            # regularly - hence this check.
                if self.timestamps:
                    self.h5_file.create_dataset(timestamps_dataset,data=np.array(self.timestamps))
                    # create timestamp array before closing file. This means that it will be contiguous and hence much faster to read than if it was
                    # distributed across the entire file.
                else:
                    self.logger.warning("H5 file contains no data and hence no timestamps")
                    # exception if there is no data (and hence no timestamps) in the file.
        else: self.logger.warning("Write timestamps called, but h5 file already closed. No timestamps will be written.")

    def write_label(self, label):
        """Write a sensor value directly into the current hdf5 at the specified locations.
           Note that this will create a new HDF5 file if one does not already exist..."""
        if self._label_idx > 0:
            self.h5_file['/Markup/labels'].resize(self._label_idx+1,axis=0)
        self.h5_file['/Markup/labels'][self._label_idx] = (time.time(), label)
        self._label_idx += 1

    def finalise(self):
        """Write any final information to file and mark file as not current."""
        if self.h5_file is not None:
            self.write_timestamps()
        self.h5_file = None

    def drop_sdisp_ip(self, ip):
        self.logger.info("Removing ip %s from the signal display list." % (ip))
        del self.sdisp_ips[ip]

    def add_sdisp_ip(self, ip, port):
        self.logger.info("Adding %s:%s to signal display list. Starting transport..." % (ip,port))
        self.sdisp_ips[ip] = spead.Transmitter(spead.TransportUDPtx(ip, port))
        if self._sd_metadata is not None:
            mdata = copy.deepcopy(self._sd_metadata)
            self.sdisp_ips[ip].send_heap(mdata)
             # new connection requires headers...

    def run(self):
        self.logger.info("Initalising SPEAD transports at %f" % time.time())
        self.logger.info("Data reception on port %i" % self.data_port)
        rx = spead.TransportUDPrx(self.data_port, pkt_count=1024, buffer_size=51200000)
        ig = spead.ItemGroup()
        idx = 0
        self.status_sensor.set_value("idle")
        dump_size = 0
        datasets = {}
        datasets_index = {}
        current_dbe_target = ''
        dbe_target_since = 0.0
        current_ant_activities = {}
        ant_activities_since = {}
         # track the current DBE target and antenna activities via sensor updates
        sd_slots = None
        sd_timestamp = None

        for heap in spead.iterheaps(rx):
            if idx == 0:
                self.status_sensor.set_value("capturing")

            #### Update the telescope model

            ig.update(heap)
            self.model.update_from_ig(ig, proxy_path=self.cbf_name)
             # any interesting attributes will now end up in the model
             # this means we are only really interested in actual data now
            if not ig._names.has_key('xeng_raw'): self.logger.warning("CBF Data received but either no metadata or xeng_raw group is present"); continue
            if not ig._names.has_key('timestamp'): self.logger.warning("No timestamp received for current data frame - discarding"); continue
            data_ts = ig['timestamp']
            data_item = ig.get_item('xeng_raw')
            if not data_item._changed: self.logger.debug("Xeng_raw is unchanged"); continue
             # we have new data...

             # check to see if our CBF model is valid
             # i.e. make sure any attributes marked as critical are present
            if not self.cbf_component.is_valid(check_sensors=False):
                self.logger.warning("CBF Component Model is not currently valid as critical attribute items are missing. Data will be discarded until these become available.")
                continue

            ##### Configure datasets and other items now that we have complete metedata

            self.baseline_mask = range(self.cbf_attr['n_bls'].value)
             # default mask is to include all known baseline

            if self._script_ants is not None:
             # we need to calculate a baseline_mask to match the specified script_ants
                self.cbf_attr['bls_ordering'].value = self.set_baseline_mask(self.cbf_attr['bls_ordering'].value)

            if idx == 0:
                 # we need to create the raw and timestamp datasets.
                new_shape = list(data_item.shape)
                new_shape[-2] = len(self.baseline_mask)
                self.logger.debug("Creating cbf_data dataset with shape: {0}, dtype: {1}".format(str(new_shape),np.float32))
                self.h5_file.create_dataset(cbf_data_dataset, [1] + new_shape, maxshape=[None] + new_shape, dtype=np.float32)
                self.h5_file.create_dataset(flags_dataset, [1] + new_shape[:-1], maxshape=[None] + new_shape[:-1], dtype=np.uint8)

                 # configure the signal processing blocks
                self.scale.scale_factor = np.float32(self.cbf_attr['n_accs'].value)
                self.write_process_log(*self.scale.description())

                self.van_vleck = sp.VanVleck(np.float32(self.cbf_attr['n_accs'].value), bls_ordering=self.cbf_attr['bls_ordering'].value)
                self.write_process_log(*self.van_vleck.description())
            else:
                 # resize datasets
                self.h5_file[cbf_data_dataset].resize(idx+1, axis=0)
                self.h5_file[flags_dataset].resize(idx+1, axis=0)

            if self.sd_frame is None:
                self.sd_frame = np.zeros((self.cbf_attr['n_chans'].value,len(self.baseline_mask),2),dtype=np.float32)
                 # initialise the signal display data frame

            if sd_slots is None:
                self.sd_frame.dtype = np.dtype(np.float32) # if self.acc_scale else ig[name].dtype
                         # make sure we have the right dtype for the sd data
                sd_slots = np.zeros(self.cbf_attr['n_chans'].value/data_item.shape[0])
                self.send_sd_metadata()

            ##### Generate timestamps
            current_ts = self.cbf_attr['sync_time'].value + (data_ts / self.cbf_attr['scale_factor_timestamp'].value)
            self._my_sensors["last-dump-timestamp"].set_value(current_ts)
            self.timestamps.append(current_ts)

            ##### Perform data processing

            sp.ProcBlock.current = data_item.get_value()[...,self.baseline_mask,:]
             # update our data pointer. at this stage dtype is int32 and shape (channels, baselines, 2)
            self.scale.proc()
             # scale the data
            if False: #self.van_vleck is not None:
                power_before = np.median(sp.ProcBlock.current[:, self.cpref.autos, 0])
                try:
                    self.van_vleck.proc()
                      # in place van vleck correction of the data
                except sp.VanVleckOutOfRangeError:
                    self.logger.warning("Out of range error whilst applying Van Vleck data correction")
                power_after = np.median(sp.ProcBlock.current[:, self.cpref.autos, 0])
                print "Van vleck power correction: %.2f => %.2f (%.2f scaling)\n" % (power_before,power_after,power_after/power_before)
            sp.ProcBlock.current = sp.ProcBlock.current.copy()
             # *** Dont delete this line ***
             # since array uses advanced selection (baseline_mask not contiguous) we may
             # have a changed the striding underneath wich can prevent the array being viewed
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
            self.h5_file[flags_dataset][idx] = flags
             # write flags to file
            self.h5_file[cbf_data_dataset][idx] = sp.ProcBlock.current.view(np.float32).reshape(list(sp.ProcBlock.current.shape) + [2])[np.newaxis,...]
                     # write data to file (with temporary remap as mentioned above...)

            #### Send signal display information
            self.logger.info("Sending signal display frame with timestamp %i (local: %f). %s." % (current_ts, time.time(), \
                        "Unscaled" if not self.acc_scale else "Scaled by %i" % self.data_scale_factor))
            self.ig_sd['sd_timestamp'] = int(current_ts * 100)
            self.ig_sd['sd_data'] = self.h5_file[cbf_data_dataset][idx]
             # send out a copy of the data we are writing to disk. In the future this will need to be rate limited to some extent
             # check that this is from cache, not re-read from disk
            self.ig_sd['sd_flags'] = flags
             # send out RFI flags with the data
            self.send_sd_data(self.ig_sd.get_heap())

            #### Done with writing this frame
            idx += 1
            self.h5_file.flush()
            self.pkt_sensor.set_value(idx)

        #### Stop received.

        self.logger.info("CBF ingest complete at %f" % time.time())
        self.logger.debug("\nProcessing Blocks\n=================\n%s\n%s\n" % (self.scale,self.rfi))
        self.status_sensor.set_value("complete")

