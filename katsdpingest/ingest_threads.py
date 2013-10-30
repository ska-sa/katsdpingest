#!/usr/bin/python

# Threads for ingesting data and meta-data in order to produce a complete HDF5 file for further
# processing.
#
# Currently has a CBFIngest and TMIngest class
#
# Details on these are provided in the class documentation
import numpy as np
import threading
import spead
import time
import katcapture.sigproc as sp

mapping = {'xeng_raw':'/Data/correlator_data',
           'timestamp':'/Data/raw_timestamps'}
 # maps SPEAD element names to HDF5 paths
timestamps_dataset = '/Data/timestamps'
flags_dataset = '/Markup/flags'
correlator_map = '/MetaData/Configuration/Correlator/'
observation_map = '/MetaData/Configuration/Observation/'
 # default path for things that are not mentioned above
obs_sensors = ['obs_ants','obs_script_arguments','obs_description','obs_experiment_id','obs_script_name','obs_observer','obs_starttime','obs_endtime','obs_status']
 # sensors to update based on observation parameters set by the executing script
sdisp_ips = {}
 # dict storing the configured signal destination ip addresses

class TMIngest(threading.Thread):
    """The TM Ingest class receives meta-data updates in the form
    of sensor information from the TM via SPEAD. It uses these to
    update a model of the telescope that is specific to the current
    ingest configuration (subarray)."""
    def __init__(self, meta_data_port, h5_file, model, logger):
        self.logger = logger
        self.meta_data_port = meta_data_port
        self.h5_file = h5_file
        self.model = model
        self.ig = None
        threading.Thread.__init__(self)

    def run(self):
        self.logger.info("Meta-data reception on port %i" % self.meta_data_port)
        self.ig = spead.ItemGroup()
        rx_md = spead.TransportUDPrx(self.meta_data_port)

        for heap in spead.iterheaps(rx_md):
            self.ig.update(heap)
            self.model.update_from_ig(self.ig, debug=True)

        self.logger.info("Meta-data reception complete at %f" % time.time())

    def write_model(self):
        """Write the current model into the HDF5 file."""
        valid = self.model.is_valid(timespec=5)
         # check to see if we are valid up until the last 5 seconds
        if not valid: self.logger.warning("Model is not valid. Writing to disk anyway.")
        else: self.model.write_h5(self.h5_file)
         # write the model

class CBFIngest(threading.Thread):
    def __init__(self, data_port, h5_file, my_sensors, logger):
        self.logger = logger
        self.data_port = data_port
        self.h5_file = h5_file
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
        self.meta = {}
        self.ig_sd = spead.ItemGroup()
        self.cpref = None
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
        self.scale = sp.Scale(self.data_scale_factor)
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
        if self.meta.has_key('bls_ordering') and self.meta.has_key('bandwidth') and self.meta.has_key('n_chans'):
            self.ig_sd.add_item(name=('bls_ordering'), id=0x100C, description="Mapping of antenna/pol pairs to data output products.", init_val=self.meta['bls_ordering'])
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

    def write_obs_param(self, key_string, value_string):
        self.h5_file['/MetaData/Configuration/Observation'].attrs[key_string.replace("-","_")] = value_string

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
        f = None
        self.status_sensor.set_value("idle")
        dump_size = 0
        datasets = {}
        datasets_index = {}
        meta_required = set(['n_chans','n_accs','n_bls','bls_ordering','bandwidth'])
         # we need these bits of meta data before being able to assemble and transmit signal display data
        meta_desired = ['int_time','center_freq']
         # if we find these, then what hey :)
        current_dbe_target = ''
        dbe_target_since = 0.0
        current_ant_activities = {}
        ant_activities_since = {}
         # track the current DBE target and antenna activities via sensor updates
        sd_slots = None
        sd_timestamp = None
        for heap in spead.iterheaps(rx):
            if idx == 0:
                f = self.h5_file
                self.status_sensor.set_value("capturing")
            ig.update(heap)
            for name in ig.keys():
                item = ig.get_item(name)
                if not item._changed and datasets.has_key(name): continue
                 # the item is not marked as changed, and we have a record for it
                if name in meta_desired:
                    self.logger.info("Meta data received (desired) %s: %s => %s" % (time.ctime(), name, str(ig[name])))
                    self.meta[name] = ig[name]
                    if name == 'center_freq':
                        #self.center_freq = self.meta[name]
                        self._my_sensors["spead-center-freq"].set_value(self.meta[name])
                    if name == 'int_time' and self.int_time == 1:
                        self.int_time = self.meta[name]
                        self._my_sensors["spead-dump-period"].set_value(self.meta[name])
                if name in meta_required:
                    self.logger.info("Meta data received (required) %s: %s => %s" % (time.ctime(), name, str(ig[name])))
                    self.meta[name] = ig[name]
                    meta_required.remove(name)
                    if name == 'n_accs':
                        self.data_scale_factor = np.float32(ig[name])
                        self.scale.scale_factor = self.data_scale_factor
                        self.write_process_log(*self.scale.description())
                        self._my_sensors["spead-accum-per-dump"].set_value(self.meta[name])
                        self.logger.debug("Scale factor set to: %f\n" % self.data_scale_factor)
                    if name == 'n_chans':
                        self._my_sensors["spead-num-chans"].set_value(self.meta[name])
                    if name == 'n_bls':
                        self._my_sensors["spead-num-bls"].set_value(self.meta[name])
                    if not meta_required:
                        self.set_baseline_mask()
                         # we first set the baseline mask in order to have the correct number of baselines
                        if self.data_scale_factor >= 1:
                            self.van_vleck = sp.VanVleck(self.data_scale_factor, bls_ordering=self.meta['bls_ordering'])
                            self.write_process_log(*self.van_vleck.description())
                            self.logger.info("Initialised Van Vleck correction using scale factor %i\n" % self.data_scale_factor)
                        else:
                            self.logger.error("Refused to initialize Van Vleck correction with scale factor of 0. Van Vleck will *NOT* be applied for this capture session.")
                            self.data_scale_factor=1
                             # at least ensure that valid (but unscaled) data can be written...
                        self.cpref = CorrProdRef(bls_ordering=self.meta['bls_ordering'])
                        self.ant_gains = sp.AntennaGains(bls_ordering=self.meta['bls_ordering'])
                         # since we now know the baseline ordering we can create the Van Vleck correction and antenna gain blocks
                        self.sd_frame = np.zeros((self.meta['n_chans'],len(self.baseline_mask),2),dtype=np.float32)
                        self.logger.debug("Initialised sd frame to shape %s" % str(self.sd_frame.shape))
                        meta_required = set(['n_chans','n_bls','bls_ordering','bandwidth'])
                        sd_slots = None
                if name.startswith('sensor_'):
                    if not item._changed: continue
                    self.logger.info("Sensor data received %s: %s => %s" % (time.ctime(), name, str(ig[name])))
                    sensor_name = name.partition('_')[2]
                    update = item.get_value()[0]
                    part1, sep, tail = update.partition(' ')
                    part2, sep, part3 = tail.partition(' ')
                    update_timestamp, update_status, update_value = float(part1), part2, eval(part3, {})
                     # unpack sensor name and split update into timestamp + status + value
                    if sensor_name == 'dbe7_target':
                        current_dbe_target = update_value
                        dbe_target_since = update_timestamp
                    elif sensor_name.endswith('activity'):
                        ant_name = sensor_name.partition('_')[0]
                        current_ant_activities[ant_name] = update_value
                        ant_activities_since[ant_name] = update_timestamp
                    self.logger.debug("Updated sensor %s: DBE target '%s' since %r, %s" %
                                 (sensor_name, current_dbe_target, dbe_target_since,
                                  ', '.join([("antenna '%s' did '%s' since %r" % (ant, current_ant_activities[ant], ant_activities_since[ant]))
                                             for ant in current_ant_activities])))
                    item._changed = False
                    continue
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
                    self.logger.debug("Creating dataset for name: %s, shape: %s, dtype: %s" % (name, str(new_shape), dtype))
                    if new_shape == [1]:
                        new_shape = []
                    f.create_dataset(self.remap(name),[1] + new_shape, maxshape=[None] + new_shape, dtype=dtype)
                    dump_size += np.multiply.reduce(shape) * dtype.itemsize
                    datasets[name] = self.h5_file[self.remap(name)]
                    datasets_index[name] = 0
                    if name == 'timestamp':
                        item._changed = False
                    if name == 'xeng_raw':
                        f.create_dataset(flags_dataset, [1] + new_shape[:-1], maxshape=[None] + new_shape[:-1], dtype=np.uint8)
                    if not item._changed:
                        continue
                     # if we built from an empty descriptor
                else:
                    if not item._changed:
                        continue
                    self.h5_file[self.remap(name)].resize(datasets_index[name]+1, axis=0)
                    if name == 'xeng_raw':
                        self.h5_file[flags_dataset].resize(datasets_index[name]+1, axis=0)
                if name.startswith("xeng_raw"):
                     # prepare signal display data handling if required
                    if self.sd_frame is not None:
                        sd_timestamp = ig['sync_time'] + (ig['timestamp'] / ig['scale_factor_timestamp'])
                        if sd_slots is None:
                            self.sd_frame.dtype = np.dtype(np.float32) # if self.acc_scale else ig[name].dtype
                             # make sure we have the right dtype for the sd data
                            sd_slots = np.zeros(self.meta['n_chans']/ig[name].shape[0])
                            self.send_sd_metadata()
                        self.ig_sd['sd_timestamp'] = int(sd_timestamp * 100)

                    sp.ProcBlock.current = ig[name][...,self.baseline_mask,:]
                     # update our data pointer. at this stage dtype is int32 and shape (channels, baselines, 2)
                    self.scale.proc()
                     # scale the data
                    if self.van_vleck is not None:
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
                    self.h5_file[flags_dataset][datasets_index[name]] = flags
                     # write flags to file
                    if self.ant_gains is not None:
                        self.ant_gains.proc(current_dbe_target, dbe_target_since, current_ant_activities, ant_activities_since,
                                            self._script_ants.replace(" ","").split(",") if self._script_ants is not None else None,
                                            self.center_freq, self.meta['bandwidth'], self._my_sensors)
                    self.h5_file[self.remap(name)][datasets_index[name]] = sp.ProcBlock.current.view(np.float32).reshape(list(sp.ProcBlock.current.shape) + [2])[np.newaxis,...]
                     # write data to file (with temporary remap as mentioned above...)
                    if self.sd_frame is not None:
                        self.logger.info("Sending signal display frame with timestamp %i (local: %f). %s." % (sd_timestamp, time.time(), \
                                    "Unscaled" if not self.acc_scale else "Scaled by %i" % self.data_scale_factor))
                        self.ig_sd['sd_data'] = self.h5_file[self.remap(name)][datasets_index[name]]
                         # send out a copy of the data we are writing to disk. In the future this will need to be rate limited to some extent
                        self.ig_sd['sd_flags'] = flags
                         # send out RFI flags with the data
                        self.send_sd_data(self.ig_sd.get_heap())
                else:
                    self.h5_file[self.remap(name)][datasets_index[name]] = ig[name]
                if name == 'timestamp':
                    try:
                        current_ts = ig['sync_time'] + (ig['timestamp'] / ig['scale_factor_timestamp'])
                        self._my_sensors["last-dump-timestamp"].set_value(current_ts)
                        self.timestamps.append(current_ts)
                         # temporarily store derived timestamps
                    except KeyError:
                        self.h5_file[timestamps_dataset][datasets_index[name]] = -1.0
                datasets_index[name] += 1
                item._changed = False
                  # we have dealt with this item so continue...
            if idx==0 and self.ctl is not None:
                # add script metadata after receiving first frame. This should ensure that the values pulled are fresh.
                for s in script_sensors:
                    self.h5_file[observation_map].attrs[s] = getattr(kat.sys.sensor, s).get_value()
                self.logger.info("Added initial observation sensor values...\n")
            idx+=1
            f.flush()
            self.pkt_sensor.set_value(idx)

        if self.baseline_mask is not None:
            del self.h5_file[self.remap('bls_ordering')]
            del datasets_index['bls_ordering']
            self.h5_file[correlator_map].attrs['bls_ordering'] = self.meta['bls_ordering']
             # we have generated a new baseline mask, so fix bls_ordering attribute...
        self.logger.info("Capture complete at %f" % time.time())
        self.logger.debug("\nProcessing Blocks\n=================\n%s\n%s\n" % (self.scale,self.rfi))
        self.status_sensor.set_value("complete")
