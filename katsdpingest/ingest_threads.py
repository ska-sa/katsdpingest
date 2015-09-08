#!/usr/bin/python

# Threads for ingesting data and meta-data in order to produce a complete HDF5 file for further
# processing.
#
# Currently has a CBFIngest and CAMIngest class
#
# Details on these are provided in the class documentation

import numpy as np
import numpy.lib.stride_tricks
import threading
import spead2
import spead2.send
import spead2.recv
import time
import copy
import katsdpingest.sigproc as sp
import katsdpsigproc.accel as accel
import katsdpsigproc.rfi.device as rfi
from katsdpsigproc import percentile
from katsdpsigproc import maskedsum
import katsdpdisp.data as sdispdata
import katsdptelstate
import logging
import socket
import struct


timestamps_dataset = '/Data/timestamps'
flags_dataset = '/Data/flags'
cbf_data_dataset = '/Data/correlator_data'
sdisp_ips = {}
 # dict storing the configured signal destination ip addresses

CBF_SPEAD_SENSORS = ["flags_xeng_raw"]
 # CBF SPEAD metadata items that should be stored as sensors rather than attributes
 # Schwardt/Merry: Let the debate ensue as to the whole attribute/sensor utility in the first place


class CAMIngest(threading.Thread):
    """The CAM Ingest class receives meta-data updates in the form
    of sensor information from the CAM via SPEAD. It uses these to
    update a model of the telescope that is specific to the current
    ingest configuration (subarray)."""
    def __init__(self, spead_endpoints, h5_file, telstate, model, logger):
        self.logger = logger
        self.spead_endpoints = spead_endpoints
        self.telstate = telstate
        self.h5_file = h5_file
        self.model = model
        self.ig = None
        threading.Thread.__init__(self)

    def enable_debug(self, debug):
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)

    def _update_telstate_from_ig(self, updated):
        for name, item in updated.iteritems():
            try:
                (value_time, status, sensor_value) = item.value.split(" ",2)
                value_time = float(value_time)
                sensor_value = np.safe_eval(sensor_value)
            except ValueError:
             # our update is not a classical sensor triplet of time / status / value
                sensor_value = item.value
                value_time = time.time()
                status = "nominal"
                 # fake up a realistic looking sensor
                if sensor_value == '':
                    self.logger.error("Not inserting empty string into sensor {} due to existing numpy/pickle bug"
                                      .format(item_name))
                     # TODO: once fixed in numpy remove this check
                    continue
            if status == 'unknown':
                self.logger.debug("Sensor {0} received update '{1}' with status 'unknown' (ignored)"
                                  .format(item_name, item.value))
            elif self.telstate is not None:
                # XXX Nasty hack to get SDP onto cbf name for AR1 integration
                item_name = item_name.replace('data_1_', 'cbf_')
                self.telstate.add(item_name, sensor_value, value_time)

    def run(self):
        self.ig = spead2.ItemGroup()
        self.logger.debug("Initalising SPEAD transports at %f" % time.time())
        self.logger.info("CAM SPEAD stream reception on {0}".format([str(x) for x in self.spead_endpoints]))
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
            self.model.update_from_ig(updated)
            self._update_telstate_from_ig(updated)

        self.logger.info("CAM ingest thread complete at %f" % time.time())

class _TimeAverage(object):
    """Manages a collection of dumps that are averaged together at a specific
    cadence. Note that all timestamps in this case are in raw form i.e., ticks
    of the ADC clock since the sync time.

    This object never sees dump contents directly, only timestamps. When a
    timestamp is added that is not part of the current group, :func:`flush`
    is called, which must be overloaded or set to a callback function.

    Parameters
    ----------
    cbf_attr : dict
        CBF attributes (the critical attributes must be present)
    int_time : float
        requested integration time, which will be rounded to a multiple of
        the CBF integration time

    Attributes
    ----------
    interval : int
        length of each group, in timestamp units
    ratio : int
        number of CBF dumps per output dump
    int_time : float
        quantised integration time
    _start_ts : int or NoneType
        Timestamp of first dump in the current group, or `None` if no dumps have been seen
    _ts : list of int
        All timestamps in the current group. Empty only if no dumps have ever been seen
    """
    def __init__(self, cbf_attr, int_time):
        self.ratio = max(1, int(round(int_time / cbf_attr['int_time'].value)))
        self.int_time = self.ratio * cbf_attr['int_time'].value
        # Interval in ADC clock cycles
        clocks = 2 * cbf_attr['n_chans'].value * cbf_attr['n_accs'].value * self.ratio
        self.interval = int(round(clocks * cbf_attr['scale_factor_timestamp'].value / cbf_attr['adc_sample_rate'].value))
        self._start_ts = None
        self._ts = []

    def add_timestamp(self, timestamp):
        """Record that a dump with a given timestamp has arrived and is about to
        be processed. This may call :func:`flush`."""

        if self._start_ts is None:
            # First time: special case
            self._start_ts = timestamp

        if timestamp >= self._start_ts + self.interval:
            self.flush(self._ts)
            skip_groups = (timestamp - self._start_ts) // self.interval
            self._ts = []
            self._start_ts += skip_groups * self.interval
        self._ts.append(timestamp)

    def flush(self, timestamps):
        raise NotImplementedError

    def finish(self):
        """Flush if not empty, and reset to initial state"""
        if self._ts:
            self.flush(self._ts)
        self._start_ts = None
        self._ts = []

def _split_array(x, dtype):
    """Return a view of x which has one extra dimension. Each element is x is
    treated as some number of elements of type `dtype`, whose size must divide
    into the element size of `x`."""
    in_dtype = x.dtype
    out_dtype = np.dtype(dtype)
    if in_dtype.hasobject or out_dtype.hasobject:
        raise ValueError('dtypes containing objects are not supported')
    if in_dtype.itemsize % out_dtype.itemsize != 0:
        raise ValueError('item size does not evenly divide')

    interface = dict(x.__array_interface__)
    if interface.get('mask', None) is not None:
        raise ValueError('masked arrays are not supported')
    interface['shape'] = x.shape + (in_dtype.itemsize // out_dtype.itemsize,)
    if interface['strides'] is not None:
        interface['strides'] = x.strides + (out_dtype.itemsize,)
    interface['typestr'] = out_dtype.str
    interface['descr'] = out_dtype.descr
    return np.asarray(np.lib.stride_tricks.DummyArray(interface, base=x))

def _slot_shape(x, split_dtype=None):
    """Return the dtype and shape of an array as a dict of keys that can be
    passed to :func:`spead2.ItemGroup.add_item`.

    Parameters
    ----------
    x : object
        Array or array-like object with `shape` and `dtype` attributes
    split_dtype : numpy dtype, optional
        If `split_dtype` is specified, it considers the result of passing an array
        shaped like `x` to :func:`_split_array`.

    Returns
    -------
    dtype : numpy dtype
        Data type
    shape : tuple
        Data shape
    """
    dtype = np.dtype(x.dtype)
    shape = tuple(x.shape)
    if split_dtype is not None:
        new_dtype = np.dtype(split_dtype)
        if dtype.itemsize % new_dtype.itemsize != 0:
            raise ValueError('item size does not evenly divide')
        ratio = dtype.itemsize // new_dtype.itemsize
        shape = shape + (ratio,)
        dtype = new_dtype
    return {'dtype': dtype, 'shape': shape}

class CBFIngest(threading.Thread):
    # To avoid excessive autotuning, the following parameters are quantised up
    # to the next element of these lists when generating templates. These
    # lists are also used by ingest_autotune.py for pre-tuning standard
    # configurations.
    tune_antennas = [2, 4, 8, 16]
    tune_channels = [4096, 32768]

    @classmethod
    def _tune_next(cls, value, predef):
        """Return the smallest value in `predef` greater than or equal to
        `value`, or `value` if it is larger than the largest element of
        `predef`."""
        valid = [x for x in predef if x >= value]
        if valid:
            return min(valid)
        else:
            return value

    @classmethod
    def _tune_next_antennas(cls, value):
        """Round `value` up to the next power of 2 (excluding 1)."""
        out = 2
        while out < value:
            out *= 2
        return out

    @classmethod
    def create_proc_template(cls, context, antennas, channels):
        """Create a processing template. This is a potentially slow operation,
        since it invokes autotuning.

        Parameters
        ----------
        context : :class:`katsdpsigproc.cuda.Context` or :class:`katsdpsigproc.opencl.Context`
            Context in which to compile device code
        antennas : int
            Number of antennas, *after* any masking
        channels : int
            Number of channels, *prior* to any clipping
        """
        # Quantise to reduce number of options to autotune
        max_antennas = cls._tune_next_antennas(antennas)
        max_channels = cls._tune_next(channels, cls.tune_channels)

        flag_value = 1 << sp.IngestTemplate.flag_names.index('ingest_rfi')
        background_template = rfi.BackgroundMedianFilterDeviceTemplate(context, width=13)
        noise_est_template = rfi.NoiseEstMADTDeviceTemplate(context, max_channels=max_channels)
        threshold_template = rfi.ThresholdSimpleDeviceTemplate(
                context, transposed=True, flag_value=flag_value)
        flagger_template = rfi.FlaggerDeviceTemplate(
                background_template, noise_est_template, threshold_template)
        n_cross = max_antennas * (max_antennas - 1) // 2
        percentile_sizes = [
                max_antennas, 2 * max_antennas,
                n_cross, 2 * n_cross]
        return sp.IngestTemplate(context, flagger_template, percentile_sizes=percentile_sizes)

    def __init__(self, opts, proc_template,
            h5_file, my_sensors, telstate, model, cbf_name, logger):
        ## TODO: remove my_sensors and rather use the model to drive local sensor updates
        self.logger = logger
        self.cbf_spead_endpoints = opts.cbf_spead
        self.spectral_spead_endpoint = opts.l0_spectral_spead
        self.spectral_spead_rate = opts.l0_spectral_spead_rate
        self.continuum_spead_endpoint = opts.l0_continuum_spead
        self.continuum_spead_rate = opts.l0_continuum_spead_rate
        self.output_int_time = opts.output_int_time
        self.sd_int_time = opts.sd_int_time
        self.cont_factor = opts.continuum_factor
        self.sd_cont_factor = opts.sd_continuum_factor
        self.channels = opts.cbf_channels
        if opts.antenna_mask is not None:
            self.antenna_mask = set(opts.antenna_mask)
        else:
            self.antenna_mask = None
        self.proc_template = proc_template
        self.h5_file = h5_file
        self.telstate = telstate
        self.model = model
        self.cbf_name = cbf_name
        self.cbf_component = self.model.components[self.cbf_name]
        self.cbf_attr = self.cbf_component.attributes

        self.maskedsum_weightedmask=[]

        self._process_log_idx = 0
        self._my_sensors = my_sensors
        self.pkt_sensor = self._my_sensors['packets-captured']
        self.status_sensor = self._my_sensors['status']
        self.status_sensor.set_value("init")
        self._sd_metadata = None
        self.sdisp_ips = {}
        self._sd_count = 0
        self.center_freq = 0
        self.ig_sd = None
        self.timestamps = []
         # temporary timestamp store
        #### Initialise processing blocks used
        self.command_queue = proc_template.context.create_command_queue()
        self.proc = None    # Instantiation of the template delayed until data shape is known (TODO: can do it here)
        self.flags_description = zip(self.proc_template.flag_names, self.proc_template.flag_descriptions)
         # an array describing the flags produced by the rfi flagger
        if self.h5_file is not None:
            self.h5_file['/Data'].create_dataset('flags_description',data=self.flags_description)
         # insert flags descriptions into output file
        #### Done with blocks
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

    def _create_ig_sd(self):
        """Create the item group for signal displays. Static values are also filled in."""
        flavour = spead2.Flavour(4, 64, 40, spead2.BUG_COMPAT_PYSPEAD_0_5_2)
        inline_format = [('u', flavour.heap_address_bits)]
        self.ig_sd = spead2.send.ItemGroup(flavour=flavour)
         # we need to clear the descriptor so as not to accidently send a signal display frame twice...
        self.ig_sd.add_item(name=('sd_data'),id=(0x3501), description="Combined raw data from all x engines.",
            **_slot_shape(self.proc.buffer('sd_spec_vis'), np.float32))
        self.ig_sd.add_item(name=('sd_blmxdata'), id=0x3507, description="Reduced data for baseline matrix.",
            **_slot_shape(self.proc.buffer('sd_cont_vis'), np.float32))
        self.ig_sd.add_item(name=('sd_flags'),id=(0x3503), description="8bit packed flags for each data point.",
            **_slot_shape(self.proc.buffer('sd_spec_flags')))
        self.ig_sd.add_item(name=('sd_blmxflags'),id=(0x3508), description="Reduced data flags for baseline matrix.",
            **_slot_shape(self.proc.buffer('sd_cont_flags')))
        self.ig_sd.add_item(name=('sd_timeseries'),id=(0x3504), description="Computed timeseries.",
            **_slot_shape(self.proc.buffer('timeseries'), np.float32))
        n_perc_signals = 0
        perc_idx = 0
        while True:
            try:
                n_perc_signals += self.proc.buffer('percentile{0}'.format(perc_idx)).shape[0]
                perc_idx += 1
            except KeyError:
                break
        self.ig_sd.add_item(name=('sd_percspectrum'),id=(0x3505), description="Percentiles of spectrum data.",
            dtype=np.float32,shape=(self.proc.buffer('percentile0').shape[1],n_perc_signals))
        self.ig_sd.add_item(name=('sd_percspectrumflags'),id=(0x3506), description="Flags for percentiles of spectrum.",
            dtype=np.uint8,shape=(self.proc.buffer('percentile0').shape[1],n_perc_signals))
        self.ig_sd.add_item(name="center_freq",id=0x1011, description="The center frequency of the DBE in Hz, 64-bit IEEE floating-point number.",
                            shape=[],dtype=np.float64, value=self.center_freq)
        self.ig_sd.add_item(name=('sd_timestamp'), id=0x3502, description='Timestamp of this sd frame in centiseconds since epoch (40 bit limitation).',
                            shape=[], dtype=None, format=inline_format)
        bls_ordering = np.asarray(self.cbf_attr['bls_ordering'].value)
        self.ig_sd.add_item(name=('bls_ordering'), id=0x100C, description="Mapping of antenna/pol pairs to data output products.",
            shape=bls_ordering.shape, dtype=bls_ordering.dtype, value=bls_ordering)
        self.ig_sd.add_item(name="bandwidth",id=0x1013, description="The analogue bandwidth of the digitally processed signal in Hz.",
                            shape=[], dtype=np.float64, value=self.cbf_attr['bandwidth'].value)
        self.ig_sd.add_item(name="n_chans",id=0x1009, description="The total number of frequency channels present in any integration.",
                            shape=[], dtype=None, format=inline_format, value=self.cbf_attr['n_chans'].value)

    @classmethod
    def baseline_permutation(cls, bls_ordering, antenna_mask=None):
        """Construct a permutation to place the baselines into the desired
        order for internal processing.

        Parameters
        ----------
        bls_ordering : list of pairs of strings
            Names of inputs in current ordering
        antenna_mask : set of strings, optional
            Antennas to retain in the permutation (without polarisation suffix)

        Returns
        -------
        permutation : list
            The permutation specifying the reordering. Element *i* indicates
            the position in the new order corresponding to element *i* of
            the original order, or -1 if the baseline was masked out.
        new_ordering : ndarray
            Replacement ordering, in the same format as `bls_ordering`
        """
        def keep(baseline):
            if antenna_mask:
                input1, input2 = baseline
                return input1[:-1] in antenna_mask and input2[:-1] in antenna_mask
            else:
                return True

        def key(item):
            input1, input2 = item[1]
            pol1 = input1[-1]
            pol2 = input2[-1]
            return (input1[:-1] != input2[:-1], pol1 != pol2, pol1, pol2)

        # Eliminate baselines not covered by antenna_mask
        filtered = [x for x in enumerate(bls_ordering) if keep(x[1])]
        # Sort what's left
        reordered = sorted(filtered, key=key)
        # reordered contains the mapping from new position to original
        # position, but we need the inverse.
        permutation = [-1] * len(bls_ordering)
        for i in range(len(reordered)):
            permutation[reordered[i][0]] = i
        return permutation, np.array([x[1] for x in reordered])

    def send_sd_metadata(self):
        if self.ig_sd is not None:
            heap = self.ig_sd.get_heap(descriptors='all', data='none')
            for tx in self.sdisp_ips.itervalues():
                tx.send_heap(heap)

    def write_process_log(self, process, args, revision):
        """Write an entry into the process log."""
        if self.h5_file is not None:
            if self._process_log_idx > 0:
                self.h5_file['/History/process_log'].resize(self._process_log_idx+1, axis=0)
            self.h5_file['/History/process_log'][self._process_log_idx] = (process, args, revision)
            self._process_log_idx += 1
        self.logger.info("Processing element: %s (%s) %d", process, args, revision)

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

    def finalise(self):
        """Write any final information to file and mark file as not current."""
        if self.h5_file is not None:
            self.write_timestamps()
        self.h5_file = None

    def drop_sdisp_ip(self, ip):
        self.logger.info("Removing ip %s from the signal display list." % (ip))
        del self.sdisp_ips[ip]

    def add_sdisp_ip(self, ip, port):
        # TODO: rate limiting in config
        config = spead2.send.StreamConfig(max_packet_size=9172)
        self.logger.info("Adding %s:%s to signal display list. Starting stream..." % (ip,port))
        self.sdisp_ips[ip] = spead2.send.UdpStream(spead2.ThreadPool(), ip, port, config)
        if self.ig_sd is not None:
            self.sdisp_ips[ip].send_heap(self.ig_sd.get_heap(descriptors='all', data='all'))
             # new connection requires headers...

    def set_timeseries_mask(self,maskstr):
        self.logger.info("Setting timeseries mask to %s" % (maskstr))
        self.maskedsum_weightedmask = sdispdata.parse_timeseries_mask(maskstr,self.channels)[1]

    def _send_visibilities(self, tx, ig, vis, flags, ts_rel):
        # Create items on first use. This is simpler than figuring out the
        # correct shapes ahead of time.
        if 'correlator_data' not in ig:
            ig.add_item(id=None, name='correlator_data', description="Visibilities",
                    shape=vis.shape, dtype=vis.dtype)
            ig.add_item(id=None, name='flags', description="Flags for visibilities",
                    shape=flags.shape, dtype=flags.dtype)
            ig.add_item(id=None, name='timestamp', description="Seconds since sync time",
                    shape=[], dtype=np.float64)
        ig['correlator_data'].value = vis
        ig['flags'].value = flags
        ig['timestamp'].value = ts_rel
        tx.send_heap(ig.get_heap())

    def _append_visibilities(self, vis, flags, ts):
        # resize datasets
        h5_cbf = self.h5_file[cbf_data_dataset]
        h5_flags = self.h5_file[flags_dataset]
        idx = h5_cbf.shape[0]
        h5_cbf.resize(idx+1, axis=0)
        h5_flags.resize(idx+1, axis=0)

        # Complex values are written to file as an extra dimension of size 2,
        # rather than as structs. Revisit this later to see if either the HDF5
        # file format can be changed to store complex data (rather than
        # having a real/imag axis for reals).
        vis = _split_array(vis, np.float32)
        h5_cbf[idx] = vis
        h5_flags[idx] = flags
        self.timestamps.append(ts)
        self.h5_file.flush()

    def _initialise(self, ig_cbf):
        """Initialise variables on reception of the first usable dump."""
        cbf_baselines = len(self.cbf_attr['bls_ordering'].value)
        # Configure the masking and reordering of baselines
        permutation, self.cbf_attr['bls_ordering'].value = \
                self.baseline_permutation(self.cbf_attr['bls_ordering'].value, self.antenna_mask)
        baselines = len(self.cbf_attr['bls_ordering'].value)
        channels = self.cbf_attr['n_chans'].value
        n_accs = self.cbf_attr['n_accs'].value
        self._set_telstate_entry('bls_ordering', self.cbf_attr['bls_ordering'].value)

         # we need to create the raw datasets.
        data_item = ig_cbf['xeng_raw']
        new_shape = list(data_item.shape)
        new_shape[-2] = baselines
        self.logger.info("Creating cbf_data dataset with shape: {0}, dtype: {1}".format(str(new_shape),np.float32))
        if self.h5_file is not None:
            self.h5_file.create_dataset(cbf_data_dataset, [0] + new_shape, maxshape=[None] + new_shape, dtype=np.float32)
            self.h5_file.create_dataset(flags_dataset, [0] + new_shape[:-1], maxshape=[None] + new_shape[:-1], dtype=np.uint8)

        # Configure time averaging
        self._output_avg = _TimeAverage(self.cbf_attr, self.output_int_time)
        self._output_avg.flush = self._flush_output
        self._set_telstate_entry('sdp_l0_int_time', self._output_avg.int_time, add_cbf_prefix=False)
        self.logger.info("Averaging {0} input dumps per output dump".format(self._output_avg.ratio))

        self._sd_avg = _TimeAverage(self.cbf_attr, self.sd_int_time)
        self._sd_avg.flush = self._flush_sd
        self.logger.info("Averaging {0} input dumps per signal display dump".format(self._sd_avg.ratio))

        # configure the signal processing blocks
        collection_products = sdispdata.set_bls(self.cbf_attr['bls_ordering'].value)[0]
        percentile_ranges = []
        for p in collection_products:
            if p:
                start = p[0]
                end = p[-1] + 1
                if not np.array_equal(np.arange(start, end), p):
                    raise ValueError("percentile baselines are not contiguous: {}".format(p))
                percentile_ranges.append((start, end))
            else:
                percentile_ranges.append((0, 0))
        self.maskedsum_weightedmask = sdispdata.parse_timeseries_mask('',channels)[1]

        self.proc = self.proc_template.instantiate(
                self.command_queue, channels, (0, channels), cbf_baselines, baselines,
                self.cont_factor, self.sd_cont_factor, percentile_ranges,
                threshold_args={'n_sigma': 11.0})
        self.proc.set_scale(1.0 / n_accs)
        self.proc.ensure_all_bound()
        self.proc.buffer('permutation').set(self.command_queue, np.asarray(permutation, dtype=np.int16))
        self.proc.start_sum()
        self.proc.start_sd_sum()

        # TODO: configure van_vleck once implemented
        for description in self.proc.descriptions():
            self.write_process_log(*description)

        # initialise the signal display metadata
        self._create_ig_sd()
        self.send_sd_metadata()

    def _flush_output(self, timestamps):
        """Finalise averaging of a group of input dumps and emit an output dump"""
        self.proc.end_sum()
        spec_flags = self.proc.buffer('spec_flags').get(self.command_queue)
        spec_vis = self.proc.buffer('spec_vis').get(self.command_queue)
        cont_flags = self.proc.buffer('cont_flags').get(self.command_queue)
        cont_vis = self.proc.buffer('cont_vis').get(self.command_queue)

        ts_rel = np.mean(timestamps) / self.cbf_attr['scale_factor_timestamp'].value
        ts = self.cbf_attr['sync_time'].value + ts_rel
        self._send_visibilities(self.tx_spectral, self.ig_spectral, spec_vis, spec_flags, ts_rel)
        self._send_visibilities(self.tx_continuum, self.ig_continuum, cont_vis, cont_flags, ts_rel)
        if self.h5_file is not None:
            self._append_visibilities(spec_vis, spec_flags, ts)

        self.logger.info("Finished dump group with raw timestamps {0} (local: {1:.3f})".format(
            timestamps, time.time()))
        #### Prepare for the next group
        self.proc.start_sum()

    def _flush_sd(self, timestamps):
        """Finalise averaging of a group of dumps for signal display, and send
        signal display data to the signal display server"""

        # TODO: this currently gets done every time because it wouldn't be thread-safe
        # to poke the value directly in response to the katcp command. Once the code
        # is redesigned to be single-threaded, push the value directly
        self.proc.buffer('timeseries_weights').set(self.command_queue, self.maskedsum_weightedmask)

        self.proc.end_sd_sum()
        ts_rel = np.mean(timestamps) / self.cbf_attr['scale_factor_timestamp'].value
        ts = self.cbf_attr['sync_time'].value + ts_rel
        cont_vis = self.proc.buffer('sd_cont_vis').get(self.command_queue)
        cont_flags = self.proc.buffer('sd_cont_flags').get(self.command_queue)
        spec_vis = self.proc.buffer('sd_spec_vis').get(self.command_queue)
        spec_flags = self.proc.buffer('sd_spec_flags').get(self.command_queue)
        timeseries = self.proc.buffer('timeseries').get(self.command_queue)
        percentiles = []
        percentiles_flags = []
        for i in range(len(self.proc.percentiles)):
            name = 'percentile{0}'.format(i)
            p = self.proc.buffer(name).get(self.command_queue)
            pflags = self.proc.buffer(name + '_flags').get(self.command_queue)
            percentiles.append(p)
            # Signal display server wants flags duplicated to broadcast with
            # the percentiles
            percentiles_flags.append(np.tile(pflags, (p.shape[0], 1)))

        #populate new datastructure to supersede sd_data etc
        self.ig_sd['sd_timestamp'].value = int(ts * 100)
        self.ig_sd['sd_data'].value = _split_array(spec_vis, np.float32)
        self.ig_sd['sd_flags'].value = spec_flags
        self.ig_sd['sd_blmxdata'].value = _split_array(cont_vis, np.float32)
        self.ig_sd['sd_blmxflags'].value = cont_flags
        self.ig_sd['sd_timeseries'].value = _split_array(timeseries, np.float32)
        self.ig_sd['sd_percspectrum'].value = np.vstack(percentiles).transpose()
        self.ig_sd['sd_percspectrumflags'].value = np.vstack(percentiles_flags).transpose()

         # In the future this will need to be rate limited to some extent
        self.send_sd_data(self.ig_sd.get_heap())
        self.logger.info("Finished SD group with raw timestamps {0} (local: {1:.3f})".format(
            timestamps, time.time()))
        #### Prepare for the next group
        self.proc.start_sd_sum()

    def _set_telstate_entry(self, name, value, add_cbf_prefix=True, attribute=True):
        if self.telstate is not None:
            if add_cbf_prefix:
                name = '{0}_{1}'.format(self.cbf_name, name)
            try:
                self.telstate.add(name, value, immutable=attribute)
            except katsdptelstate.ImmutableKeyError:
                old = self.telstate.get(name)
                if not np.array_equal(old, value):
                    self.logger.warning('Attribute %s could not be set to %s because it is already set to %s',
                            name, value, old)

    def _update_telstate_from_ig(self, updated):
        """Updates the telescope state from new values in the item group."""
        for item_name, item in updated.iteritems():
            # bls_ordering is set later by _initialize, after permuting it.
            # The other items are data rather than metadata, and so do not
            # live in the telescope state.
            if item_name not in ['bls_ordering', 'timestamp', 'xeng_raw']:
                self._set_telstate_entry(item_name, item.value, attribute=(item_name not in CBF_SPEAD_SENSORS))
                 # store as an attribute unless item is in CBF_SPEAD_SENSORS (e.g. flags_xeng_raw)

    def run(self):
        """Thin wrapper than runs the real code and handles some cleanup."""

        # PyCUDA has a bug/limitation regarding cleanup
        # (http://wiki.tiker.net/PyCuda/FrequentlyAskedQuestions) that tends
        # to cause device objects and `HostArray`s to leak. To prevent it,
        # we need to ensure that references are dropped (and hence objects
        # are deleted) with the context being current.
        with self.proc_template.context:
            try:
                self._run()
            finally:
                # These have references to self, causing circular references
                self._output_avg = None
                self._sd_avg = None
                # Drop last references to all the objects
                self.proc = None

    def _run(self):
        """Real implementation of `run`."""
        self.logger.debug("Initialising SPEAD transports at %f" % time.time())
        self.logger.info("CBF SPEAD stream reception on {0}".format([str(x) for x in self.cbf_spead_endpoints]))
        # Socket only used for multicast subscription
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        for endpoint in self.cbf_spead_endpoints:
            if endpoint.multicast_subscribe(sock):
                self.logger.info("Subscribing to multicast address {0}".format(endpoint.host))
            elif endpoint.host != '':
                self.logger.warning("Ignoring non-multicast address {0}".format(endpoint.host))
        thread_pool = spead2.ThreadPool(4)
        rx = spead2.recv.Stream(thread_pool, bug_compat=spead2.BUG_COMPAT_PYSPEAD_0_5_2)
        rx.add_udp_reader(self.cbf_spead_endpoints[0].port)
        self.rx = rx
        self.tx_spectral = spead2.send.UdpStream(
            thread_pool,
            self.spectral_spead_endpoint.host,
            self.spectral_spead_endpoint.port,
            spead2.send.StreamConfig(max_packet_size=9172, rate=self.spectral_spead_rate / 8))
        self.tx_continuum = spead2.send.UdpStream(
            thread_pool,
            self.continuum_spead_endpoint.host,
            self.continuum_spead_endpoint.port,
            spead2.send.StreamConfig(max_packet_size=9172, rate=self.continuum_spead_rate / 8))
        l0_flavour = spead2.Flavour(4, 64, 48, spead2.BUG_COMPAT_PYSPEAD_0_5_2)
        self.ig_spectral = spead2.send.ItemGroup(descriptor_frequency=1, flavour=l0_flavour)
        self.ig_continuum = spead2.send.ItemGroup(descriptor_frequency=1, flavour=l0_flavour)
        ig_cbf = spead2.ItemGroup()
        idx = 0
        self.status_sensor.set_value("idle")
        prev_ts = 0
        datasets = {}
        datasets_index = {}
        current_dbe_target = ''
        dbe_target_since = 0.0
        current_ant_activities = {}
        ant_activities_since = {}
         # track the current DBE target and antenna activities via sensor updates
        self._output_avg = None
        self._sd_avg = None

        for heap in rx:
            st = time.time()
            if idx == 0:
                self.status_sensor.set_value("capturing")

            #### Update the telescope model

            updated = ig_cbf.update(heap)
            self.model.update_from_ig(updated, proxy_path=self.cbf_name)
            self._update_telstate_from_ig(updated)
             # any interesting attributes will now end up in the model
             # this means we are only really interested in actual data now
            if 'xeng_raw' not in updated:
                self.logger.warning("CBF Data received but either no metadata or xeng_raw group is present")
                continue
            if 'timestamp' not in updated:
                self.logger.warning("No timestamp received for current data frame - discarding")
                continue
            data_ts = ig_cbf['timestamp'].value
            data_item = ig_cbf['xeng_raw']
            if data_ts <= prev_ts:
                self.logger.warning("Data timestamps have gone backwards (%d <= %d), dropping heap", data_ts, prev_ts)
                continue
            prev_ts = data_ts
             # we have new data...

             # check to see if our CBF model is valid
             # i.e. make sure any attributes marked as critical are present
            if not self.cbf_component.is_valid(check_sensors=False):
                self.logger.warning("CBF Component Model is not currently valid as critical attribute items are missing. Data will be discarded until these become available.")
                continue

            ##### Configure datasets and other items now that we have complete metadata
            if idx == 0:
                self._initialise(ig_cbf)

            self._output_avg.add_timestamp(data_ts)
            self._sd_avg.add_timestamp(data_ts)

            ##### Generate timestamps
            current_ts_rel = data_ts / self.cbf_attr['scale_factor_timestamp'].value
            current_ts = self.cbf_attr['sync_time'].value + current_ts_rel
            self._my_sensors["last-dump-timestamp"].set_value(current_ts)

            ##### Perform data processing
            self.proc.buffer('vis_in').set(self.command_queue, data_item.value)
            self.proc()

            #### Done with reading this frame
            idx += 1
            self.pkt_sensor.set_value(idx)
            tt = time.time() - st
            self.logger.info("Captured CBF dump with timestamp %i (local: %.3f, process_time: %.2f, index: %i)" % (current_ts, tt+st, tt, idx))

        #### Stop received.

        if self._output_avg is not None:  # Could be None if no heaps arrived
            self._output_avg.finish()
            self._sd_avg.finish()
        self.logger.info("CBF ingest complete at %f" % time.time())
        self.tx_spectral.send_heap(self.ig_spectral.get_end())
        self.tx_spectral = None
        self.tx_continuum.send_heap(self.ig_continuum.get_end())
        self.tx_continuum = None
        self.ig_spectral = None
        self.ig_continuum = None
        if self.proc is not None:   # Could be None if no heaps arrived
            self.logger.debug("\nProcessing Blocks\n=================\n")
            for description in self.proc.descriptions():
                self.logger.debug("\t".join([str(x) for x in description]))
        self.status_sensor.set_value("complete")
