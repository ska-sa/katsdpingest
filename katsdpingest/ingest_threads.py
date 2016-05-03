#!/usr/bin/python

# Threads for ingesting data and meta-data in order to produce a complete HDF5 file for further
# processing.
#
# Currently has a CBFIngest class
#
# Details on these are provided in the class documentation

import numpy as np
import threading
import spead2
import spead2.send
import spead2.recv
import time
import katsdpingest.sigproc as sp
import katsdpsigproc.rfi.device as rfi
from katcp import Sensor
import katsdpdisp.data as sdispdata
import katsdptelstate
import logging
import socket


timestamps_dataset = '/Data/timestamps'
flags_dataset = '/Data/flags'
cbf_data_dataset = '/Data/correlator_data'

# CBF SPEAD metadata items that should be stored as sensors rather than attributes
# Schwardt/Merry: Let the debate ensue as to the whole attribute/sensor utility in the first place
CBF_SPEAD_SENSORS = ["flags_xeng_raw"]
# Attributes that are required for data to be correctly ingested
CBF_CRITICAL_ATTRS = frozenset([
    'adc_sample_rate', 'n_chans', 'n_accs', 'n_bls', 'bls_ordering',
    'bandwidth', 'sync_time', 'int_time', 'scale_factor_timestamp'])


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
        self.ratio = max(1, int(round(int_time / cbf_attr['int_time'])))
        self.int_time = self.ratio * cbf_attr['int_time']
        # Interval in ADC clock cycles
        clocks = 2 * cbf_attr['n_chans'] * cbf_attr['n_accs'] * self.ratio
        self.interval = int(round(clocks * cbf_attr['scale_factor_timestamp'] /
                                  cbf_attr['adc_sample_rate']))
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
                 my_sensors, telstate, cbf_name, logger):
        self._sdisp_ips = {}
        self._center_freq = None

        # Lock used to synchronise access between the katcp device server
        # thread and this thread. It protects all attributes declared above
        # this point.
        self._lock = threading.Lock()

        # TODO: remove my_sensors and rather use the model to drive local sensor updates
        self.logger = logger
        self.cbf_spead_endpoints = opts.cbf_spead
        self.spectral_spead_endpoint = opts.l0_spectral_spead
        self.spectral_spead_rate = opts.l0_spectral_spead_rate
        self.continuum_spead_endpoint = opts.l0_continuum_spead
        self.continuum_spead_rate = opts.l0_continuum_spead_rate
        self.sd_spead_rate = opts.sd_spead_rate
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
        self.telstate = telstate
        self.telstate_name = opts.name
        self.cbf_name = cbf_name
        self.cbf_attr = {}

        self._my_sensors = my_sensors
        self.pkt_sensor = self._my_sensors['packets-captured']
        self.status_sensor = self._my_sensors['status']
        self.status_sensor.set_value("init")
        self.sd_flavour = spead2.Flavour(4, 64, 48, spead2.BUG_COMPAT_PYSPEAD_0_5_2)
        self.ig_sd = spead2.send.ItemGroup(flavour=self.sd_flavour)
        # Initialise processing blocks used
        self.command_queue = proc_template.context.create_command_queue()
        # Instantiation of the template delayed until data shape is known (TODO: can do it here)
        self.proc = None
        # Done with blocks

        self.logger.debug("Initialising SPEAD transports at %f" % time.time())
        self.logger.info("CBF SPEAD stream reception on {0}".format(
            [str(x) for x in self.cbf_spead_endpoints]))
        thread_pool = spead2.ThreadPool(4)
        self.rx = spead2.recv.Stream(thread_pool)
        for endpoint in self.cbf_spead_endpoints:
            self.rx.add_udp_reader(endpoint.port, bind_hostname=endpoint.host)
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

        threading.Thread.__init__(self)

    def enable_debug(self, debug):
        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

    def _send_sd_data(self, data):
        """Send a heap to all signal display servers.

        This method is thread-safe.
        """
        # The actual sending is blocking, so first make a copy so that we can
        # drop the lock quickly.
        with self._lock:
            sdisp_ips = self._sdisp_ips.values()
        for tx in sdisp_ips:
            tx.send_heap(data)

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

    def drop_sdisp_ip(self, ip):
        """Drop a signal display server from the list.

        Raises
        ------
        KeyError
            if `ip` is not currently in the list
        """
        with self._lock:
            self.logger.info("Removing ip %s from the signal display list." % (ip))
            stream = self._sdisp_ips[ip]
            del self._sdisp_ips[ip]
        stream.send_heap(self.ig_sd.get_end())

    def add_sdisp_ip(self, ip, port):
        """Add a new server to the signal display list.

        Parameters
        ----------
        ip : str
            Hostname or IP address
        port : int
            UDP port number

        Raises
        ------
        KeyError
            if `ip` is already in the list (even if on a different port)
        """
        with self._lock:
            if ip in self._sdisp_ips:
                raise ValueError('{0} is already in the active list of recipients'.format(ip))
            config = spead2.send.StreamConfig(max_packet_size=9172, rate=self.sd_spead_rate / 8)
            self.logger.info("Adding %s:%s to signal display list. Starting stream..." % (ip, port))
            self._sdisp_ips[ip] = spead2.send.UdpStream(spead2.ThreadPool(), ip, port, config)

    def set_center_freq(self, center_freq):
        """Change the center frequency reported to signal displays.

        This function is thread-safe.
        """
        with self._lock:
            self._center_freq = center_freq

    def _send_visibilities(self, tx, ig, vis, flags, ts_rel):
        # Create items on first use. This is simpler than figuring out the
        # correct shapes ahead of time.
        if 'correlator_data' not in ig:
            ig.add_item(id=None, name='correlator_data', description="Visibilities",
                        shape=vis.shape, dtype=vis.dtype)
            ig.add_item(id=None, name='flags', description="Flags for visibilities",
                        shape=flags.shape, dtype=flags.dtype)
            ig.add_item(id=None, name='timestamp', description="Seconds since sync time",
                        shape=(), dtype=None, format=[('f', 64)])
        ig['correlator_data'].value = vis
        ig['flags'].value = flags
        ig['timestamp'].value = ts_rel
        tx.send_heap(ig.get_heap())

    def _initialise_ig_sd(self):
        """Create a item group for signal displays."""
        sd_flavour = spead2.Flavour(4, 64, 48, spead2.BUG_COMPAT_PYSPEAD_0_5_2)
        inline_format = [('u', sd_flavour.heap_address_bits)]
        self.ig_sd = spead2.send.ItemGroup(flavour=self.sd_flavour)
        self.ig_sd.add_item(
            name=('sd_data'), id=(0x3501), description="Combined raw data from all x engines.",
            format=[('f', 32)], shape=(self.proc.buffer('sd_spec_vis').shape[0], None, 2))
        self.ig_sd.add_item(
            name=('sd_data_index'), id=(0x3509), description="Indices for transmitted sd_data.",
            format=[('u', 32)], shape=(None,))
        self.ig_sd.add_item(
            name=('sd_blmxdata'), id=0x3507, description="Reduced data for baseline matrix.",
            **_slot_shape(self.proc.buffer('sd_cont_vis'), np.float32))
        self.ig_sd.add_item(
            name=('sd_flags'), id=(0x3503), description="8bit packed flags for each data point.",
            format=[('u', 8)], shape=(self.proc.buffer('sd_spec_flags').shape[0], None))
        self.ig_sd.add_item(
            name=('sd_blmxflags'), id=(0x3508),
            description="Reduced data flags for baseline matrix.",
            **_slot_shape(self.proc.buffer('sd_cont_flags')))
        self.ig_sd.add_item(
            name=('sd_timeseries'), id=(0x3504), description="Computed timeseries.",
            **_slot_shape(self.proc.buffer('timeseries'), np.float32))
        n_perc_signals = 0
        perc_idx = 0
        while True:
            try:
                n_perc_signals += self.proc.buffer('percentile{0}'.format(perc_idx)).shape[0]
                perc_idx += 1
            except KeyError:
                break
        self.ig_sd.add_item(
            name=('sd_percspectrum'), id=(0x3505),
            description="Percentiles of spectrum data.",
            dtype=np.float32, shape=(self.proc.buffer('percentile0').shape[1], n_perc_signals))
        self.ig_sd.add_item(
            name=('sd_percspectrumflags'), id=(0x3506),
            description="Flags for percentiles of spectrum.",
            dtype=np.uint8, shape=(self.proc.buffer('percentile0').shape[1], n_perc_signals))
        self.ig_sd.add_item(
            name="center_freq", id=0x1011,
            description="The center frequency of the DBE in Hz, 64-bit IEEE floating-point number.",
            shape=(), dtype=None, format=[('f', 64)])
        self.ig_sd.add_item(
            name=('sd_timestamp'), id=0x3502,
            description='Timestamp of this sd frame in centiseconds since epoch (40 bit limitation).',
            shape=(), dtype=None, format=inline_format)
        bls_ordering = np.asarray(self.cbf_attr['bls_ordering'])
        self.ig_sd.add_item(
            name=('bls_ordering'), id=0x100C,
            description="Mapping of antenna/pol pairs to data output products.",
            shape=bls_ordering.shape, dtype=bls_ordering.dtype, value=bls_ordering)
        self.ig_sd.add_item(
            name="bandwidth", id=0x1013,
            description="The analogue bandwidth of the digitally processed signal in Hz.",
            shape=(), dtype=None, format=[('f', 64)], value=self.cbf_attr['bandwidth'])
        self.ig_sd.add_item(
            name="n_chans", id=0x1009,
            description="The total number of frequency channels present in any integration.",
            shape=(), dtype=None, format=inline_format, value=self.cbf_attr['n_chans'])

    def _initialise(self, ig_cbf):
        """Initialise variables on reception of the first usable dump."""
        cbf_baselines = len(self.cbf_attr['bls_ordering'])
        # Configure the masking and reordering of baselines
        orig_bls_ordering = self.cbf_attr['bls_ordering']
        permutation, self.cbf_attr['bls_ordering'] = \
            self.baseline_permutation(self.cbf_attr['bls_ordering'], self.antenna_mask)
        baselines = len(self.cbf_attr['bls_ordering'])
        channels = self.cbf_attr['n_chans']
        n_accs = self.cbf_attr['n_accs']
        self._set_telstate_entry('bls_ordering', self.cbf_attr['bls_ordering'])
        if baselines <= 0:
            raise ValueError('No baselines (bls_ordering={}, antenna_mask = {})'.format(
                orig_bls_ordering, self.antenna_mask))
        if channels <= 0:
            raise ValueError('No channels')

        # we need to create the raw datasets.
        data_item = ig_cbf['xeng_raw']
        new_shape = list(data_item.shape)
        new_shape[-2] = baselines
        self.logger.info("Creating cbf_data dataset with shape: {0}, dtype: {1}".format(
            str(new_shape), np.float32))

        # Configure time averaging
        self._output_avg = _TimeAverage(self.cbf_attr, self.output_int_time)
        self._output_avg.flush = self._flush_output
        self._set_telstate_entry('sdp_l0_int_time', self._output_avg.int_time, add_cbf_prefix=False)
        self.logger.info("Averaging {0} input dumps per output dump".format(self._output_avg.ratio))

        self._sd_avg = _TimeAverage(self.cbf_attr, self.sd_int_time)
        self._sd_avg.flush = self._flush_sd
        self.logger.info("Averaging {0} input dumps per signal display dump".format(
            self._sd_avg.ratio))

        # configure the signal processing blocks
        collection_products = sdispdata.set_bls(self.cbf_attr['bls_ordering'])[0]
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

        self.proc = self.proc_template.instantiate(
                self.command_queue, channels, (0, channels), cbf_baselines, baselines,
                self.cont_factor, self.sd_cont_factor, percentile_ranges,
                threshold_args={'n_sigma': 11.0})
        self.proc.set_scale(1.0 / n_accs)
        self.proc.ensure_all_bound()
        self.proc.buffer('permutation').set(
            self.command_queue, np.asarray(permutation, dtype=np.int16))
        self.proc.start_sum()
        self.proc.start_sd_sum()
        # Record information about the processing in telstate
        if self.telstate_name is not None and self.telstate is not None:
            descriptions = list(self.proc.descriptions())
            attribute_name = self.telstate_name.replace('.', '_') + '_process_log'
            self._set_telstate_entry(attribute_name, descriptions, add_cbf_prefix=False)

        # initialise the signal display metadata
        self._initialise_ig_sd()
        self._send_sd_data(self.ig_sd.get_start())

    def _flush_output(self, timestamps):
        """Finalise averaging of a group of input dumps and emit an output dump"""
        self.proc.end_sum()
        spec_flags = self.proc.buffer('spec_flags').get(self.command_queue)
        spec_vis = self.proc.buffer('spec_vis').get(self.command_queue)
        cont_flags = self.proc.buffer('cont_flags').get(self.command_queue)
        cont_vis = self.proc.buffer('cont_vis').get(self.command_queue)

        ts_rel = np.mean(timestamps) / self.cbf_attr['scale_factor_timestamp']
        # Shift to the centre of the dump
        ts_rel += 0.5 * self.cbf_attr['int_time']
        self._send_visibilities(self.tx_spectral, self.ig_spectral, spec_vis, spec_flags, ts_rel)
        self._send_visibilities(self.tx_continuum, self.ig_continuum, cont_vis, cont_flags, ts_rel)

        self.logger.info("Finished dump group with raw timestamps {0} (local: {1:.3f})".format(
            timestamps, time.time()))
        # Prepare for the next group
        self.proc.start_sum()

    def _flush_sd(self, timestamps):
        """Finalise averaging of a group of dumps for signal display, and send
        signal display data to the signal display server"""

        with self._lock:
            center_freq = self._center_freq

        # For now, both telstate and katcp can be used to set the mask and
        # custom signals, but telstate takes precedence.
        custom_signals_indices = None
        mask = None
        if self.telstate is not None:
            try:
                custom_signals_indices = np.array(
                    self.telstate['sdp_sdisp_custom_signals'],
                    dtype=np.uint32, copy=False)
            except KeyError:
                pass
            try:
                mask = np.array(
                    self.telstate['sdp_sdisp_timeseries_mask'],
                    dtype=np.float32, copy=False)
            except KeyError:
                pass

        if custom_signals_indices is None:
            custom_signals_indices = np.array([], dtype=np.uint32)
        if mask is None:
            mask = np.ones(self.channels, np.float32) / self.channels

        try:
            self.proc.buffer('timeseries_weights').set(self.command_queue, mask)
        except Exception:
            self.logger.warn('Failed to set timeseries_weights', exc_info=True)

        self.proc.end_sd_sum()
        ts_rel = np.mean(timestamps) / self.cbf_attr['scale_factor_timestamp']
        ts = self.cbf_attr['sync_time'] + ts_rel
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

        # populate new datastructure to supersede sd_data etc
        self.ig_sd['sd_timestamp'].value = int(ts * 100)
        if np.all(custom_signals_indices < spec_vis.shape[1]):
            self.ig_sd['sd_data'].value = \
                _split_array(spec_vis, np.float32)[:, custom_signals_indices, :]
            self.ig_sd['sd_data_index'].value = custom_signals_indices
            self.ig_sd['sd_flags'].value = spec_flags[:, custom_signals_indices]
        else:
            self.logger.warn('sdp_sdisp_custom_signals out of range, not updating (%s)',
                             custom_signals_indices)
        self.ig_sd['sd_blmxdata'].value = _split_array(cont_vis, np.float32)
        self.ig_sd['sd_blmxflags'].value = cont_flags
        self.ig_sd['sd_timeseries'].value = _split_array(timeseries, np.float32)
        self.ig_sd['sd_percspectrum'].value = np.vstack(percentiles).transpose()
        self.ig_sd['sd_percspectrumflags'].value = np.vstack(percentiles_flags).transpose()
        if center_freq is not None:
            self.ig_sd['center_freq'].value = center_freq

        self._send_sd_data(self.ig_sd.get_heap(descriptors='all', data='all'))
        self.logger.info("Finished SD group with raw timestamps {0} (local: {1:.3f})".format(
            timestamps, time.time()))
        # Prepare for the next group
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

    def _update_telstate(self, updated):
        """Updates the telescope state from new values in the item group."""
        for item_name, item in updated.iteritems():
            # bls_ordering is set later by _initialise, after permuting it.
            # The other items are data rather than metadata, and so do not
            # live in the telescope state.
            if item_name not in ['bls_ordering', 'timestamp', 'xeng_raw']:
                # store as an attribute unless item is in CBF_SPEAD_SENSORS (e.g. flags_xeng_raw)
                self._set_telstate_entry(item_name, item.value,
                                         attribute=(item_name not in CBF_SPEAD_SENSORS))

    def _update_cbf_attr(self, updated):
        """Updates the internal cbf_attr dictionary from new values in the item group."""
        for item_name, item in updated.iteritems():
            if (item_name not in ['timestamp', 'xeng_raw'] and
                    item_name not in CBF_SPEAD_SENSORS and
                    item.value is not None):
                if item_name not in self.cbf_attr:
                    self.cbf_attr[item_name] = item.value
                else:
                    self.logger.warning('Item %s is already set to %s, not setting to %s',
                                        item_name, self.cbf_attr[item_name], item.value)

    def run(self):
        """Thin wrapper than runs the real code and handles some cleanup."""

        try:
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
        except Exception:
            self.logger.error('CBFIngest thread threw an uncaught exception', exc_info=True)
            self._my_sensors['device-status'].set_value('fail', Sensor.ERROR)

    def _run(self):
        """Real implementation of `run`."""
        ig_cbf = spead2.ItemGroup()
        idx = 0
        self.status_sensor.set_value("idle")
        prev_ts = -1
        ts_wrap_offset = 0        # Value added to compensate for CBF timestamp wrapping
        ts_wrap_period = 2**48
        self._output_avg = None
        self._sd_avg = None

        for heap in self.rx:
            st = time.time()
            if idx == 0:
                self.status_sensor.set_value("capturing")

            # Update the telescope state and local cbf_attr cache
            updated = ig_cbf.update(heap)
            self._update_telstate(updated)
            self._update_cbf_attr(updated)
            if 'xeng_raw' not in updated:
                self.logger.warning(
                    "CBF Data received but either no metadata or xeng_raw group is present")
                continue
            if 'timestamp' not in updated:
                self.logger.warning("No timestamp received for current data frame - discarding")
                continue
            data_ts = ig_cbf['timestamp'].value + ts_wrap_offset
            data_item = ig_cbf['xeng_raw']
            if data_ts <= prev_ts:
                # This happens either because packets ended up out-of-order (in
                # which case we just discard the heap that arrived too late),
                # or because the CBF timestamp wrapped. Out-of-order should
                # jump backwards a tiny amount while wraps should jump back by
                # close to ts_wrap_period. If both happen at the same time
                # then things will go wrong.
                if data_ts < prev_ts - ts_wrap_period // 2:
                    ts_wrap_offset += ts_wrap_period
                    data_ts += ts_wrap_period
                    self.logger.warning('Data timestamps wrapped')
                else:
                    self.logger.warning(
                        "Data timestamps have gone backwards (%d <= %d), dropping heap",
                        data_ts, prev_ts)
                    continue
            prev_ts = data_ts
            # we have new data...

            # check to see if our CBF attributes are complete
            # i.e. make sure any attributes marked as critical are present
            if not CBF_CRITICAL_ATTRS.issubset(self.cbf_attr.keys()):
                self.logger.warning("CBF Component Model is not currently valid as critical attribute items are missing. Data will be discarded until these become available.")
                continue

            # Configure datasets and other items now that we have complete metadata
            if idx == 0:
                self._initialise(ig_cbf)

            self._output_avg.add_timestamp(data_ts)
            self._sd_avg.add_timestamp(data_ts)

            # Generate timestamps
            current_ts_rel = data_ts / self.cbf_attr['scale_factor_timestamp']
            current_ts = self.cbf_attr['sync_time'] + current_ts_rel
            self._my_sensors["last-dump-timestamp"].set_value(current_ts)

            # Perform data processing
            self.proc.buffer('vis_in').set(self.command_queue, data_item.value)
            self.proc()

            # Done with reading this frame
            idx += 1
            self.pkt_sensor.set_value(idx)
            tt = time.time() - st
            self.logger.info(
                "Captured CBF dump with timestamp %i (local: %.3f, process_time: %.2f, index: %i)",
                current_ts, tt+st, tt, idx)

        # Stop received.

        if self._output_avg is not None:  # Could be None if no heaps arrived
            self._output_avg.finish()
            self._sd_avg.finish()
        self.logger.info("CBF ingest complete at %f" % time.time())
        self.tx_spectral.send_heap(self.ig_spectral.get_end())
        self.tx_spectral = None
        self.tx_continuum.send_heap(self.ig_continuum.get_end())
        self.tx_continuum = None
        self._send_sd_data(self.ig_sd.get_end())
        self.ig_spectral = None
        self.ig_continuum = None
        if self.proc is not None:   # Could be None if no heaps arrived
            self.logger.debug("\nProcessing Blocks\n=================\n")
            for description in self.proc.descriptions():
                self.logger.debug("\t".join([str(x) for x in description]))
        self.status_sensor.set_value("complete")
