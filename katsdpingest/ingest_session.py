"""Class for ingesting data, processing it, and sending L0 visibilities onwards."""

from __future__ import division, print_function, absolute_import
import numpy as np
import spead2
import spead2.send
import spead2.recv
import spead2.send.trollius
import spead2.recv.trollius
import time
import fractions
import katsdpingest.sigproc as sp
from katsdpsigproc import resource
import katsdpsigproc.rfi.device as rfi
from katcp import Sensor
import katsdptelstate
import logging
import trollius
from trollius import From
from . import utils, receiver, sender


logger = logging.getLogger(__name__)


def get_collection_products(bls_ordering):
    """
    This is a clone (and cleanup) of :func:`katsdpdisp.data.set_bls`.
    """
    auto = []
    autohh = []
    autovv = []
    autohv = []
    cross = []
    crosshh = []
    crossvv = []
    crosshv = []
    for ibls, bls in enumerate(bls_ordering):
        if bls[0][:-1] == bls[1][:-1]:       # auto
            if bls[0][-1] == bls[1][-1]:     # autohh or autovv
                auto.append(ibls)
                if bls[0][-1] == 'h':
                    autohh.append(ibls)
                else:
                    autovv.append(ibls)
            else:                            # autohv or vh
                autohv.append(ibls)
        else:                                # cross
            if bls[0][-1] == bls[1][-1]:     # crosshh or crossvv
                cross.append(ibls)
                if bls[0][-1] == 'h':
                    crosshh.append(ibls)
                else:
                    crossvv.append(ibls)
            else:                            # crosshv or vh
                crosshv.append(ibls)
    return [auto, autohh, autovv, autohv, cross, crosshh, crossvv, crosshv]


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
                                  (2 * cbf_attr['bandwidth'])))
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


class ChannelRanges(object):
    """
    Tracks the various channel ranges involved in ingest. Each channel range
    is represented as a pair of integers (start and past-the-end), relative to
    the full band being correlated by CBF.

    Parameters
    ----------
    channels : int
        Number of input channels from CBF
    cont_factor : int
        Number of output channels averaged to produce a continuum output channel
    sd_cont_factor : int
        Number of channels averaged to produce a signal display continuum channel
    streams : int
        Number of input streams available for subscription (must divide `channels`)
    guard : int
        Minimum number of channels to keep on either side of the output for RFI
        flagging.
    output : :class:`katsdpingest.utils.Range`
        Output spectral channel range
    sd_output : :class:`katsdpingest.utils.Range`
        Signal display channel range

    Raises
    ------
    ValueError
        if `channels` is not a multiple of `cont_factor`, `sd_cont_factor` and
        `streams`
    ValueError
        if `output` is not aligned to `cont_factor`
    ValueError
        if `sd_output` is not aligned to `sd_cont_factor`
    ValueError
        if `output` or `sd_output` overflows the whole channel range

    Attributes
    ----------
    cont_factor : int
        Number of output channels averaged to produce a continuum output channel
    sd_cont_factor : int
        Number of channels averaged to produce a signal display continuum channel
    guard : int
        Minimum number of channels to keep on either side of the output for RFI
        flagging.
    cbf : :class:`katsdpingest.utils.Range`
        The full range of channels correlated by CBF (first element is always
        0)
    subscribed : :class:`katsdpingest.utils.Range`
        The set of channels we receive from the CBF from our multicast
        subscriptions (subset of `cbf`)
    input : :class:`katsdpingest.utils.Range`
        The set of channels transferred to the compute device (subset of
        `subscribed`).
    computed : :class:`katsdpingest.utils.Range`
        The set of channels for which we compute visibilities, flags and
        weights on the compute device (subset of `input`). This will be inset
        from `input` by at least `guard`, except where there is insufficient space
        in `cbf`.

        These channels are all read back to host memory. This
        range is guaranteed to be a multiple of both the signal display and
        output continuum factors.
    output : :class:`katsdpingest.utils.Range`
        The set of channels contained in the L0 spectral output (subset of
        `computed`). This range is guaranteed to be a multiple of the output
        continuum factor.
    sd_output : :class:`katsdpingest.utils.Range`
        The set of channels contained in the signal display output (subset of
        `computed`). This range is guaranteed to be a multiple of the signal
        display continuum factor.
    """

    def __init__(self, channels, cont_factor, sd_cont_factor, streams, guard,
                 output, sd_output):
        self.cont_factor = cont_factor
        self.sd_cont_factor = sd_cont_factor
        self.guard = guard
        self.cbf = utils.Range(0, channels)
        if not self.cbf.isaligned(streams):
            raise ValueError('channel count is not a multiple of the number of streams')
        if not self.cbf.isaligned(cont_factor):
            raise ValueError('channel count is not a multiple of the continuum factor')
        if not self.cbf.isaligned(sd_cont_factor):
            raise ValueError('channel count is not a multiple of the sd continuum factor')
        if not output.issubset(self.cbf):
            raise ValueError('output range overflows full channel range')
        if not sd_output.issubset(self.cbf):
            raise ValueError('sd output range overflows full channel range')
        if not output.isaligned(cont_factor):
            raise ValueError('output range is not aligned to continuum factor')
        if not sd_output.isaligned(sd_cont_factor):
            raise ValueError('sd output range is not aligned to continuum factor')
        self.output = output
        self.sd_output = sd_output
        # Compute least common multiple
        alignment = cont_factor * sd_cont_factor // fractions.gcd(cont_factor, sd_cont_factor)
        self.computed = output.union(sd_output).alignto(alignment)
        self.input = utils.Range(self.computed.start - guard, self.computed.stop + guard)
        self.input = self.input.intersection(self.cbf)
        stream_channels = channels // streams
        self.subscribed = self.input.alignto(stream_channels)
        assert self.subscribed.issubset(self.cbf)


class CBFIngest(object):
    """
    Ingest session.

    .. note:: The list of attributes is incomplete

    Attributes
    ----------
    vis_in_resource : :class:`katsdpsigproc.resource.Resource`
        Resource wrapping the `vis_in` device buffer
    channel_flags_resource : :class:`katsdpsigproc.resource.Resource`
        Resource wrapping the `channel_flags` device buffer
    timeseries_weights_resource : :class:`katsdpsigproc.resource.Resource`
        Resource wrapping the `timeseries_weights` device buffer
    output_resource : :class:`katsdpsigproc.resource.Resource`
        Resource wrapping the L0 output device buffers, namely `spec_vis`,
        `spec_flags`, `cont_vis` and `cont_flags`
    sd_output_resource : :class:`katsdpsigproc.resource.Resource`
        Resource wrapping the signal display output device buffers.
    proc_resource : :class:`katsdpsigproc.resource.Resource`
        The proc object, and the contents of all its buffers except for those
        covered by other resources above.
    rx : :class:`katsdpingest.receiver.Receiver`
        Receiver that combines data from the SPEAD streams into frames
    """
    # To avoid excessive autotuning, the following parameters are quantised up
    # to the next element of these lists when generating templates. These
    # lists are also used by ingest_autotune.py for pre-tuning standard
    # configurations.
    tune_antennas = [2, 4, 8, 16]
    tune_channels = [4096, 8192, 9216, 32768]

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
    def create_proc_template(cls, context, antennas, max_channels):
        """Create a processing template. This is a potentially slow operation,
        since it invokes autotuning.

        Parameters
        ----------
        context : :class:`katsdpsigproc.cuda.Context` or :class:`katsdpsigproc.opencl.Context`
            Context in which to compile device code
        antennas : int
            Number of antennas, *after* any masking
        max_channels : int
            Maximum number of incoming channels to support
        """
        # Quantise to reduce number of options to autotune
        max_antennas = cls._tune_next_antennas(antennas)
        max_channels = cls._tune_next(max_channels, cls.tune_channels)

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

    def __init__(self, opts, channel_ranges, proc_template,
                 my_sensors, telstate, cbf_name):
        self._sdisp_ips = {}
        self._center_freq = None
        self._run_future = None

        # TODO: remove my_sensors and rather use the model to drive local sensor updates
        self.spectral_spead_endpoints = opts.l0_spectral_spead
        self.continuum_spead_endpoints = opts.l0_continuum_spead
        self.sd_spead_rate = opts.sd_spead_rate
        self.output_int_time = opts.output_int_time
        self.sd_int_time = opts.sd_int_time
        self.channel_ranges = channel_ranges
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
        self.input_rate_sensor = self._my_sensors['input-rate']
        self.input_bytes = 0
        self.output_rate_sensor = self._my_sensors['output-rate']
        self.output_bytes = 0
        self.status_sensor = self._my_sensors['status']
        self.status_sensor.set_value("init")
        self.ig_sd = None
        # Initialise processing blocks used
        self.command_queue = proc_template.context.create_command_queue()
        # Instantiation of the template delayed until data shape is known (TODO: can do it here)
        self.proc = None
        self.proc_resource = None
        self.vis_in_resource = None
        self.channel_flags_resource = None
        self.timeseries_weights_resource = None
        self.output_resource = None
        self.sd_output_resource = None
        self.jobs = resource.JobQueue()
        # Done with blocks

        self.rx = receiver.Receiver(
            opts.cbf_spead, self.channel_ranges.subscribed,
            len(self.channel_ranges.cbf), telstate, cbf_name)
        self.cbf_attr = self.rx.cbf_attr
        # Instantiation of the output streams delayed until exact integration time is known
        self.tx_spectral = None
        self.tx_continuum = None

    def enable_debug(self, debug):
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

    def _send_sd_data(self, data):
        """Send a heap to all signal display servers, asynchronously.

        Parameters
        ----------
        data : :class:`spead2.send.Heap`
            Heap to send

        Returns
        -------
        future : `trollius.Future`
            A future that is completed when the heap has been sent to all receivers.
        """
        return trollius.gather(*(trollius.async(sender.async_send_heap(tx, data))
                                 for tx in self._sdisp_ips.itervalues()))

    @classmethod
    def baseline_permutation(cls, bls_ordering, antenna_mask=None, rotate=False):
        """Construct a permutation to place the baselines into the desired
        order for internal processing.

        Parameters
        ----------
        bls_ordering : list of pairs of strings
            Names of inputs in current ordering
        antenna_mask : set of strings, optional
            Antennas to retain in the permutation (without polarisation suffix)
        rotate : bool, optional
            Rotate received CBF baseline ordering up by 1 to account for CBF bustedness
            (20 May 2016 - may be fixed shortly after...)

        Returns
        -------
        permutation : list
            The permutation specifying the reordering. Element *i* indicates
            the position in the new order corresponding to element *i* of
            the original order, or -1 if the baseline was masked out.
        input_auto_baseline : list
            The post-permutation baseline index for each autocorrelation
        baseline_inputs : list
            Inputs (indexed as for `input_auto_baseline`) for each baseline
        new_ordering : ndarray
            Replacement ordering, in the same format as `bls_ordering`
        """
        if rotate:
            bls_ordering = bls_ordering[range(1, len(bls_ordering)) + [0]]

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

        def input_idx(input):
            try:
                return inputs.index(input)
            except ValueError:
                inputs.append(input)
                return len(inputs) - 1

        # Eliminate baselines not covered by antenna_mask
        filtered = [x for x in enumerate(bls_ordering) if keep(x[1])]
        # Sort what's left
        reordered = sorted(filtered, key=key)
        # reordered contains the mapping from new position to original
        # position, but we need the inverse.
        permutation = [-1] * len(bls_ordering)
        for i in range(len(reordered)):
            permutation[reordered[i][0]] = i
        # Can now discard the indices from reordered
        reordered = [x[1] for x in reordered]
        # Construct input_auto_baseline and baseline_inputs
        inputs = []
        baseline_inputs = [ [input_idx(x[0]), input_idx(x[1])] for x in reordered]
        input_auto_baseline = [-1] * len(inputs)
        for i, inputs in enumerate(baseline_inputs):
            if inputs[0] == inputs[1]:
                input_auto_baseline[inputs[0]] = i
        if -1 in input_auto_baseline:
            idx = input_auto_baseline.index(-1)
            raise ValueError('No auto-correlation baseline found for ' + inputs[idx])
        return permutation, input_auto_baseline, baseline_inputs, np.array(reordered)

    @trollius.coroutine
    def _stop_stream(self, stream, ig):
        """Send a stop packet to a stream. To ensure that it won't be lost
        on the sending side, the stream is first flushed, then the stop
        heap is sent and waited for."""
        if stream is not None:
            yield From(stream.async_flush())
            yield From(stream.async_send_heap(ig.get_end()))

    @trollius.coroutine
    def drop_sdisp_ip(self, ip):
        """Drop a signal display server from the list.

        Raises
        ------
        KeyError
            if `ip` is not currently in the list
        """
        logger.info("Removing ip %s from the signal display list." % (ip))
        stream = self._sdisp_ips[ip]
        del self._sdisp_ips[ip]
        if self.ig_sd is not None:
            yield From(self._stop_stream(stream, self.ig_sd))

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
        if ip in self._sdisp_ips:
            raise ValueError('{0} is already in the active list of recipients'.format(ip))
        config = spead2.send.StreamConfig(max_packet_size=8972, rate=self.sd_spead_rate / 8)
        logger.info("Adding %s:%s to signal display list. Starting stream..." % (ip, port))
        self._sdisp_ips[ip] = spead2.send.trollius.UdpStream(spead2.ThreadPool(), ip, port, config)

    def set_center_freq(self, center_freq):
        """Change the center frequency reported to signal displays."""
        self._center_freq = center_freq

    def _initialise_ig_sd(self):
        """Create a item group for signal displays."""
        sd_flavour = spead2.Flavour(4, 64, 48, spead2.BUG_COMPAT_PYSPEAD_0_5_2)
        inline_format = [('u', sd_flavour.heap_address_bits)]
        self.ig_sd = spead2.send.ItemGroup(flavour=sd_flavour)
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
        # Determine bandwidth of the signal display product
        sd_bandwidth = self.cbf_attr['bandwidth'] * len(self.channel_ranges.sd_output) / len(self.channel_ranges.cbf)
        self.ig_sd.add_item(
            name="bandwidth", id=0x1013,
            description="The analogue bandwidth of the digitally processed signal in Hz.",
            shape=(), dtype=None, format=[('f', 64)], value=sd_bandwidth)
        self.ig_sd.add_item(
            name="n_chans", id=0x1009,
            description="The total number of frequency channels present in any integration.",
            shape=(), dtype=None, format=inline_format, value=len(self.channel_ranges.sd_output))

    @trollius.coroutine
    def _initialise(self):
        """Initialise variables on reception of the first usable frame."""
        cbf_baselines = len(self.cbf_attr['bls_ordering'])
        # Configure the masking and reordering of baselines
        orig_bls_ordering = self.cbf_attr['bls_ordering']
        permutation, input_auto_baseline, baseline_inputs, self.cbf_attr['bls_ordering'] = \
            self.baseline_permutation(self.cbf_attr['bls_ordering'], self.antenna_mask)
        baselines = len(self.cbf_attr['bls_ordering'])
        n_accs = self.cbf_attr['n_accs']
        self._set_telstate_entry('bls_ordering', self.cbf_attr['bls_ordering'])
        if baselines <= 0:
            raise ValueError('No baselines (bls_ordering = {}, antenna_mask = {})'.format(
                orig_bls_ordering, self.antenna_mask))

        # Configure time averaging
        self._output_avg = _TimeAverage(self.cbf_attr, self.output_int_time)
        self._output_avg.flush = self._flush_output
        self._set_telstate_entry('sdp_l0_int_time', self._output_avg.int_time, add_cbf_prefix=False)
        logger.info("Averaging {0} input dumps per output dump".format(self._output_avg.ratio))

        self._sd_avg = _TimeAverage(self.cbf_attr, self.sd_int_time)
        self._sd_avg.flush = self._flush_sd
        logger.info("Averaging {0} input dumps per signal display dump".format(
                    self._sd_avg.ratio))

        # configure the signal processing blocks
        collection_products = get_collection_products(self.cbf_attr['bls_ordering'])
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
                self.command_queue, len(self.channel_ranges.input),
                self.channel_ranges.computed.relative_to(self.channel_ranges.input),
                len(input_auto_baseline),
                cbf_baselines, baselines,
                self.channel_ranges.cont_factor,
                self.channel_ranges.sd_cont_factor,
                percentile_ranges,
                threshold_args={'n_sigma': 11.0})
        self.proc.n_accs = n_accs
        self.proc.ensure_all_bound()
        self.proc.buffer('permutation').set(
            self.command_queue, np.asarray(permutation, dtype=np.int16))
        self.proc.buffer('input_auto_baseline').set(
            self.command_queue, np.asarray(input_auto_baseline, dtype=np.uint16))
        self.proc.buffer('baseline_inputs').set(
            self.command_queue, np.asarray(baseline_inputs, dtype=np.uint16))
        self.proc.start_sum()
        self.proc.start_sd_sum()
        # Set up resources
        self.proc_resource = resource.Resource(self.proc)
        self.vis_in_resource = resource.Resource(self.proc.buffer('vis_in'))
        self.channel_flags_resource = resource.Resource(self.proc.buffer('channel_flags'))
        self.timeseries_weights_resource = resource.Resource(self.proc.buffer('timeseries_weights'))
        self.output_resource = resource.Resource(None)
        self.sd_output_resource = resource.Resource(None)
        # Record information about the processing in telstate
        if self.telstate_name is not None and self.telstate is not None:
            descriptions = list(self.proc.descriptions())
            attribute_name = self.telstate_name.replace('.', '_') + '_process_log'
            self._set_telstate_entry(attribute_name, descriptions, add_cbf_prefix=False)

        # initialise the signal display metadata
        self._initialise_ig_sd()
        yield From(self._send_sd_data(self.ig_sd.get_start()))

        # Initialise the output streams
        l0_flavour = spead2.Flavour(4, 64, 48, spead2.BUG_COMPAT_PYSPEAD_0_5_2)
        thread_pool = spead2.ThreadPool(2)
        spectral_channels = self.channel_ranges.output.relative_to(self.channel_ranges.computed)
        continuum_channels = utils.Range(spectral_channels.start // self.channel_ranges.cont_factor,
                                         spectral_channels.stop // self.channel_ranges.cont_factor)
        self.tx_spectral = sender.VisSenderSet(
            thread_pool,
            self.spectral_spead_endpoints,
            l0_flavour,
            self._output_avg.int_time,
            spectral_channels,
            baselines)
        self.tx_continuum = sender.VisSenderSet(
            thread_pool,
            self.continuum_spead_endpoints,
            l0_flavour,
            self._output_avg.int_time,
            continuum_channels,
            baselines)
        yield From(self.tx_spectral.start())
        yield From(self.tx_continuum.start())

    def _flush_output(self, timestamps):
        """Finalise averaging of a group of input dumps and emit an output dump"""
        proc_a = self.proc_resource.acquire()
        output_a = self.output_resource.acquire()
        self.jobs.add(self._flush_output_job(proc_a, output_a, timestamps))

    @trollius.coroutine
    def _flush_output_job(self, proc_a, output_a, timestamps):
        with proc_a as proc, output_a:
            # Wait for resources
            events = yield From(proc_a.wait())
            events += yield From(output_a.wait())
            self.command_queue.enqueue_wait_for_events(events)

            # Compute
            proc.end_sum()
            proc_done = self.command_queue.enqueue_marker()
            proc_a.ready([proc_done])

            # Transfer (TODO: use pre-allocated pinned memory, with corresponding resource)
            spec = sender.Data()
            cont = sender.Data()
            for field in ['vis', 'flags', 'weights', 'weights_channel']:
                setattr(spec, field, proc.buffer('spec_' + field).get_async(self.command_queue))
                setattr(cont, field, proc.buffer('cont_' + field).get_async(self.command_queue))
            transfer_done = self.command_queue.enqueue_marker()
            # Prepare for the next group (which only touches the output
            # buffers).
            proc.start_sum()
            output_done = self.command_queue.enqueue_marker()
            self.command_queue.flush()
            output_a.ready([output_done])

            ts_rel = np.mean(timestamps) / self.cbf_attr['scale_factor_timestamp']
            # Shift to the centre of the dump
            ts_rel += 0.5 * self.cbf_attr['int_time']
            self.output_bytes += spec.nbytes + cont.nbytes
            yield From(resource.async_wait_for_events([transfer_done]))
            yield From(trollius.gather(
                self.tx_spectral.send(spec, ts_rel),
                self.tx_continuum.send(cont, ts_rel)))
            logger.info("Finished dump group with raw timestamps {0}".format(
                        timestamps))

    def _flush_sd(self, timestamps):
        """Finalise averaging of a group of dumps for signal display, and send
        signal display data to the signal display server"""
        custom_signals_indices = None
        full_mask = None
        if self.telstate is not None:
            try:
                custom_signals_indices = np.array(
                    self.telstate['sdp_sdisp_custom_signals'],
                    dtype=np.uint32, copy=False)
            except KeyError:
                pass
            try:
                full_mask = np.array(
                    self.telstate['sdp_sdisp_timeseries_mask'],
                    dtype=np.float32, copy=False)
            except KeyError:
                pass

        if custom_signals_indices is None:
            custom_signals_indices = np.array([], dtype=np.uint32)
        if full_mask is None:
            n_channels = len(self.channel_ranges.cbf)
            full_mask = np.ones(n_channels, np.float32) / n_channels
        # Create mask from full_mask. mask contains a weight for each channel
        # in computed, but those outside of sd_output are zero.
        mask = np.zeros(len(self.channel_ranges.computed), np.float32)
        used = self.channel_ranges.sd_output.relative_to(self.channel_ranges.computed)
        mask[used.asslice()] = full_mask[self.channel_ranges.sd_output.asslice()]

        proc_a = self.proc_resource.acquire()
        sd_output_a = self.sd_output_resource.acquire()
        timeseries_weights_a = self.timeseries_weights_resource.acquire()
        self.jobs.add(self._flush_sd_job(
                proc_a, sd_output_a, timeseries_weights_a,
                timestamps, custom_signals_indices, mask))

    @trollius.coroutine
    def _flush_sd_job(self, proc_a, sd_output_a, timeseries_weights_a,
                      timestamps, custom_signals_indices, mask):
        with proc_a as proc, sd_output_a, timeseries_weights_a as timeseries_weights:
            # Load timeseries weights
            events = yield From(timeseries_weights_a.wait())
            self.command_queue.enqueue_wait_for_events(events)
            try:
                timeseries_weights.set_async(self.command_queue, mask)
            except Exception:
                logger.warn('Failed to set timeseries_weights', exc_info=True)

            # Compute
            events = yield From(proc_a.wait())
            events += yield From(sd_output_a.wait())
            self.command_queue.enqueue_wait_for_events(events)
            proc.end_sd_sum()
            proc_done = self.command_queue.enqueue_marker()
            proc_a.ready([proc_done])
            timeseries_weights_a.ready([proc_done])

            # Transfer
            ts_rel = np.mean(timestamps) / self.cbf_attr['scale_factor_timestamp']
            ts = self.cbf_attr['sync_time'] + ts_rel
            cont_vis = proc.buffer('sd_cont_vis').get_async(self.command_queue)
            cont_flags = proc.buffer('sd_cont_flags').get_async(self.command_queue)
            spec_vis = proc.buffer('sd_spec_vis').get_async(self.command_queue)
            spec_flags = proc.buffer('sd_spec_flags').get_async(self.command_queue)
            timeseries = proc.buffer('timeseries').get_async(self.command_queue)
            percentiles = []
            percentiles_flags = []
            for i in range(len(proc.percentiles)):
                name = 'percentile{0}'.format(i)
                p = proc.buffer(name).get_async(self.command_queue)
                pflags = proc.buffer(name + '_flags').get(self.command_queue)
                percentiles.append(p)
                # Signal display server wants flags duplicated to broadcast with
                # the percentiles
                percentiles_flags.append(np.tile(pflags, (p.shape[0], 1)))
            transfer_done = self.command_queue.enqueue_marker()
            # Prepare for the next group
            proc.start_sd_sum()
            output_done = self.command_queue.enqueue_marker()
            self.command_queue.flush()
            sd_output_a.ready([output_done])

            # populate new datastructure to supersede sd_data etc
            yield From(resource.async_wait_for_events([transfer_done]))
            spec_channels = self.channel_ranges.sd_output.relative_to(self.channel_ranges.computed).asslice()
            cont_channels = utils.Range(spec_channels.start // self.channel_ranges.sd_cont_factor,
                                        spec_channels.stop // self.channel_ranges.sd_cont_factor).asslice()
            self.ig_sd['sd_timestamp'].value = int(ts * 100)
            if np.all(custom_signals_indices < spec_vis.shape[1]):
                self.ig_sd['sd_data'].value = \
                    _split_array(spec_vis, np.float32)[spec_channels, custom_signals_indices, :]
                self.ig_sd['sd_data_index'].value = custom_signals_indices
                self.ig_sd['sd_flags'].value = spec_flags[spec_channels, custom_signals_indices]
            else:
                logger.warn('sdp_sdisp_custom_signals out of range, not updating (%s)',
                            custom_signals_indices)
            self.ig_sd['sd_blmxdata'].value = _split_array(cont_vis[cont_channels, ...], np.float32)
            self.ig_sd['sd_blmxflags'].value = cont_flags[cont_channels, ...]
            self.ig_sd['sd_timeseries'].value = _split_array(timeseries, np.float32)
            self.ig_sd['sd_percspectrum'].value = np.vstack(percentiles).transpose()
            self.ig_sd['sd_percspectrumflags'].value = np.vstack(percentiles_flags).transpose()
            # Translate CBF-width center_freq to the center_freq for the
            # signal display product.
            cbf_center_freq = self._center_freq
            if cbf_center_freq is None:
                cbf_center_freq = self.cbf_attr.get('center_freq')
            if cbf_center_freq is not None:
                channel_width = self.cbf_attr['bandwidth'] / len(self.channel_ranges.cbf)
                cbf_mid = (self.channel_ranges.cbf.start + self.channel_ranges.cbf.stop) / 2
                sd_mid = (self.channel_ranges.sd_output.start + self.channel_ranges.sd_output.stop) / 2
                sd_center_freq = cbf_center_freq + (sd_mid - cbf_mid) * channel_width
                self.ig_sd['center_freq'].value = sd_center_freq

            yield From(self._send_sd_data(self.ig_sd.get_heap(descriptors='all', data='all')))
            logger.info("Finished SD group with raw timestamps {0}".format(
                        timestamps))

    @trollius.coroutine
    def _frame_job(self, proc_a, vis_in_a, channel_flags_a, frame):
        with proc_a as proc, \
             vis_in_a as vis_in_buffer, \
             channel_flags_a as channel_flags_buffer:
            # Load data
            events = (yield From(vis_in_a.wait())) + (yield From(channel_flags_a.wait()))
            self.command_queue.enqueue_wait_for_events(events)
            # First channel of the current item
            item_channel = self.channel_ranges.subscribed.start
            # We only receive frames with at least one populated item, so we
            # can always find out the number of channels per item
            channels_per_item = None
            for item in frame.items:
                if item is not None:
                    channels_per_item = item.shape[0]
                    break
            assert channels_per_item is not None
            if not frame.ready():
                # We want missing data to be zero-filled. katsdpsigproc doesn't
                # currently have a zero_region, and device bandwidth is so much
                # higher than PCIe transfer bandwidth that it doesn't really
                # cost much more to zero-fill the entire buffer.
                vis_in_buffer.zero(self.command_queue)
            channel_flags = channel_flags_buffer.empty_like()
            channel_flags.fill(0)
            data_lost_flag = 1 << sp.IngestTemplate.flag_names.index('data_lost')
            for item in frame.items:
                item_range = utils.Range(item_channel, item_channel + channels_per_item)
                item_channel = item_range.stop
                channel_flags_range = item_range.intersection(self.channel_ranges.computed)
                if item is None:
                    channel_flags[channel_flags_range.asslice()] = data_lost_flag
                use_range = item_range.intersection(self.channel_ranges.input)
                if not use_range or item is None:
                    continue
                dest_range = use_range.relative_to(self.channel_ranges.input)
                src_range = use_range.relative_to(item_range)
                vis_in_buffer.set_region(
                    self.command_queue, item,
                    dest_range.asslice(), src_range.asslice(), blocking=False)
            channel_flags_buffer.set_async(self.command_queue, channel_flags)
            transfer_done = self.command_queue.enqueue_marker()
            self.command_queue.flush()

            # Perform data processing
            events = yield From(proc_a.wait())
            self.command_queue.enqueue_wait_for_events(events)
            proc()
            done_event = self.command_queue.enqueue_marker()
            vis_in_a.ready([done_event])
            channel_flags_a.ready([done_event])
            proc_a.ready([done_event])

            # Keep vis_in live until the transfer is complete
            yield From(resource.async_wait_for_events([transfer_done]))

    def _set_telstate_entry(self, name, value, add_cbf_prefix=True, attribute=True):
        utils.set_telstate_entry(self.telstate, name, value, self.cbf_name if add_cbf_prefix else None, attribute)

    def start(self):
        assert self._run_future is None
        self._run_future = trollius.async(self.run())

    @trollius.coroutine
    def stop(self):
        if self._run_future:
            self.rx.stop()
            yield From(self._run_future)
            self._run_future = None

    @trollius.coroutine
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
                    yield From(self._run())
                finally:
                    # These have references to self, causing circular references
                    self._output_avg = None
                    self._sd_avg = None
                    # Drop last references to all the objects
                    self.proc = None
        except Exception:
            logger.error('CBFIngest session threw an uncaught exception', exc_info=True)
            self._my_sensors['device-status'].set_value('fail', Sensor.ERROR)

    @trollius.coroutine
    def _run(self):
        """Real implementation of `run`."""
        idx = 0
        rate_timer = 0
        self.status_sensor.set_value("idle")
        self._output_avg = None
        self._sd_avg = None
        while True:
            try:
                frame = yield From(self.rx.get())
            except spead2.Stopped:
                break

            st = time.time()
            # Configure datasets and other items now that we have complete metadata
            if idx == 0:
                self.status_sensor.set_value("capturing")
                rate_timer = time.time()
                yield From(self._initialise())

            # Generate timestamps
            current_ts_rel = frame.timestamp / self.cbf_attr['scale_factor_timestamp']
            current_ts = self.cbf_attr['sync_time'] + current_ts_rel
            self._my_sensors["last-dump-timestamp"].set_value(current_ts)

            self._output_avg.add_timestamp(frame.timestamp)
            self._sd_avg.add_timestamp(frame.timestamp)

            proc_a = self.proc_resource.acquire()
            vis_in_a = self.vis_in_resource.acquire()
            channel_flags_a = self.channel_flags_resource.acquire()
            # Limit backlog by waiting for previous job to get as far as
            # enqueuing its work before trying to carry on.
            yield From(proc_a.wait())
            self.input_bytes += frame.nbytes
            self.jobs.add(self._frame_job(proc_a, vis_in_a, channel_flags_a, frame))

            # Done with reading this frame
            idx += 1
            self.pkt_sensor.set_value(idx)
            if idx % 10 == 0:
                self.input_rate_sensor.set_value(int(self.input_bytes / (time.time() - rate_timer)))
                self.output_rate_sensor.set_value(int(self.output_bytes / (time.time() - rate_timer)))
                self.input_bytes = 0
                self.output_bytes = 0
                rate_timer = time.time()
            tt = time.time() - st
            logger.info(
                "Captured CBF frame with timestamp %i (process_time: %.2f, index: %i)",
                current_ts, tt, idx)
            # Clear completed processing, so that any related exceptions are
            # thrown as soon as possible.
            self.jobs.clean()

        # Stop received.
        yield From(self.rx.join())
        if self.proc_resource is not None:    # Could be None if no heaps arrived
            acq = self.proc_resource.acquire()
            with acq:
                yield From(acq.wait_events())
                if self._output_avg is not None:
                    self._output_avg.finish()
                self._sd_avg.finish()
                acq.ready()
        yield From(self.jobs.finish())
        logger.info("CBF ingest complete")
        if self.tx_spectral is not None:
            yield From(self.tx_spectral.stop())
            self.tx_spectral = None
        if self.tx_continuum is not None:
            yield From(self.tx_continuum.stop())
            self.tx_continuum = None
        if self.ig_sd is not None:
            for tx in self._sdisp_ips.itervalues():
                yield From(self._stop_stream(tx, self.ig_sd))
        if self.proc is not None:   # Could be None if no heaps arrived
            logger.debug("\nProcessing Blocks\n=================\n")
            for description in self.proc.descriptions():
                logger.debug("\t".join([str(x) for x in description]))
        self.status_sensor.set_value("complete")
