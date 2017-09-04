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
import katsdpservices
import logging
import trollius
from trollius import From, Return
from . import utils, receiver, sender


logger = logging.getLogger(__name__)
# Attributes that are required for data to be correctly ingested
CBF_CRITICAL_ATTRS = frozenset([
    'adc_sample_rate', 'n_chans', 'n_chans_per_substream', 'n_accs', 'bls_ordering',
    'bandwidth', 'center_freq',
    'sync_time', 'int_time', 'scale_factor_timestamp', 'ticks_between_spectra'])


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
        # Integration time in timestamp ticks
        self._sub_interval = cbf_attr['ticks_between_spectra'] * cbf_attr['n_accs']
        self.interval = self.ratio * self._sub_interval
        self._start_ts = None
        self._ts = []

    def add_timestamp(self, timestamp):
        """Record that a dump with a given timestamp has arrived and is about to
        be processed. This may call :func:`flush`."""

        if self._start_ts is None:
            # First time: special case. We need to choose _start_ts in a way
            # that will have the same phase as other ingest processes, even if
            # they see a different timestamp first. We do this by choosing the
            # largest _start_ts such that:
            # 1. _start_ts <= timestamp.
            # 2. _start_ts % interval < sub_interval
            si = self._sub_interval
            self._start_ts = timestamp - (timestamp % self.interval) // si * si

        if timestamp >= self._start_ts + self.interval:
            self.flush(self._ts)
            skip_groups = (timestamp - self._start_ts) // self.interval
            self._ts = []
            self._start_ts += skip_groups * self.interval
        self._ts.append(timestamp)

    def flush(self, timestamps):
        raise NotImplementedError

    def finish(self, flush=True):
        """Flush if not empty and `flush` is true, and reset to initial state"""
        if self._ts and flush:
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


class ChannelRanges(object):
    """
    Tracks the various channel ranges involved in ingest. Each channel range
    is represented as a pair of integers (start and past-the-end), relative to
    the full band being correlated by CBF.

    Parameters
    ----------
    servers : int
        Number of servers jointly producing the output
    server_id : int
        Zero-based index of this server amongst `servers`
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
    all_output : :class:`katsdpingest.utils.Range`
        Output spectral channel range
    all_sd_output : :class:`katsdpingest.utils.Range`
        Signal display channel range

    Raises
    ------
    ValueError
        if `channels` is not a multiple of `cont_factor`, `sd_cont_factor` and
        `streams`
    ValueError
        if the length of `all_output` or `all_sd_output` is not a multiple of
        `servers`.
    ValueError
        if the per-server `output` is not aligned to `cont_factor`
    ValueError
        if the per-server `sd_output` is not aligned to `sd_cont_factor`
    ValueError
        if `all_output` or `all_sd_output` overflows the whole channel range

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
        The set of channels placed in the L0 spectral output by this server
        (subset of `computed`). This range is guaranteed to be a multiple of
        the output continuum factor.
    all_output : :class:`katsdpingest.utils.Range`
        L0 spectral output channels produced by all the servers. This range is
        guaranteed to be a multiple of the output continuum factor.
    sd_output : :class:`katsdpingest.utils.Range`
        The set of channels placed in the signal display output by this server
        (subset of `computed`). This range is guaranteed to be a multiple of
        the signal display continuum factor.
    all_sd_output : :class:`katsdpingest.utils.Range`
        Signal display output channels produced by all the servers. This range
        is guaranteed to be a multiple of the signal display continuum
        factor.
    """

    def __init__(self, servers, server_id,
                 channels, cont_factor, sd_cont_factor, streams, guard,
                 all_output, all_sd_output):
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
        if not all_output.issubset(self.cbf):
            raise ValueError('output range overflows full channel range')
        if not all_sd_output.issubset(self.cbf):
            raise ValueError('sd output range overflows full channel range')
        self.all_output = all_output
        self.all_sd_output = all_sd_output
        self.output = all_output.split(servers, server_id)
        self.sd_output = all_sd_output.split(servers, server_id)
        if not self.output.isaligned(cont_factor):
            raise ValueError('output range is not aligned to continuum factor')
        if not self.sd_output.isaligned(sd_cont_factor):
            raise ValueError('sd output range is not aligned to continuum factor')
        # Compute least common multiple
        alignment = cont_factor * sd_cont_factor // fractions.gcd(cont_factor, sd_cont_factor)
        self.computed = self.output.union(self.sd_output).alignto(alignment)
        self.input = utils.Range(self.computed.start - guard, self.computed.stop + guard)
        self.input = self.input.intersection(self.cbf)
        stream_channels = channels // streams
        self.subscribed = self.input.alignto(stream_channels)
        assert self.subscribed.issubset(self.cbf)


class _ResourceSet(object):
    """Collection of device buffers with host staging areas.

    A resource set groups together
    - a :class:`katsdpsigproc.resource.Resource` containing a dict mapping
      names to :class:`katsdpsigproc.accel.DeviceArray`s.
    - N :class:`katsdpsigproc.resource.Resource`s containing dicts mapping
      names to :class:`katsdpsigproc.accel.HostArray`s.
    This provides N-buffered staging of data into or out of the device buffers.

    Parameters
    ----------
    proc : :class:`katsdpsigproc.accel.Operation`
        Processing operation from which the buffers will be found by name.
    names : list of str
        Names of buffers to find in `proc`
    N : int
        Number of host arrays to create for each buffer
    """
    def __init__(self, proc, names, N):
        if N <= 0:
            raise ValueError('_ResourceSet needs at least one buffer')
        buffers = {name: proc.buffer(name) for name in names}
        self._device = resource.Resource(buffers)
        self._host = []
        for i in range(N):
            host = {name: buffer.empty_like() for name, buffer in buffers.items()}
            self._host.append(resource.Resource(host))
        self._next = 0     # Next host buffer to return from acquire

    def acquire(self):
        """Acquire device resource and next available host resource."""
        ret = (self._device.acquire(), self._host[self._next].acquire())
        self._next += 1
        if self._next == len(self._host):
            self._next = 0
        return ret


def get_cbf_attr(telstate, cbf_name):
    """Load the configuration of the CBF stream from a telescope state.

    Parameters
    ----------
    telstate : :class:`katsdptelstate.TelescopeState`
        Telescope state from which the CBF stream metadata is retrieved.
    cbf_name : str
        Common prefix on telstate keys
    """
    cbf_attr = {}
    for attr in CBF_CRITICAL_ATTRS:
        telstate_name = '{}_{}'.format(cbf_name, attr)
        try:
            cbf_attr[attr] = telstate[telstate_name]
            logger.info('Setting cbf_attr %s to %r', attr, cbf_attr[attr])
        except KeyError:
            # Telstate's KeyError does not have a useful description
            raise KeyError('Telstate key {} not found'.format(telstate_name))
    logger.info('All metadata received from telstate')
    return cbf_attr


def _convert_center_freq(old_channels, old_center_freq, old_bandwidth, new_channels):
    """Compute the center frequency of a channel range, given the center
    frequency and bandwidth of a different set of channels.

    The implementation is careful to avoid introducing rounding errors where the answer
    is an exactly representable value.
    """
    old_mid = (old_channels.start + old_channels.stop) / 2
    new_mid = (new_channels.start + new_channels.stop) / 2
    return old_center_freq + (new_mid - old_mid) * old_bandwidth / len(old_channels)


class BaselineOrdering(object):
    """Encapsulates lookup tables related to baseline ordering.

    Parameters
    ----------
    cbf_bls_ordering : list of pairs of str
        Input ordering.
    antenna_mask : iterable of str, optional
        If given, only those antennas in this set will be retained in the output.

    Attributes
    ----------
    permutation : list
        The permutation specifying the reordering. Element *i* indicates
        the position in the new order corresponding to element *i* of
        the original order, or -1 if the baseline was masked out.
    input_auto_baseline : list
        The post-permutation baseline index for each autocorrelation
    baseline_inputs : list
        Inputs (indexed as for `input_auto_baseline`) for each baseline
    sdp_bls_ordering : ndarray
        Replacement ordering, in the same format as `cbf_bls_ordering`
    percentile_ranges : list of pairs of int
        Intervals of the new ordering which get grouped together for percentile
        calculations.
    """

    def __init__(self, cbf_bls_ordering, antenna_mask=None):
        def keep(baseline):
            ant1 = baseline[0][:-1]
            ant2 = baseline[1][:-1]
            # Eliminate baselines that have a lower-numbered antenna as the
            # second input as these are almost certainly duplicates. This is only
            # a problem in single pol mode and could be removed in the future.
            if ant2 < ant1:
                return False
            if antenna_mask:
                return ant1 in antenna_mask and ant2 in antenna_mask
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

        def get_collection_products(bls_ordering):
            """This is a clone (and cleanup) of :func:`katsdpdisp.data.set_bls`."""
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

        if antenna_mask is not None:
            antenna_mask = set(antenna_mask)
        # Eliminate baselines not covered by antenna_mask
        filtered = [x for x in enumerate(cbf_bls_ordering) if keep(x[1])]
        # Sort what's left
        reordered = sorted(filtered, key=key)
        # reordered contains the mapping from new position to original
        # position, but we need the inverse.
        self.permutation = [-1] * len(cbf_bls_ordering)
        for i in range(len(reordered)):
            self.permutation[reordered[i][0]] = i
        # Can now discard the indices from reordered
        reordered = [x[1] for x in reordered]
        # Construct input_auto_baseline and baseline_inputs
        inputs = []
        self.baseline_inputs = [[input_idx(x[0]), input_idx(x[1])] for x in reordered]
        self.input_auto_baseline = [-1] * len(inputs)
        for i, inputs in enumerate(self.baseline_inputs):
            if inputs[0] == inputs[1]:
                self.input_auto_baseline[inputs[0]] = i
        if -1 in self.input_auto_baseline:
            idx = self.input_auto_baseline.index(-1)
            raise ValueError('No auto-correlation baseline found for ' + inputs[idx])
        self.sdp_bls_ordering = np.array(reordered)

        # Collect percentile ranges
        collection_products = get_collection_products(self.sdp_bls_ordering)
        self.percentile_ranges = []
        for p in collection_products:
            if p:
                start = p[0]
                end = p[-1] + 1
                if not np.array_equal(np.arange(start, end), p):
                    raise ValueError("percentile baselines are not contiguous: {}".format(p))
                self.percentile_ranges.append((start, end))
            else:
                self.percentile_ranges.append((0, 0))


class CBFIngest(object):
    """
    Ingest session.

    .. note:: The list of attributes is incomplete

    Attributes
    ----------
    input_resource : :class:`_ResourceSet`
        Resource wrapping the device buffers that contain inputs
    output_resource : :class:`_ResourceSet`
        Wrapper of the L0 output device buffers, namely `spec_vis`,
        `spec_flags`, `spec_weights` and `spec_weights_channel`, and the
        same for `cont_`.
    sd_input_resource : :class:`_ResourceSet`
        Wrapper of the buffers that serve as inputs to signal display
        processing (currently `timeseries_weights`).
    sd_output_resource : :class:`_ResourceSet`
        Wrapper of the signal display output device buffers.
    proc_resource : :class:`katsdpsigproc.resource.Resource`
        The proc object, and the contents of all its buffers except for those
        covered by other resources above.
    rx : :class:`katsdpingest.receiver.Receiver`
        Receiver that combines data from the SPEAD streams into frames
    cbf_attr : dict
        Input stream attributes, as returned by :func:`get_cbf_attr`
    bls_ordering : :class:`BaselineOrdering`
        Baseline ordering and related tables
    """
    # To avoid excessive autotuning, the following parameters are quantised up
    # to the next element of these lists when generating templates. These
    # lists are also used by ingest_autotune.py for pre-tuning standard
    # configurations.
    tune_channels = [4096, 8192, 9216, 32768]
    tune_percentile_sizes = set()
    for ants in [2, 4, 8, 16, 32]:
        tune_percentile_sizes.update({ants, 2 * ants, ants * (ants - 1) // 2, ants * (ants - 1)})
    tune_percentile_sizes = list(sorted(tune_percentile_sizes))

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
    def create_proc_template(cls, context, percentile_sizes, max_channels, excise):
        """Create a processing template. This is a potentially slow operation,
        since it invokes autotuning.

        Callers won't normally call this, since it is done by the constructor.
        It is provided for the pre-tuner.

        Parameters
        ----------
        context : :class:`katsdpsigproc.cuda.Context` or :class:`katsdpsigproc.opencl.Context`
            Context in which to compile device code
        percentile_sizes : list of int
            Sizes of baseline groupings, *after* any masking
        max_channels : int
            Maximum number of incoming channels to support
        excise : bool
            Excise flagged data by downweighting it massively.
        """
        # Quantise to reduce number of options to autotune
        max_percentile_sizes = [cls._tune_next(s, cls.tune_percentile_sizes)
                                for s in percentile_sizes]
        max_percentile_sizes = list(sorted(set(max_percentile_sizes)))
        max_channels = cls._tune_next(max_channels, cls.tune_channels)

        flag_value = 1 << sp.IngestTemplate.flag_names.index('ingest_rfi')
        background_template = rfi.BackgroundMedianFilterDeviceTemplate(
                context, width=13, use_flags=True)
        noise_est_template = rfi.NoiseEstMADTDeviceTemplate(context, max_channels=max_channels)
        threshold_template = rfi.ThresholdSimpleDeviceTemplate(
                context, transposed=True, flag_value=flag_value)
        flagger_template = rfi.FlaggerDeviceTemplate(
                background_template, noise_est_template, threshold_template)
        return sp.IngestTemplate(context, flagger_template, percentile_sizes=max_percentile_sizes,
                                 excise=excise)

    def _zero_counters(self):
        self.output_bytes = 0
        self.output_bytes_sensor.set_value(0)
        self.output_heaps = 0
        self.output_heaps_sensor.set_value(0)
        self.output_dumps = 0
        self.output_dumps_sensor.set_value(0)

    def _init_sensors(self, my_sensors):
        self._my_sensors = my_sensors
        self.output_bytes_sensor = self._my_sensors['output-bytes-total']
        self.output_heaps_sensor = self._my_sensors['output-heaps-total']
        self.output_dumps_sensor = self._my_sensors['output-dumps-total']
        self.status_sensor = self._my_sensors['status']
        self.status_sensor.set_value("init")
        self._zero_counters()

    def _init_baselines(self, antenna_mask):
        # Configure the masking and reordering of baselines
        self.bls_ordering = BaselineOrdering(self.cbf_attr['bls_ordering'], antenna_mask)
        if not len(self.bls_ordering.sdp_bls_ordering):
            raise ValueError('No baselines (bls_ordering = {}, antenna_mask = {})'.format(
                self.cbf_attr['bls_ordering'], antenna_mask))

    def _init_time_averaging(self, output_int_time, sd_int_time):
        self._output_avg = _TimeAverage(self.cbf_attr, output_int_time)
        self._output_avg.flush = self._flush_output
        logger.info("Averaging {0} input dumps per output dump".format(self._output_avg.ratio))

        self._sd_avg = _TimeAverage(self.cbf_attr, sd_int_time)
        self._sd_avg.flush = self._flush_sd
        logger.info("Averaging {0} input dumps per signal display dump".format(
                    self._sd_avg.ratio))

    def _init_proc(self, context, excise):
        percentile_sizes = list(set(r[1] - r[0] for r in self.bls_ordering.percentile_ranges))
        proc_template = self.create_proc_template(
            context, percentile_sizes, len(self.channel_ranges.input), excise)
        self.command_queue = proc_template.context.create_command_queue()
        self.proc = proc_template.instantiate(
                self.command_queue, len(self.channel_ranges.input),
                self.channel_ranges.computed.relative_to(self.channel_ranges.input),
                len(self.bls_ordering.input_auto_baseline),
                len(self.cbf_attr['bls_ordering']),
                len(self.bls_ordering.sdp_bls_ordering),
                self.channel_ranges.cont_factor,
                self.channel_ranges.sd_cont_factor,
                self.bls_ordering.percentile_ranges,
                threshold_args={'n_sigma': 11.0})
        self.proc.n_accs = self.cbf_attr['n_accs']
        self.proc.ensure_all_bound()
        self.proc.buffer('permutation').set(
            self.command_queue, np.asarray(self.bls_ordering.permutation, dtype=np.int16))
        self.proc.buffer('input_auto_baseline').set(
            self.command_queue, np.asarray(self.bls_ordering.input_auto_baseline, dtype=np.uint16))
        self.proc.buffer('baseline_inputs').set(
            self.command_queue, np.asarray(self.bls_ordering.baseline_inputs, dtype=np.uint16))
        self.proc.start_sum()
        self.proc.start_sd_sum()
        logger.debug("\nProcessing Blocks\n=================\n")
        for description in self.proc.descriptions():
            logger.debug("\t".join([str(x) for x in description]))

    def _init_resources(self):
        self.jobs = resource.JobQueue()
        self.proc_resource = resource.Resource(self.proc)
        self.input_resource = _ResourceSet(
            self.proc, ['vis_in', 'channel_flags', 'baseline_flags'], 2)
        self.output_resource = _ResourceSet(
            self.proc, [prefix + suffix
                        for prefix in ['spec_', 'cont_']
                        for suffix in ['vis', 'flags', 'weights', 'weights_channel']], 2)
        self.sd_input_resource = _ResourceSet(self.proc, ['timeseries_weights'], 2)
        sd_output_names = ['sd_cont_vis', 'sd_cont_flags', 'sd_spec_vis', 'sd_spec_flags',
                           'timeseries', 'timeseriesabs']
        for i in range(len(self.proc.percentiles)):
            base_name = 'percentile{}'.format(i)
            sd_output_names.append(base_name)
            sd_output_names.append(base_name + '_flags')
        self.sd_output_resource = _ResourceSet(self.proc, sd_output_names, 2)

    def _init_tx_one(self, args, tx_type, cont_factor):
        l0_flavour = spead2.Flavour(4, 64, 48, spead2.BUG_COMPAT_PYSPEAD_0_5_2)
        all_output = self.channel_ranges.all_output
        # Compute channel ranges relative to those computed
        spectral_channels = self.channel_ranges.output.relative_to(self.channel_ranges.computed)
        channels = utils.Range(spectral_channels.start // cont_factor,
                               spectral_channels.stop // cont_factor)
        baselines = len(self.bls_ordering.sdp_bls_ordering)
        endpoints = getattr(args, 'l0_{}_spead'.format(tx_type))
        if len(endpoints) % args.servers:
            raise ValueError('Number of endpoints ({}) not divisible by number of servers ({})'
                .format(len(endpoints), args.servers))
        endpoint_lo = (args.server_id - 1) * len(endpoints) // args.servers
        endpoint_hi = args.server_id * len(endpoints) // args.servers
        endpoints = endpoints[endpoint_lo:endpoint_hi]
        tx = sender.VisSenderSet(
            spead2.ThreadPool(),
            endpoints,
            katsdpservices.get_interface_address(getattr(args, 'l0_{}_interface'.format(tx_type))),
            l0_flavour,
            self._output_avg.int_time,
            channels,
            (self.channel_ranges.output.start - all_output.start) // cont_factor,
            len(all_output) // cont_factor,
            baselines)

        # Put attributes into telstate
        prefix = getattr(args, 'l0_{}_name'.format(tx_type))
        self._set_telstate_entry('n_chans', len(all_output) // cont_factor, prefix)
        self._set_telstate_entry('n_chans_per_substream', tx.sub_channels, prefix)
        self._set_telstate_entry('n_bls', baselines, prefix)
        self._set_telstate_entry('bls_ordering', self.bls_ordering.sdp_bls_ordering, prefix)
        self._set_telstate_entry('sync_time', self.cbf_attr['sync_time'], prefix)
        bandwidth = self.cbf_attr['bandwidth'] * len(all_output) / len(self.channel_ranges.cbf)
        center_freq = _convert_center_freq(self.channel_ranges.cbf,
                                           self.cbf_attr['center_freq'],
                                           self.cbf_attr['bandwidth'],
                                           all_output)
        self._set_telstate_entry('bandwidth', bandwidth, prefix)
        self._set_telstate_entry('center_freq', center_freq, prefix)
        self._set_telstate_entry('channel_range', all_output.astuple(), prefix)
        self._set_telstate_entry('int_time', self._output_avg.int_time, prefix)
        return tx

    def _init_tx(self, args):
        self.tx_spectral = self._init_tx_one(args, 'spectral', 1)
        self.tx_continuum = self._init_tx_one(args, 'continuum', self.channel_ranges.cont_factor)

    def _init_ig_sd(self):
        """Create a item group for signal displays."""
        sd_flavour = spead2.Flavour(4, 64, 48, spead2.BUG_COMPAT_PYSPEAD_0_5_2)
        inline_format = [('u', sd_flavour.heap_address_bits)]
        n_spec_channels = len(self.channel_ranges.sd_output)
        n_cont_channels = n_spec_channels // self.channel_ranges.sd_cont_factor
        n_baselines = len(self.bls_ordering.sdp_bls_ordering)
        self.ig_sd = spead2.send.ItemGroup(flavour=sd_flavour)
        self.ig_sd.add_item(
            name=('sd_data'), id=(0x3501), description="Combined raw data from all x engines.",
            format=[('f', 32)], shape=(n_spec_channels, None, 2))
        self.ig_sd.add_item(
            name=('sd_data_index'), id=(0x3509), description="Indices for transmitted sd_data.",
            format=[('u', 32)], shape=(None,))
        self.ig_sd.add_item(
            name=('sd_blmxdata'), id=0x3507, description="Reduced data for baseline matrix.",
            shape=(n_cont_channels, n_baselines, 2),
            dtype=np.float32)
        self.ig_sd.add_item(
            name=('sd_flags'), id=(0x3503), description="8bit packed flags for each data point.",
            format=[('u', 8)], shape=(n_spec_channels, None))
        self.ig_sd.add_item(
            name=('sd_blmxflags'), id=(0x3508),
            description="Reduced data flags for baseline matrix.",
            shape=(n_cont_channels, n_baselines), dtype=np.uint8)
        self.ig_sd.add_item(
            name=('sd_timeseries'), id=(0x3504), description="Computed timeseries.",
            shape=(n_baselines, 2), dtype=np.float32)
        self.ig_sd.add_item(
            name=('sd_timeseriesabs'), id=(0x3510), description="Computed timeseries magnitude.",
            shape=(n_baselines, ), dtype=np.float32)
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
            dtype=np.float32, shape=(n_spec_channels, n_perc_signals))
        self.ig_sd.add_item(
            name=('sd_percspectrumflags'), id=(0x3506),
            description="Flags for percentiles of spectrum.",
            dtype=np.uint8, shape=(n_spec_channels, n_perc_signals))
        self.ig_sd.add_item(
            name=('sd_timestamp'), id=0x3502,
            description='Timestamp of this sd frame in centiseconds since epoch (40 bit limitation).',
            shape=(), dtype=None, format=inline_format)
        bls_ordering = np.asarray(self.bls_ordering.sdp_bls_ordering)
        self.ig_sd.add_item(
            name=('bls_ordering'), id=0x100C,
            description="Mapping of antenna/pol pairs to data output products.",
            shape=bls_ordering.shape, dtype=bls_ordering.dtype, value=bls_ordering)
        # Determine bandwidth and centre frequency of the signal display product
        sd_bandwidth = (self.cbf_attr['bandwidth'] * len(self.channel_ranges.all_sd_output)
                        / len(self.channel_ranges.cbf))
        sd_center_freq = _convert_center_freq(
            self.channel_ranges.cbf, self.cbf_attr['center_freq'], self.cbf_attr['bandwidth'],
            self.channel_ranges.all_sd_output)
        self.ig_sd.add_item(
            name="bandwidth", id=0x1013,
            description="The analogue bandwidth of the signal display product in Hz.",
            shape=(), dtype=None, format=[('f', 64)], value=sd_bandwidth)
        self.ig_sd.add_item(
            name="center_freq", id=0x1011,
            description="The center frequency of the signal display product in Hz.",
            shape=(), dtype=None, format=[('f', 64)], value=sd_center_freq)
        self.ig_sd.add_item(
            name="n_chans", id=0x1009,
            description="The total number of frequency channels in the signal display product.",
            shape=(), dtype=None, format=inline_format,
            value=len(self.channel_ranges.all_sd_output))
        self.ig_sd.add_item(
            name="frequency", id=0x4103,
            description="The frequency channel of the data in this heap.",
            shape=(), dtype=None, format=inline_format,
            value=self.channel_ranges.sd_output.start - self.channel_ranges.all_sd_output.start)

    def __init__(self, args, cbf_attr, channel_ranges, context, my_sensors, telstate):
        self._sdisp_ips = {}
        self._run_future = None
        # Set by stop to abort prior to creating the receiver
        self._stopped = True

        self.rx_spead_endpoints = args.cbf_spead
        self.rx_spead_ifaddr = katsdpservices.get_interface_address(args.cbf_interface)
        self.rx_spead_ibv = args.cbf_ibv
        self.sd_spead_rate = args.sd_spead_rate
        self.channel_ranges = channel_ranges
        self.telstate = telstate
        self.cbf_attr = cbf_attr

        self._init_sensors(my_sensors)
        self._init_baselines(args.antenna_mask)
        self._init_time_averaging(args.output_int_time, args.sd_int_time)
        self._init_proc(context, args.excise)
        self._init_resources()

        # Instantiation of input streams is delayed until the asynchronous task
        # is running, to avoid receiving data we're not yet ready for.
        self.rx = None
        self._init_tx(args)  # Note: must be run after _init_time_averaging
        self._init_ig_sd()

        # Record information about the processing in telstate
        if args.name is not None:
            descriptions = list(self.proc.descriptions())
            attribute_name = args.name.replace('.', '_') + '_process_log'
            self._set_telstate_entry(attribute_name, descriptions)

    def enable_debug(self, debug):
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.NOTSET)

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

    @trollius.coroutine
    def _stop_stream(self, stream, ig):
        """Send a stop packet to a stream. To ensure that it won't be lost
        on the sending side, the stream is first flushed, then the stop
        heap is sent and waited for."""
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
        logger.info("Removing ip %s from the signal display list.", ip)
        stream = self._sdisp_ips[ip]
        del self._sdisp_ips[ip]
        if self.capturing:
            yield From(self._stop_stream(stream, self.ig_sd))

    def add_sdisp_ip(self, endpoint):
        """Add a new server to the signal display list.

        Parameters
        ----------
        endpoint : :class:`katsdptelstate.endpoint.Endpoint`
            Destination host and port

        Raises
        ------
        KeyError
            if `ip` is already in the list (even if on a different port)
        """
        if endpoint.host in self._sdisp_ips:
            raise ValueError('{0} is already in the active list of recipients'.format(endpoint))
        config = spead2.send.StreamConfig(max_packet_size=8972, rate=self.sd_spead_rate / 8)
        logger.info("Adding %s to signal display list. Starting stream...", endpoint)
        stream = spead2.send.trollius.UdpStream(
            spead2.ThreadPool(), endpoint.host, endpoint.port, config)
        # Ensure that signal display streams that form the full band between
        # them always have unique heap cnts. The first output channel is used
        # as a unique key.
        stream.set_cnt_sequence(self.channel_ranges.sd_output.start,
                                len(self.channel_ranges.cbf))
        self._sdisp_ips[endpoint.host] = stream

    def _flush_output(self, timestamps):
        """Finalise averaging of a group of input dumps and emit an output dump"""
        proc_a = self.proc_resource.acquire()
        output_a, host_output_a = self.output_resource.acquire()
        self.jobs.add(self._flush_output_job(proc_a, output_a, host_output_a, timestamps))

    @trollius.coroutine
    def _flush_output_job(self, proc_a, output_a, host_output_a, timestamps):
        with proc_a as proc, output_a as output, host_output_a as host_output:
            # Wait for resources
            events = yield From(proc_a.wait())
            events += yield From(output_a.wait())
            self.command_queue.enqueue_wait_for_events(events)

            # Compute
            proc.end_sum()
            self.command_queue.flush()

            # Wait for the host output buffers to be available
            events = yield From(host_output_a.wait())
            self.command_queue.enqueue_wait_for_events(events)

            # Transfer
            data = {'spec': sender.Data(), 'cont': sender.Data()}
            for prefix in ['spec', 'cont']:
                for field in ['vis', 'flags', 'weights', 'weights_channel']:
                    name = prefix + '_' + field
                    setattr(data[prefix], field, host_output[name])
                    output[name].get_async(self.command_queue, host_output[name])
            transfer_done = self.command_queue.enqueue_marker()
            # Prepare for the next group.
            proc.start_sum()
            proc_done = self.command_queue.enqueue_marker()
            self.command_queue.flush()
            proc_a.ready([proc_done])
            output_a.ready([proc_done])

            ts_rel = np.mean(timestamps) / self.cbf_attr['scale_factor_timestamp']
            # Shift to the centre of the dump
            ts_rel += 0.5 * self.cbf_attr['int_time']
            yield From(resource.async_wait_for_events([transfer_done]))
            spec = data['spec']
            cont = data['cont']
            yield From(trollius.gather(
                self.tx_spectral.send(spec, ts_rel),
                self.tx_continuum.send(cont, ts_rel)))
            self.output_bytes += spec.nbytes + cont.nbytes
            self.output_bytes_sensor.set_value(self.output_bytes)
            self.output_heaps += self.tx_spectral.size + self.tx_continuum.size
            self.output_heaps_sensor.set_value(self.output_heaps)
            self.output_dumps += 2
            self.output_dumps_sensor.set_value(self.output_dumps)
            host_output_a.ready()
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
            n_channels = len(self.channel_ranges.all_sd_output)
            full_mask = np.ones(n_channels, np.float32) / n_channels
        # Create mask from full_mask. mask contains a weight for each channel
        # in computed, but those outside of sd_output are zero.
        mask = np.zeros(len(self.channel_ranges.computed), np.float32)
        used = self.channel_ranges.sd_output.relative_to(self.channel_ranges.computed)
        sd_rel = self.channel_ranges.sd_output.relative_to(self.channel_ranges.all_sd_output)
        mask[used.asslice()] = full_mask[sd_rel.asslice()]

        proc_a = self.proc_resource.acquire()
        sd_input_a, host_sd_input_a = self.sd_input_resource.acquire()
        sd_output_a, host_sd_output_a = self.sd_output_resource.acquire()
        self.jobs.add(self._flush_sd_job(
                proc_a, sd_input_a, host_sd_input_a,
                sd_output_a, host_sd_output_a,
                timestamps, custom_signals_indices, mask))

    @trollius.coroutine
    def _flush_sd_job(self, proc_a, sd_input_a, host_sd_input_a,
                      sd_output_a, host_sd_output_a,
                      timestamps, custom_signals_indices, mask):
        with proc_a as proc, \
                sd_input_a as sd_input_buffers, \
                host_sd_input_a as host_sd_input, \
                sd_output_a as sd_output_buffers, \
                host_sd_output_a as host_sd_output:
            spec_channels = self.channel_ranges.sd_output.relative_to(self.channel_ranges.computed).asslice()
            cont_channels = utils.Range(spec_channels.start // self.channel_ranges.sd_cont_factor,
                                        spec_channels.stop // self.channel_ranges.sd_cont_factor).asslice()
            # Copy inputs to HostArrays
            yield From(host_sd_input_a.wait_events())
            host_sd_input['timeseries_weights'][:] = mask

            # Transfer to device
            events = yield From(sd_input_a.wait())
            self.command_queue.enqueue_wait_for_events(events)
            for name in host_sd_input:
                sd_input_buffers[name].set_async(self.command_queue, host_sd_input[name])
            transfer_in_done = self.command_queue.enqueue_marker()
            host_sd_input_a.ready([transfer_in_done])
            self.command_queue.flush()

            # Compute
            events = yield From(proc_a.wait())
            events += yield From(sd_output_a.wait())
            self.command_queue.enqueue_wait_for_events(events)
            proc.end_sd_sum()
            sd_input_a.ready([self.command_queue.enqueue_marker()])
            self.command_queue.flush()

            # Transfer back to host
            events = yield From(host_sd_output_a.wait())
            self.command_queue.enqueue_wait_for_events(events)
            for name in host_sd_output:
                sd_output_buffers[name].get_async(self.command_queue, host_sd_output[name])
            transfer_out_done = self.command_queue.enqueue_marker()

            # Prepare for the next group
            proc.start_sd_sum()
            proc_done = self.command_queue.enqueue_marker()
            proc_a.ready([proc_done])
            sd_output_a.ready([proc_done])
            self.command_queue.flush()

            # Mangle and transmit the retrieved values
            yield From(resource.async_wait_for_events([transfer_out_done]))
            ts_rel = np.mean(timestamps) / self.cbf_attr['scale_factor_timestamp']
            ts = self.cbf_attr['sync_time'] + ts_rel
            cont_vis = host_sd_output['sd_cont_vis']
            cont_flags = host_sd_output['sd_cont_flags']
            spec_vis = host_sd_output['sd_spec_vis']
            spec_flags = host_sd_output['sd_spec_flags']
            timeseries = host_sd_output['timeseries']
            timeseriesabs = host_sd_output['timeseriesabs']
            percentiles = []
            percentiles_flags = []
            for i in range(len(proc.percentiles)):
                name = 'percentile{0}'.format(i)
                p = host_sd_output[name]
                pflags = host_sd_output[name + '_flags']
                p = p[..., spec_channels]
                pflags = pflags[..., spec_channels]
                percentiles.append(p)
                # Signal display server wants flags duplicated to broadcast with
                # the percentiles
                percentiles_flags.append(np.tile(pflags, (p.shape[0], 1)))

            # populate new datastructure to supersede sd_data etc
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
            self.ig_sd['sd_timeseriesabs'].value = timeseriesabs
            self.ig_sd['sd_percspectrum'].value = np.vstack(percentiles).transpose()
            self.ig_sd['sd_percspectrumflags'].value = np.vstack(percentiles_flags).transpose()

            yield From(self._send_sd_data(self.ig_sd.get_heap(descriptors='all', data='all')))
            host_sd_output_a.ready()
            logger.info("Finished SD group with raw timestamps {0}".format(
                        timestamps))

    def _set_baseline_flags(self, baseline_flags, timestamp):
        """Query telstate for per-baseline flags to set.

        The last value set in prior to the end of the dump is used.
        """
        end_time = timestamp + self.cbf_attr['int_time']
        if self.telstate is None:
            baseline_flags.fill(0)
            return
        cache = {}
        baselines = self.bls_ordering.sdp_bls_ordering
        cam_flag = 1 << sp.IngestTemplate.flag_names.index('cam')
        for i, baseline in enumerate(baselines):
            # [:-1] indexing strips off h/v pol
            a = baseline[0][:-1]
            b = baseline[1][:-1]
            for antenna in (a, b):
                if antenna not in cache:
                    sensor_name = '{}_data_suspect'.format(antenna)
                    try:
                        values = self.telstate.get_range(sensor_name, et=end_time)
                        value = values[-1][0]     # Last entry, value element of pair
                    except (KeyError, IndexError):
                        value = False
                    cache[antenna] = value
            baseline_flags[i] = cam_flag if cache[a] or cache[b] else 0

    @trollius.coroutine
    def _frame_job(self, proc_a, input_a, host_input_a, frame):
        with proc_a as proc, \
             input_a as input_buffers, \
             host_input_a as host_input:
            vis_in_buffer = input_buffers['vis_in']
            channel_flags_buffer = input_buffers['channel_flags']
            baseline_flags_buffer = input_buffers['baseline_flags']
            vis_in = host_input['vis_in']
            channel_flags = host_input['channel_flags']
            baseline_flags = host_input['baseline_flags']
            # Load data
            yield From(host_input_a.wait_events())
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
            try:
                channel_mask = self.telstate['cbf_channel_mask']
                channel_mask = channel_mask[self.channel_ranges.input.asslice()]
                static_flag = 1 << sp.IngestTemplate.flag_names.index('static')
                channel_flags[:] = channel_mask * np.uint8(static_flag)
            except KeyError:
                channel_flags.fill(0)
            except (ValueError, TypeError):
                # Could happen if the telstate key has the wrong shape or type
                logger.warn('Error loading channel flags from telstate', exc_info=True)
                channel_flags.fill(0)
            self._set_baseline_flags(baseline_flags, frame.timestamp)
            data_lost_flag = 1 << sp.IngestTemplate.flag_names.index('data_lost')
            for item in frame.items:
                item_range = utils.Range(item_channel, item_channel + channels_per_item)
                item_channel = item_range.stop
                use_range = item_range.intersection(self.channel_ranges.input)
                if not use_range:
                    continue
                dest_range = use_range.relative_to(self.channel_ranges.input)
                src_range = use_range.relative_to(item_range)
                if item is None:
                    channel_flags[dest_range.asslice()] = data_lost_flag
                    vis_in[dest_range.asslice()] = 0
                else:
                    vis_in[dest_range.asslice()] = item[src_range.asslice()]

            # Transfer data to the device
            events = yield From(input_a.wait())
            self.command_queue.enqueue_wait_for_events(events)
            for name in input_buffers:
                input_buffers[name].set_async(self.command_queue, host_input[name])
            transfer_done = self.command_queue.enqueue_marker()
            self.command_queue.flush()
            host_input_a.ready([transfer_done])

            # Perform data processing
            events = yield From(proc_a.wait())
            self.command_queue.enqueue_wait_for_events(events)
            proc()
            done_event = self.command_queue.enqueue_marker()
            input_a.ready([done_event])
            proc_a.ready([done_event])

    def _set_telstate_entry(self, name, value, prefix=None, attribute=True):
        utils.set_telstate_entry(self.telstate, name, value, prefix, attribute)

    @property
    def capturing(self):
        return self._run_future is not None

    def start(self):
        assert self._run_future is None
        assert self.rx is None
        assert self._stopped
        self._stopped = False
        self._run_future = trollius.async(self.run())

    @trollius.coroutine
    def stop(self):
        """Shut down the session. It is safe to make reentrant calls: each
        will wait for the shutdown to complete. It is safe to call
        :meth:`start` again once one of the callers returns.

        Returns
        -------
        stopped : bool
            True if we were running. If multiple callers call concurrently
            when running, exactly one of them will return true.
        """
        ret = False
        if self.capturing:
            self._stopped = True
            if self.rx is not None:
                logger.info('Stopping receiver...')
                self.rx.stop()
            logger.info('Waiting for run to stop...')
            future = self._run_future
            yield From(future)
            logger.info('Run stopped')
            # If multiple callers arrive here, we want only the first to
            # return True and clean up. We also need to protect against a prior
            # task having cleaned up and immediately started a new capture
            # session. In this case _run_future will be non-None (and hence
            # capturing will be True), but the object identity of _run_future
            # will no longer match future.
            if self._run_future is future:
                ret = True
                self._run_future = None
                self.rx = None
        raise Return(ret)

    @trollius.coroutine
    def run(self):
        """Thin wrapper than runs the real code and handles exceptions."""
        try:
            yield From(self._run())
        except Exception:
            logger.error('CBFIngest session threw an uncaught exception', exc_info=True)
            self._my_sensors['device-status'].set_value('fail', Sensor.ERROR)

    def close(self):
        # PyCUDA has a bug/limitation regarding cleanup
        # (http://wiki.tiker.net/PyCuda/FrequentlyAskedQuestions) that tends
        # to cause device objects and `HostArray`s to leak. To prevent it,
        # we need to ensure that references are dropped (and hence objects
        # are deleted) with the context being current.
        with self.command_queue.context:
            # These have references to self, causing circular references
            del self._output_avg
            del self._sd_avg
            # Drop last references to all the objects
            del self.proc
            del self.proc_resource
            del self.input_resource
            del self.output_resource
            del self.sd_input_resource
            del self.sd_output_resource

    @trollius.coroutine
    def _get_data(self):
        """Receive data. This is called after the metadata has been retrieved."""
        idx = 0
        self.status_sensor.set_value("wait-data")
        while True:
            try:
                frame = yield From(self.rx.get())
            except spead2.Stopped:
                logger.info('Detected receiver stopped')
                yield From(self.rx.join())
                return

            st = time.time()
            # Configure datasets and other items now that we have complete metadata
            if idx == 0:
                self.status_sensor.set_value("capturing")

            # Generate timestamps
            current_ts_rel = frame.timestamp / self.cbf_attr['scale_factor_timestamp']
            current_ts = self.cbf_attr['sync_time'] + current_ts_rel
            self._my_sensors["last-dump-timestamp"].set_value(current_ts)

            self._output_avg.add_timestamp(frame.timestamp)
            self._sd_avg.add_timestamp(frame.timestamp)

            proc_a = self.proc_resource.acquire()
            input_a, host_input_a = self.input_resource.acquire()
            # Limit backlog by waiting for previous job to get as far as
            # start to transfer its data before trying to carry on.
            yield From(host_input_a.wait())
            self.jobs.add(self._frame_job(proc_a, input_a, host_input_a, frame))

            # Done with reading this frame
            idx += 1
            tt = time.time() - st
            logger.info(
                "Captured CBF frame with timestamp %i (process_time: %.2f, index: %i)",
                current_ts, tt, idx)
            # Clear completed processing, so that any related exceptions are
            # thrown as soon as possible.
            self.jobs.clean()

    @trollius.coroutine
    def _run(self):
        """Real implementation of `run`."""
        # Ensure we have clean state. Some of this is unnecessary in normal
        # use, but important if the previous session crashed.
        self._zero_counters()
        self._output_avg.finish(flush=False)
        self._sd_avg.finish(flush=False)
        self._init_ig_sd()
        # Send start-of-stream packets.
        yield From(self._send_sd_data(self.ig_sd.get_start()))
        yield From(self.tx_spectral.start())
        yield From(self.tx_continuum.start())
        # Initialise the input stream
        self.rx = receiver.Receiver(
            self.rx_spead_endpoints, self.rx_spead_ifaddr, self.rx_spead_ibv,
            self.channel_ranges.subscribed,
            len(self.channel_ranges.cbf), self._my_sensors, self.cbf_attr)
        # If stop() was called before we create self.rx, it won't have been able
        # to call self.rx.stop(), but it will have set _stopped.
        if self._stopped:
            self.rx.stop()

        # The main loop
        yield From(self._get_data())

        logger.info('Joined with receiver. Flushing final groups...')
        self._output_avg.finish()
        self._sd_avg.finish()
        logger.info('Waiting for jobs to complete...')
        yield From(self.jobs.finish())
        logger.info('Jobs complete')
        logger.info('Stopping tx_spectral stream...')
        yield From(self.tx_spectral.stop())
        logger.info('Stopping tx_continuum stream...')
        yield From(self.tx_continuum.stop())
        for tx in self._sdisp_ips.itervalues():
            logger.info('Stopping signal display stream...')
            yield From(self._stop_stream(tx, self.ig_sd))
        logger.info("CBF ingest complete")
        self.status_sensor.set_value("complete")
