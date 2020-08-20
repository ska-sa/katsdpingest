"""Class for ingesting data, processing it, and sending L0 visibilities onwards."""

import time
import math
import logging
import asyncio
import enum
import argparse
import textwrap
import functools
from collections import deque
import gc
from typing import (
    Mapping, Dict, List, Tuple, Deque, Set, Iterable, Callable, Awaitable,
    Optional, TypeVar, Any
)    # noqa: F401

import numpy as np
import astropy.units as u
import aiohttp

import spead2
import spead2.send
import spead2.recv
import spead2.send.asyncio
import spead2.recv.asyncio

import katsdpsigproc.accel
from katsdpsigproc import resource
import katsdpsigproc.rfi.device as rfi

import katsdpmodels.fetch.aiohttp
import katsdpmodels.band_mask
import katsdpmodels.rfi_mask

import katsdpservices

from katdal import SpectralWindow
from katdal.flags import CAM, DATA_LOST, INGEST_RFI, STATIC
import katpoint
import katsdptelstate.aio
from katsdptelstate.endpoint import endpoints_to_str, Endpoint

from . import utils, receiver, sender, sigproc
from .utils import Sensor


logger = logging.getLogger(__name__)
# Attributes that are required for data to be correctly ingested
CBF_CRITICAL_ATTRS = frozenset([
    'n_chans', 'n_chans_per_substream', 'n_accs', 'bls_ordering',
    'bandwidth', 'center_freq', 'input_labels',
    'sync_time', 'int_time', 'scale_factor_timestamp', 'ticks_between_spectra'])
_M = TypeVar('_M', bound='katsdpmodels.models.Model')


class Status(enum.Enum):
    """State of the ingest state machine"""
    INIT = 0
    WAIT_DATA = 1
    CAPTURING = 2
    COMPLETE = 3


class DeviceStatus(enum.Enum):
    """Standard katcp device status"""
    OK = 0
    DEGRADED = 1
    FAIL = 2


class _TimeAverage:
    """Manages a collection of dumps that are averaged together at a specific
    cadence.

    This object never sees dump contents directly, only dump indices. When an
    index is added that is not part of the current group, `flush` is called.

    Parameters
    ----------
    ratio : int
        Number of input dumps per output dump
    flush
        Asynchronous callback which is called once a set of dumps is ready to
        be averaged. It is passed the output dump index.

    Attributes
    ----------
    ratio : int
        number of input dumps per output dump
    _start_idx : int
        Index of first dump in the current group, or ``None`` if no dumps have been seen.
        There is at least one dump in the current group if and only if this is
        not ``None``.
    """
    def __init__(self, ratio: int, flush: Callable[[int], Awaitable[None]]) -> None:
        self.ratio = ratio
        self.flush = flush
        self._start_idx: Optional[int] = None

    def _warp_start(self, idx: int) -> None:
        """Set :attr:`start_idx` to the smallest multiple of ratio that is <= idx."""
        self._start_idx = idx // self.ratio * self.ratio

    async def add_index(self, idx: int) -> None:
        """Record that a dump with a given index has arrived and is about to be processed.

        This may call the `flush` callback given to the constructor.
        """

        if self._start_idx is None:
            self._warp_start(idx)
        elif idx >= self._start_idx + self.ratio:
            await self.flush(self._start_idx // self.ratio)
            self._warp_start(idx)

    async def finish(self, flush: bool = True) -> None:
        """Flush if not empty and `flush` is true, and reset to initial state"""
        if self._start_idx is not None and flush:
            await self.flush(self._start_idx // self.ratio)
        self._start_idx = None


def _mid_timestamp_rel(time_average: _TimeAverage, recv: receiver.Receiver, idx: int) -> float:
    """Convert an output dump index into a timestamp.

    Parameters
    ----------
    time_average : :class:`_TimeAverage`
        Averager, used to get the ratio of input to output dumps
    recv : :class:`.receiver.Receiver`
        Receiver, used to get CBF attributes, start timestamp and interval
    idx : int
        Output dump index

    Returns
    -------
    ts_rel : float
        Time in seconds from CBF sync time to the middle of the dump
    """
    ts_raw = (idx + 0.5) * time_average.ratio * recv.interval + recv.timestamp_base
    return ts_raw / recv.cbf_attr['scale_factor_timestamp']


def _split_array(x: np.ndarray, dtype) -> np.ndarray:
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


def _fix_descriptions(desc: Any) -> Any:
    """Massage operation descriptions to be suitable for telstate storage.

    It recursively:
    - Replaces Python types and numpy dtypes with their string form.
    - Sorts dictionaries so that the output is reproducible across runs (this
      will be reproducible only from Python 3.6 onwards).
    """
    if isinstance(desc, list):
        return [_fix_descriptions(item) for item in desc]
    elif isinstance(desc, tuple):
        return tuple([_fix_descriptions(item) for item in desc])
    elif isinstance(desc, set):
        return {_fix_descriptions(item) for item in desc}
    elif isinstance(desc, dict):
        return dict(sorted(_fix_descriptions(item) for item in desc.items()))
    elif isinstance(desc, np.dtype):
        return np.lib.format.dtype_to_descr(desc)
    elif isinstance(desc, type):
        if issubclass(desc, np.generic):
            # It's something like np.uint8, which is probably intended to represent
            # a numpy dtype.
            return np.lib.format.dtype_to_descr(np.dtype(desc))
        else:
            return str(desc)
    else:
        return desc


class ChannelRanges:
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

    def __init__(self, servers: int, server_id: int,
                 channels: int, cont_factor: int, sd_cont_factor: int, streams: int, guard: int,
                 all_output: utils.Range, all_sd_output: utils.Range) -> None:
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
        alignment = cont_factor * sd_cont_factor // math.gcd(cont_factor, sd_cont_factor)
        self.computed = self.output.union(self.sd_output).alignto(alignment)
        self.input = utils.Range(self.computed.start - guard, self.computed.stop + guard)
        self.input = self.input.intersection(self.cbf)
        stream_channels = channels // streams
        self.subscribed = self.input.alignto(stream_channels)
        assert self.subscribed.issubset(self.cbf)


class _ResourceSet:
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
    def __init__(self, proc: katsdpsigproc.accel.Operation,
                 names: List[str], N: int) -> None:
        if N <= 0:
            raise ValueError('_ResourceSet needs at least one buffer')
        buffers = {name: proc.buffer(name) for name in names}
        self._device = resource.Resource(buffers)
        self._host: List[resource.Resource] = []
        for i in range(N):
            host = {name: buffer.empty_like() for name, buffer in buffers.items()}
            self._host.append(resource.Resource(host))
        self._next = 0     # Next host buffer to return from acquire

    def acquire(self) -> Tuple[resource.ResourceAllocation, resource.ResourceAllocation]:
        """Acquire device resource and next available host resource."""
        ret = (self._device.acquire(), self._host[self._next].acquire())
        self._next += 1
        if self._next == len(self._host):
            self._next = 0
        return ret


class BaselineOrdering:
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
    sdp_bls_ordering : ndarray
        Replacement ordering, in the same format as `cbf_bls_ordering`
    percentile_ranges : list of pairs of int
        Intervals of the new ordering which get grouped together for percentile
        calculations.
    """

    def __init__(self,
                 cbf_bls_ordering: List[Tuple[str, str]],
                 antenna_mask: Iterable[str] = None) -> None:
        def keep(baseline: Tuple[str, str]) -> bool:
            ant1 = baseline[0][:-1]
            ant2 = baseline[1][:-1]
            # Eliminate baselines that have a lower-numbered antenna as the
            # second input as these are almost certainly duplicates. This is only
            # a problem in single pol mode and could be removed in the future.
            if ant2 < ant1:
                return False
            if antenna_mask_set:
                return ant1 in antenna_mask_set and ant2 in antenna_mask_set
            else:
                return True

        def key(item: Tuple[int, Tuple[str, str]]) -> Tuple[bool, bool, str, str]:
            input1, input2 = item[1]
            pol1 = input1[-1]
            pol2 = input2[-1]
            return (input1[:-1] != input2[:-1], pol1 != pol2, pol1, pol2)

        def get_collection_products(
                bls_ordering: Iterable[Tuple[str, str]]) -> List[List[int]]:
            """This is a clone (and cleanup) of :func:`katsdpdisp.data.set_bls`."""
            auto = []       # type: List[int]
            autohh = []     # type: List[int]
            autovv = []     # type: List[int]
            autohv = []     # type: List[int]
            cross = []      # type: List[int]
            crosshh = []    # type: List[int]
            crossvv = []    # type: List[int]
            crosshv = []    # type: List[int]
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
            antenna_mask_set: Optional[Set[str]] = set(antenna_mask)
        else:
            antenna_mask_set = None
        # Eliminate baselines not covered by antenna_mask_set
        filtered = [x for x in enumerate(cbf_bls_ordering) if keep(x[1])]
        if not filtered:
            raise ValueError('No baselines (bls_ordering = {}, antenna_mask = {})'.format(
                cbf_bls_ordering, antenna_mask_set))
        # Sort what's left
        reordered = sorted(filtered, key=key)
        # reordered contains the mapping from new position to original
        # position, but we need the inverse.
        self.permutation = [-1] * len(cbf_bls_ordering)
        for i in range(len(reordered)):
            self.permutation[reordered[i][0]] = i
        # Can now discard the indices from reordered
        reordered_bls = [x[1] for x in reordered]
        self.sdp_bls_ordering = np.array(reordered_bls)

        # Collect percentile ranges
        collection_products = get_collection_products(self.sdp_bls_ordering)
        self.percentile_ranges: List[Tuple[int, int]] = []
        for p in collection_products:
            if p:
                start = p[0]
                end = p[-1] + 1
                if not np.array_equal(np.arange(start, end), p):
                    raise ValueError("percentile baselines are not contiguous: {}".format(p))
                self.percentile_ranges.append((start, end))
            else:
                self.percentile_ranges.append((0, 0))


class SystemAttrs:
    """Static system configuration retrieved from telescope state."""

    def __init__(self,
                 cbf_attr: Mapping[str, Any],
                 rfi_mask_model: Optional[katsdpmodels.rfi_mask.RFIMask],
                 band_mask_model: Optional[katsdpmodels.band_mask.BandMask],
                 antennas: Iterable[katpoint.Antenna]) -> None:
        self.cbf_attr = dict(cbf_attr)
        self.rfi_mask_model = rfi_mask_model
        self.band_mask_model = band_mask_model
        self.antennas = list(antennas)

    @staticmethod
    async def _get_rfi_mask_model(fetcher: katsdpmodels.fetch.aiohttp.TelescopeStateFetcher) \
            -> Optional[katsdpmodels.rfi_mask.RFIMask]:
        rfi_mask_model_key = fetcher.telstate.join('model', 'rfi_mask', 'fixed')
        try:
            rfi_mask_model = await fetcher.get(rfi_mask_model_key,
                                               katsdpmodels.rfi_mask.RFIMask)
            logger.info('Loaded rfi_mask model')
            return rfi_mask_model
        except (aiohttp.ClientError, katsdpmodels.models.ModelError) as exc:
            logger.warning('Failed to load rfi_mask model: %s', exc, exc_info=True)
            return None

    @staticmethod
    async def _get_band_mask_model(fetcher: katsdpmodels.fetch.aiohttp.TelescopeStateFetcher,
                                   telstate_cbf: katsdptelstate.aio.TelescopeState) \
            -> Optional[katsdpmodels.band_mask.BandMask]:
        band_mask_model_key = telstate_cbf.join('model', 'band_mask', 'fixed')
        try:
            band_mask_model = await fetcher.get(band_mask_model_key,
                                                katsdpmodels.band_mask.BandMask,
                                                telstate=telstate_cbf)
            logger.info('Loaded band_mask model')
            return band_mask_model
        except (aiohttp.ClientError, katsdpmodels.models.ModelError) as exc:
            logger.warning('Failed to load band_mask model: %s', exc, exc_info=True)
            return None

    @staticmethod
    async def _get_antennas(input_labels: Iterable[str],
                            antenna_mask: Optional[Iterable[str]],
                            telstate: katsdptelstate.aio.TelescopeState) -> List[katpoint.Antenna]:
        antenna_names = set(label[:-1] for label in input_labels)
        if antenna_mask is not None:
            antenna_names &= set(antenna_mask)
        return [katpoint.Antenna(await telstate['{}_observer'.format(name)])
                for name in antenna_names]

    @classmethod
    async def create(cls,
                     fetcher: katsdpmodels.fetch.aiohttp.TelescopeStateFetcher,
                     telstate_cbf: katsdptelstate.aio.TelescopeState,
                     antenna_mask: Optional[Iterable[str]] = None) -> 'SystemAttrs':
        telstate = fetcher.telstate

        cbf_attr = {}
        for attr in CBF_CRITICAL_ATTRS:
            cbf_attr[attr] = await telstate_cbf[attr]
            logger.info('Setting cbf_attr %s to %s',
                        attr, textwrap.shorten(repr(cbf_attr[attr]), 50))

        rfi_mask_model = await cls._get_rfi_mask_model(fetcher)
        band_mask_model = await cls._get_band_mask_model(fetcher, telstate_cbf)
        antennas = await cls._get_antennas(cbf_attr['input_labels'], antenna_mask, telstate)
        logger.info('All metadata received from telstate')
        return cls(cbf_attr, rfi_mask_model, band_mask_model, antennas)


class TelstateReceiver(receiver.Receiver):
    """Receiver that uses telescope state to coordinate a shared first dump timestamp.

    It supports multiple telescope states (typically views on the same backing
    store), and puts the key in all of them. This critically depends on all instances
    using the same ordering to avoid race conditions.

    Parameters
    ----------
    telstates : list of :class:`katsdptelstate.aio.TelescopeState`
        Telescope state views with scopes unique to the capture session (but shared
        across cooperating ingest servers).
    l0_int_time : float
        Output integration time
    """
    def __init__(self, *args, **kwargs) -> None:
        self._telstates: List[katsdptelstate.aio.TelescopeState] = kwargs.pop('telstates')
        self._l0_int_time: float = kwargs.pop('l0_int_time')
        super().__init__(*args, **kwargs)

    async def _first_timestamp(self, candidate: int) -> int:
        scaled = candidate / self.cbf_attr['scale_factor_timestamp'] + 0.5 * self._l0_int_time
        try:
            for telstate in self._telstates:
                await telstate.set('first_timestamp_adc', candidate)
                await telstate.set('first_timestamp', scaled)
            return candidate
        except katsdptelstate.ImmutableKeyError:
            # A different ingest process beat us to it. Use its value.
            # That other process will fill in the remaining values
            return await self._telstates[0].get('first_timestamp_adc')


class SensorHistory:
    """Track sensor values until they're needed by the corresponding frame job.

    Samples are associated with a dump index, and there can be at most one
    sample per dump index (to prevent a high update rate from filling memory).
    Samples should be added in increasing order of time: samples with a lower
    dump index than the latest one are ignored and a warning is logged.

    A single querier can request the latest sample that is no later than a
    given dump index. Such queries must be made with non-decreasing dump
    indices.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._data: Deque[Tuple[int, Any]] = deque()
        self._last_get = -1

    def add(self, dump_idx: int, value: Any) -> None:
        """Add a new sample of the sensor."""
        if dump_idx <= self._last_get:
            logger.warning('Sensor update for %s index %d is too late', self.name, dump_idx)
            return
        if self._data:
            if dump_idx < self._data[-1][0]:
                logger.warning(
                    'Ignoring sensor update for %s that went backwards in time (%d < %d)',
                    self.name, dump_idx, self._data[-1][0]
                )
                return
            elif dump_idx == self._data[-1][0]:
                # May be more than one sensor update during a dump period.
                # Assume that time ran forwards, so the new value is more
                # up-to-date. Remove the previous value.
                self._data.pop()
        self._data.append((dump_idx, value))

    def get(self, dump_idx: int, default: Any = None) -> Any:
        """Obtain the last sample value that is no later than `dump_idx`.

        If there is no such sample, returns `default`.

        Raises
        ------
        ValueError
            if `dump_idx` is less than in the previous call.
        """
        if dump_idx < self._last_get:
            raise ValueError('Query dump indices must not go backwards (%d < %d)',
                             dump_idx, self._last_get)
        while len(self._data) > 1 and self._data[1][0] <= dump_idx:
            self._data.popleft()
        self._last_get = dump_idx
        if not self._data or self._data[0][0] > dump_idx:
            return default
        else:
            return self._data[0][1]


def task_wrapper(name: str, fail_status: DeviceStatus, restart: bool = False):
    """Decorator for tasks that run asynchronously during an ingest session.

    If the task raises an exception, it is logged, and the device status is
    set to at least `fail_state`.
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            while True:
                try:
                    return await func(self, *args, **kwargs)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.error('%s threw an uncaught exception%s',
                                 name, ', restarting' if restart else '',
                                 exc_info=True)
                    sensor = self._my_sensors['device-status']
                    if fail_status.value > sensor.value.value:
                        sensor.value = fail_status
                    if restart:
                        await asyncio.sleep(5)  # To avoid thrashing if it's a repeating failure
                    else:
                        break

        return wrapper
    return decorator


class CBFIngest:
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
        same for `cont_` (if enabled).
    sd_input_resource : :class:`_ResourceSet`
        Wrapper of the buffers that serve as inputs to signal display
        processing (currently `timeseries_weights`).
    sd_output_resource : :class:`_ResourceSet`
        Wrapper of the signal display output device buffers.
    proc_resource : :class:`katsdpsigproc.resource.Resource`
        The proc object, and the contents of all its buffers except for those
        covered by other resources above.
    sensor_resource : :class:`katsdpsigproc.resource.Resource`
        Ensures in-order access to the :class:`SensorHistory` objects.
    rx : :class:`katsdpingest.receiver.Receiver`
        Receiver that combines data from the SPEAD streams into frames
    cbf_attr : dict
        Input stream attributes, as returned by :func:`get_cbf_attr`
    bls_ordering : :class:`BaselineOrdering`
        Baseline ordering and related tables
    bls_channel_mask_idx : :class:`np.ndarray`, 1D
        Index into list of channel masks for each baseline. The baselines
        correspond to `bls_ordering.sdp_bls_ordering`.
    static_masks : :class:`np.ndarray`, 2D
        Static channel masks. Each row contains one channel mask, and the
        rows are indexed by the values in bls_channel_mask_idx.
    telstate : :class:`katsdptelstate.aio.TelescopeState`
        Global view of telescope state
    l0_names : list of str
        Stream names of the L0 stream, for those streams being transmitted
    capture_block_id : str
        Current capture block ID, or ``None`` if not capturing
    """

    # To avoid excessive autotuning, the following parameters are quantised up
    # to the next element of these lists when generating templates. These
    # lists are also used by ingest_autotune.py for pre-tuning standard
    # configurations.
    tune_channels = [4096, 8192, 9216, 32768]
    tune_percentile_sizes_set = set()
    for ants in [2, 4, 8, 16, 32, 64]:
        tune_percentile_sizes_set.update(
            {ants, 2 * ants, ants * (ants - 1) // 2, ants * (ants - 1)})
    tune_percentile_sizes = list(sorted(tune_percentile_sizes_set))

    @classmethod
    def _tune_next(cls, value: int, predef: Iterable[int]) -> int:
        """Return the smallest value in `predef` greater than or equal to
        `value`, or `value` if it is larger than the largest element of
        `predef`."""
        valid = [x for x in predef if x >= value]
        if valid:
            return min(valid)
        else:
            return value

    @classmethod
    def create_proc_template(
            cls, context, percentile_sizes: List[int],
            max_channels: int, excise: bool, continuum: bool) -> sigproc.IngestTemplate:
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
            Excise flagged data by downweighting it massively
        continuum : bool
            Enable continuum averaging
        """
        # Quantise to reduce number of options to autotune
        max_percentile_sizes = [cls._tune_next(s, cls.tune_percentile_sizes)
                                for s in percentile_sizes]
        max_percentile_sizes = list(sorted(set(max_percentile_sizes)))
        max_channels = cls._tune_next(max_channels, cls.tune_channels)

        background_template = rfi.BackgroundMedianFilterDeviceTemplate(
            context, width=13, use_flags=rfi.BackgroundFlags.FULL)
        noise_est_template = rfi.NoiseEstMADTDeviceTemplate(context, max_channels=max_channels)
        threshold_template = rfi.ThresholdSimpleDeviceTemplate(
            context, transposed=True, flag_value=INGEST_RFI)
        flagger_template = rfi.FlaggerDeviceTemplate(
            background_template, noise_est_template, threshold_template)
        return sigproc.IngestTemplate(context, flagger_template,
                                      percentile_sizes=max_percentile_sizes,
                                      excise=excise, continuum=continuum)

    def _zero_counters(self) -> None:
        now = time.time()
        self.output_bytes_sensor.set_value(0, timestamp=now)
        self.output_heaps_sensor.set_value(0, timestamp=now)
        self.output_dumps_sensor.set_value(0, timestamp=now)
        self.output_flagged_sensor.set_value(0, timestamp=now)
        self.output_vis_sensor.set_value(0, timestamp=now)

    def _init_baselines(self, system_attrs: SystemAttrs) -> None:
        # Configure the masking and reordering of baselines
        self.bls_ordering = BaselineOrdering(
            system_attrs.cbf_attr['bls_ordering'],
            [antenna.name for antenna in system_attrs.antennas]
        )

        # Pre-compute channel masks from the RFI mask and band mask models
        cbf_spw = SpectralWindow(
            system_attrs.cbf_attr['center_freq'], None, len(self.channel_ranges.cbf),
            bandwidth=system_attrs.cbf_attr['bandwidth'], sideband=1)
        spw = cbf_spw.subrange(self.channel_ranges.input.start,
                               self.channel_ranges.input.stop)
        freqs = spw.channel_freqs << u.Hz

        if system_attrs.rfi_mask_model is None:
            self.bls_channel_mask_idx = np.zeros(len(self.bls_ordering.sdp_bls_ordering),
                                                 np.uint32)
            self.static_masks = np.zeros((1, len(self.channel_ranges.input)), np.uint8)
        else:
            bls = self.bls_ordering.sdp_bls_ordering
            ref = system_attrs.antennas[0].array_reference_antenna()
            # Turn each antenna into a location East/North/Up of the reference
            locations = {antenna.name: np.array(ref.baseline_toward(antenna))
                         for antenna in system_attrs.antennas}
            lengths = np.array([np.linalg.norm(locations[a[:-1]] - locations[b[:-1]])
                                for (a, b) in bls]) << u.m
            # Get masks as a baseline x freq array
            masks = system_attrs.rfi_mask_model.is_masked(
                freqs[np.newaxis, :],
                lengths[:, np.newaxis]
            )
            # Typically the model will only have a few threshold lengths, so
            # most rows will be the same. Reduce it to just unique rows, with
            # a lookup table.
            masks, indices = np.unique(masks, axis=0, return_inverse=True)
            self.bls_channel_mask_idx = indices.astype(np.uint32)
            self.static_masks = masks * np.uint8(STATIC)

        if system_attrs.band_mask_model is not None:
            band_spw = katsdpmodels.band_mask.SpectralWindow(
                cbf_spw.bandwidth * u.Hz,
                cbf_spw.centre_freq * u.Hz
            )
            mask = system_attrs.band_mask_model.is_masked(band_spw, freqs)
            self.static_masks |= mask[np.newaxis, :] * np.uint8(STATIC)

    def _init_time_averaging(self, output_int_time: float, sd_int_time: float) -> None:
        output_ratio = max(1, int(round(output_int_time / self.cbf_attr['int_time'])))
        self._output_avg = _TimeAverage(output_ratio, self._flush_output)
        logger.info("Averaging {0} input dumps per output dump".format(self._output_avg.ratio))

        sd_ratio = max(1, int(round(sd_int_time / self.cbf_attr['int_time'])))
        self._sd_avg = _TimeAverage(sd_ratio, self._flush_sd)
        logger.info("Averaging {0} input dumps per signal display dump".format(
                    self._sd_avg.ratio))

    def _init_sensors(self, my_sensors: Mapping[str, Sensor]) -> None:
        self._my_sensors = my_sensors
        # Autocorrelations are required, so it suffices to take just the first
        # input from each pair to get all inputs.
        inputs = set(bl[0] for bl in self.bls_ordering.sdp_bls_ordering)
        antennas = set(input[:-1] for input in inputs)
        my_sensors['output-n-inputs'].value = len(inputs)
        my_sensors['output-n-ants'].value = len(antennas)
        my_sensors['output-n-bls'].value = len(self.bls_ordering.sdp_bls_ordering)
        my_sensors['output-n-chans'].value = len(self.channel_ranges.output)
        my_sensors['output-int-time'].value = self.cbf_attr['int_time'] * self._output_avg.ratio
        self.output_bytes_sensor = my_sensors['output-bytes-total']       # type: Sensor[int]
        self.output_heaps_sensor = my_sensors['output-heaps-total']       # type: Sensor[int]
        self.output_dumps_sensor = my_sensors['output-dumps-total']       # type: Sensor[int]
        self.output_flagged_sensor = my_sensors['output-flagged-total']   # type: Sensor[int]
        self.output_vis_sensor = my_sensors['output-vis-total']           # type: Sensor[int]
        self.status_sensor = my_sensors['status']                         # type: Sensor[Status]
        self.status_sensor.value = Status.INIT
        self._zero_counters()

    def _init_proc(self, context, excise: bool, continuum: bool) -> None:
        percentile_sizes = list(set(r[1] - r[0] for r in self.bls_ordering.percentile_ranges))
        proc_template = self.create_proc_template(
            context, percentile_sizes, len(self.channel_ranges.input), excise, continuum)
        self.command_queue = proc_template.context.create_command_queue()
        self.proc = proc_template.instantiate(
            self.command_queue, len(self.channel_ranges.input),
            self.channel_ranges.computed.relative_to(self.channel_ranges.input),
            self.channel_ranges.sd_output.relative_to(self.channel_ranges.input),
            len(self.cbf_attr['bls_ordering']),
            len(self.bls_ordering.sdp_bls_ordering),
            self.static_masks.shape[0],
            self.channel_ranges.cont_factor,
            self.channel_ranges.sd_cont_factor,
            self.bls_ordering.percentile_ranges,
            threshold_args={'n_sigma': 11.0})
        self.proc.n_accs = self.cbf_attr['n_accs']
        self.proc.ensure_all_bound()
        self.proc.buffer('permutation').set(
            self.command_queue, np.asarray(self.bls_ordering.permutation, dtype=np.int16))
        self.proc.buffer('channel_mask_idx').set(self.command_queue, self.bls_channel_mask_idx)
        self.proc.start_sum()
        self.proc.start_sd_sum()
        logger.debug("\nProcessing Blocks\n=================\n")
        for description in self.proc.descriptions():
            logger.debug("\t".join([str(x) for x in description]))

    def _init_resources(self) -> None:
        self.jobs = resource.JobQueue()
        self.proc_resource = resource.Resource(self.proc)
        self.sensor_resource = resource.Resource(self._telstate_values)
        self.input_resource = _ResourceSet(
            self.proc, ['vis_in', 'channel_mask', 'baseline_flags'], 2)
        self.output_resource = _ResourceSet(
            self.proc, [prefix + '_' + suffix
                        for prefix in self.tx
                        for suffix in ['vis', 'flags', 'weights', 'weights_channel']], 2)
        self.sd_input_resource = _ResourceSet(self.proc, ['timeseries_weights'], 2)
        sd_output_names = ['sd_cont_vis', 'sd_cont_flags', 'sd_spec_vis', 'sd_spec_flags',
                           'timeseries', 'timeseriesabs',
                           'sd_flag_counts', 'sd_flag_any_counts']
        for i in range(len(self.proc.percentiles)):
            base_name = 'percentile{}'.format(i)
            sd_output_names.append(base_name)
            sd_output_names.append(base_name + '_flags')
        self.sd_output_resource = _ResourceSet(self.proc, sd_output_names, 2)

    def _init_tx_one(self, args: argparse.Namespace, arg_name: str, name: str,
                     cont_factor: int) -> None:
        """Initialise a single transmit stream.

        If the stream has no endpoint specified, does nothing. Otherwise stores
        the :class:`VisSenderSet` into `self.tx[name]`.

        Parameters
        ----------
        args : :class:`argparse.Namespace`
            Command-line arguments
        arg_name : {'spectral', 'continuum'}
            Name used in command-line arguments
        name : {'spec', 'cont'}
            Name used in internal data structures
        cont_factor : int
            Continuum factor (1 for spectral product)
        """
        endpoints: List[Endpoint] = getattr(args, 'l0_{}_spead'.format(arg_name))
        if not endpoints:
            return

        l0_flavour = spead2.Flavour(4, 64, 48)
        all_output = self.channel_ranges.all_output
        # Compute channel ranges relative to those computed
        spectral_channels = self.channel_ranges.output.relative_to(self.channel_ranges.computed)
        channels = utils.Range(spectral_channels.start // cont_factor,
                               spectral_channels.stop // cont_factor)
        baselines = len(self.bls_ordering.sdp_bls_ordering)
        if len(endpoints) % args.servers:
            raise ValueError('Number of endpoints ({}) not divisible by number of servers ({})'
                             .format(len(endpoints), args.servers))
        endpoint_lo: int = (args.server_id - 1) * len(endpoints) // args.servers
        endpoint_hi: int = args.server_id * len(endpoints) // args.servers
        endpoints = endpoints[endpoint_lo:endpoint_hi]
        logger.info('Sending %s output to %s', arg_name, endpoints_to_str(endpoints))
        int_time = self.cbf_attr['int_time'] * self._output_avg.ratio
        tx = sender.VisSenderSet(
            spead2.ThreadPool(),
            endpoints,
            katsdpservices.get_interface_address(getattr(args, 'l0_{}_interface'.format(arg_name))),
            l0_flavour,
            int_time * args.clock_ratio,
            channels,
            (self.channel_ranges.output.start - all_output.start) // cont_factor,
            len(all_output) // cont_factor,
            baselines)

        # Put attributes into telstate. This will be done by all the ingest
        # nodes, with the same values.
        prefix: str = getattr(args, 'l0_{}_name'.format(arg_name))
        cbf_spw = SpectralWindow(
            self.cbf_attr['center_freq'], None, len(self.channel_ranges.cbf),
            bandwidth=self.cbf_attr['bandwidth'], sideband=1)
        output_spw = cbf_spw.subrange(all_output.start, all_output.stop)
        output_spw = output_spw.rechannelise(len(all_output) // cont_factor)

        self._telstate_updates.append((prefix, 'n_chans', output_spw.num_chans))
        self._telstate_updates.append((prefix, 'n_chans_per_substream', tx.sub_channels))
        self._telstate_updates.append((prefix, 'n_bls', baselines))
        self._telstate_updates.append((prefix, 'bls_ordering', self.bls_ordering.sdp_bls_ordering))
        self._telstate_updates.append((prefix, 'sync_time', self.cbf_attr['sync_time']))
        self._telstate_updates.append((prefix, 'bandwidth', output_spw.bandwidth))
        self._telstate_updates.append((prefix, 'center_freq', output_spw.centre_freq))
        self._telstate_updates.append((prefix, 'channel_range', all_output.astuple()))
        self._telstate_updates.append((prefix, 'int_time', int_time))
        self._telstate_updates.append((prefix, 'excise', args.excise))
        self._telstate_updates.append((prefix, 'src_streams', [self.src_stream]))
        self._telstate_updates.append((prefix, 'stream_type', 'sdp.vis'))
        self._telstate_updates.append((prefix, 'calibrations_applied', []))
        self._telstate_updates.append((prefix, 'need_weights_power_scale', True))
        self.tx[name] = tx
        self.l0_names.append(prefix)

    def _init_tx(self, args: argparse.Namespace) -> None:
        self.tx: Dict[str, sender.VisSenderSet] = {}
        self.l0_names: List[str] = []
        self._init_tx_one(args, 'spectral', 'spec', 1)
        self._init_tx_one(args, 'continuum', 'cont', self.channel_ranges.cont_factor)

    def _init_ig_sd(self) -> None:
        """Create a item group for signal displays."""
        sd_flavour = spead2.Flavour(4, 64, 48)
        inline_format = [('u', sd_flavour.heap_address_bits)]
        n_spec_channels = len(self.channel_ranges.sd_output)
        n_cont_channels = n_spec_channels // self.channel_ranges.sd_cont_factor
        all_spec_channels = len(self.channel_ranges.all_sd_output)
        all_cont_channels = all_spec_channels // self.channel_ranges.sd_cont_factor
        n_baselines = len(self.bls_ordering.sdp_bls_ordering)
        self.ig_sd = spead2.send.ItemGroup(flavour=sd_flavour)
        # If any items are added/changed here, update _timeplot_frame_size in
        # katsdpcontroller/generate.py as well.
        self.ig_sd.add_item(
            name='sd_data', id=0x3501, description="Combined raw data from all x engines.",
            format=[('f', 32)], shape=(n_spec_channels, None, 2))
        self.ig_sd.add_item(
            name='sd_data_index', id=0x3509, description="Indices for transmitted sd_data.",
            format=[('u', 32)], shape=(None,))
        self.ig_sd.add_item(
            name='sd_blmxdata', id=0x3507, description="Reduced data for baseline matrix.",
            shape=(n_cont_channels, n_baselines, 2),
            dtype=np.float32)
        self.ig_sd.add_item(
            name='sd_flags', id=0x3503, description="8bit packed flags for each data point.",
            format=[('u', 8)], shape=(n_spec_channels, None))
        self.ig_sd.add_item(
            name='sd_blmxflags', id=0x3508,
            description="Reduced data flags for baseline matrix.",
            shape=(n_cont_channels, n_baselines), dtype=np.uint8)
        self.ig_sd.add_item(
            name='sd_timeseries', id=0x3504, description="Computed timeseries.",
            shape=(n_baselines, 2), dtype=np.float32)
        self.ig_sd.add_item(
            name='sd_timeseriesabs', id=0x3510, description="Computed timeseries magnitude.",
            shape=(n_baselines,), dtype=np.float32)
        n_perc_signals = 0
        perc_idx = 0
        while True:
            try:
                n_perc_signals += self.proc.buffer('percentile{0}'.format(perc_idx)).shape[0]
                perc_idx += 1
            except KeyError:
                break
        self.ig_sd.add_item(
            name='sd_percspectrum', id=0x3505,
            description="Percentiles of spectrum data.",
            dtype=np.float32, shape=(n_spec_channels, n_perc_signals))
        self.ig_sd.add_item(
            name='sd_percspectrumflags', id=0x3506,
            description="Flags for percentiles of spectrum.",
            dtype=np.uint8, shape=(n_spec_channels, n_perc_signals))
        self.ig_sd.add_item(
            name='sd_timestamp', id=0x3502,
            description='Timestamp of this sd frame in centiseconds since epoch.',
            shape=(), dtype=None, format=inline_format)
        bls_ordering = np.asarray(self.bls_ordering.sdp_bls_ordering)
        if bls_ordering.dtype.kind == 'U':
            # Bandwidth calculations assume it is 1-byte ASCII not 4-byte UCS-4
            bls_ordering = np.char.encode(bls_ordering)
        self.ig_sd.add_item(
            name='bls_ordering', id=0x100C,
            description="Mapping of antenna/pol pairs to data output products.",
            shape=bls_ordering.shape, dtype=bls_ordering.dtype, value=bls_ordering)
        cbf_spw = SpectralWindow(
            self.cbf_attr['center_freq'], None, len(self.channel_ranges.cbf),
            bandwidth=self.cbf_attr['bandwidth'], sideband=1)
        sd_spw = cbf_spw.subrange(*self.channel_ranges.all_sd_output.astuple())
        self.ig_sd.add_item(
            name="bandwidth", id=0x1013,
            description="The analogue bandwidth of the signal display product in Hz.",
            shape=(), dtype=None, format=[('f', 64)], value=sd_spw.bandwidth)
        self.ig_sd.add_item(
            name="center_freq", id=0x1011,
            description="The center frequency of the signal display product in Hz.",
            shape=(), dtype=None, format=[('f', 64)], value=sd_spw.centre_freq)
        self.ig_sd.add_item(
            name="n_chans", id=0x1009,
            description="The total number of frequency channels in the signal display product.",
            shape=(), dtype=None, format=inline_format,
            value=all_spec_channels)
        self.ig_sd.add_item(
            name="sd_blmx_n_chans", id=0x350A,
            description="The total number of frequency channels in the baseline matrix product.",
            shape=(), dtype=None, format=inline_format,
            value=all_cont_channels)
        self.ig_sd.add_item(
            name='sd_flag_fraction', id=0x350B,
            description="Fraction of channels having each flag bit, per baseline",
            shape=(n_baselines, 8), dtype=np.float32)
        self.ig_sd.add_item(
            name="frequency", id=0x4103,
            description="The frequency channel of the data in this heap.",
            shape=(), dtype=None, format=inline_format,
            value=self.channel_ranges.sd_output.start - self.channel_ranges.all_sd_output.start)

    def __init__(self, args: argparse.Namespace, system_attrs: SystemAttrs,
                 channel_ranges: ChannelRanges,
                 context,
                 my_sensors: Mapping[str, Sensor],
                 telstate_cbf: katsdptelstate.aio.TelescopeState) -> None:
        self._sdisp_ips: Dict[str, spead2.send.asyncio.UdpStream] = {}
        self._run_task: Optional[asyncio.Task] = None
        self._sidecar_tasks: List[asyncio.Task] = []
        # Local cache of values from telstate_sdisp
        self._telstate_sdisp_values: Dict[str, Any] = {}
        # Local cache of values from various telstates
        self._telstate_values: Dict[str, SensorHistory] = {}
        # Set by stop to abort prior to creating the receiver
        self._stopped = True
        self.capture_block_id: Optional[str] = None

        self.rx_spead_endpoints: List[Endpoint] = args.cbf_spead
        self.rx_spead_ifaddr = katsdpservices.get_interface_address(args.cbf_interface)
        self.rx_spead_ibv: bool = args.cbf_ibv
        self.rx_spead_max_streams: int = args.input_streams
        self.rx_spead_max_packet_size: int = args.input_max_packet_size
        self.rx_spead_buffer_size: int = args.input_buffer
        self.sd_spead_rate: float = (
            args.sd_spead_rate / args.clock_ratio if args.clock_ratio else 0.0
        )
        self.sd_spead_ifaddr = katsdpservices.get_interface_address(args.sdisp_interface)
        self.channel_ranges = channel_ranges
        self.telstate = telstate_cbf.root()
        self.telstate_cbf = telstate_cbf
        self.telstate_sdisp = self.telstate.view('sdp', exclusive=True).view(args.l0_spectral_name)
        self.cbf_attr = system_attrs.cbf_attr
        self.src_stream: str = args.cbf_name
        self.use_data_suspect: bool = args.use_data_suspect
        self._telstate_updates: List[Tuple[str, str, Any]] = []   # Values to populate in telstate

        self._init_baselines(system_attrs)
        self._init_time_averaging(args.output_int_time, args.sd_int_time)
        self._init_sensors(my_sensors)
        self._init_tx(args)  # NB: must be run after _init_time_averaging, before _init_proc
        self._init_proc(context, args.excise, 'cont' in self.tx)
        self._init_resources()

        # Instantiation of input streams is delayed until the asynchronous task
        # is running, to avoid receiving data we're not yet ready for.
        self.rx: Optional[receiver.Receiver] = None
        self._init_ig_sd()

        # Record information about the processing in telstate
        if args.name is not None:
            descriptions = _fix_descriptions(list(self.proc.descriptions()))
            process_prefix = args.name.replace('.', '_')
            self._telstate_updates.append((process_prefix, 'process_log', descriptions))

    async def populate_telstate(self) -> None:
        """Update telescope state with information about the outputs."""
        for prefix, name, value in self._telstate_updates:
            try:
                full_name = self.telstate.join(prefix, name)
                await self.telstate.set(full_name, value)
            except katsdptelstate.ImmutableKeyError as error:
                logger.warning('%s', error)
        self._telstate_updates = []

    def enable_debug(self, debug: bool) -> None:
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.NOTSET)

    def _send_sd_data(self, data: spead2.send.Heap) -> asyncio.Future:
        """Send a heap to all signal display servers, asynchronously.

        Parameters
        ----------
        data : :class:`spead2.send.Heap`
            Heap to send

        Returns
        -------
        future : `asyncio.Future`
            A future that is completed when the heap has been sent to all receivers.
        """
        return asyncio.gather(*(sender.async_send_heap(tx, data)
                                for tx in self._sdisp_ips.values()))

    async def _stop_stream(self, stream: spead2.send.asyncio.UdpStream,
                           ig: spead2.send.ItemGroup) -> None:
        """Send a stop packet to a stream. To ensure that it won't be lost
        on the sending side, the stream is first flushed, then the stop
        heap is sent and waited for."""
        await stream.async_flush()
        await stream.async_send_heap(ig.get_end())

    async def drop_sdisp_ip(self, ip: str) -> None:
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
            await self._stop_stream(stream, self.ig_sd)

    def add_sdisp_ip(self, endpoint: Endpoint) -> None:
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
        config = spead2.send.StreamConfig(max_packet_size=8872, rate=self.sd_spead_rate / 8)
        logger.info("Adding %s to signal display list. Starting stream...", endpoint)
        if self.sd_spead_ifaddr is None:
            extra_args: Dict[str, Any] = {}
        else:
            extra_args = dict(ttl=1, interface_address=self.sd_spead_ifaddr)
        stream = spead2.send.asyncio.UdpStream(
            spead2.ThreadPool(), endpoint.host, endpoint.port, config, **extra_args)
        # Ensure that signal display streams that form the full band between
        # them always have unique heap cnts. The first output channel is used
        # as a unique key.
        stream.set_cnt_sequence(self.channel_ranges.sd_output.start,
                                len(self.channel_ranges.cbf))
        self._sdisp_ips[endpoint.host] = stream

    async def _flush_output(self, output_idx: int) -> None:
        """Finalise averaging of a group of input dumps and emit an output dump"""
        proc_a = self.proc_resource.acquire()
        output_a, host_output_a = self.output_resource.acquire()
        self.jobs.add(self._flush_output_job(proc_a, output_a, host_output_a, output_idx))

    async def _flush_output_job(
            self,
            proc_a: resource.ResourceAllocation,
            output_a: resource.ResourceAllocation,
            host_output_a: resource.ResourceAllocation,
            output_idx: int) -> None:
        with proc_a as proc, output_a as output, host_output_a as host_output:
            # Wait for resources
            events = await proc_a.wait()
            events += await output_a.wait()
            self.command_queue.enqueue_wait_for_events(events)

            # Compute
            proc.end_sum()
            self.command_queue.flush()

            # Wait for the host output buffers to be available
            events = await host_output_a.wait()
            self.command_queue.enqueue_wait_for_events(events)

            # Transfer
            data: Dict[str, sender.Data] = {}
            for prefix in self.tx:
                kwargs: Dict[str, np.ndarray] = {}
                for field in ['vis', 'flags', 'weights', 'weights_channel']:
                    name = prefix + '_' + field
                    kwargs[field] = host_output[name]
                    output[name].get_async(self.command_queue, host_output[name])
                data[prefix] = sender.Data(**kwargs)
            transfer_done = self.command_queue.enqueue_marker()
            # Prepare for the next group.
            proc.start_sum()
            proc_done = self.command_queue.enqueue_marker()
            self.command_queue.flush()
            proc_a.ready([proc_done])
            output_a.ready([proc_done])

            assert self.rx is not None     # keep mypy happy
            ts_rel = _mid_timestamp_rel(self._output_avg, self.rx, output_idx)
            await resource.async_wait_for_events([transfer_done])
            futures = []
            # Compute deltas before updating the sensors, so that only a single
            # update is observed.
            inc_bytes = 0
            inc_heaps = 0
            inc_dumps = 0
            for (name, tx) in self.tx.items():
                part = data[name]
                inc_bytes += part.nbytes
                inc_heaps += tx.size
                inc_dumps += 1
                futures.append(tx.send(part, output_idx, ts_rel))
            await asyncio.gather(*futures)
            now = time.time()
            self.output_bytes_sensor.increment(inc_bytes, timestamp=now)
            self.output_heaps_sensor.increment(inc_heaps, timestamp=now)
            self.output_dumps_sensor.increment(inc_dumps, timestamp=now)
            host_output_a.ready()
            logger.debug("Finished dump group with index %d", output_idx)

    async def _flush_sd(self, output_idx: int) -> None:
        """Finalise averaging of a group of dumps for signal display, and send
        signal display data to the signal display server"""
        all_channels = len(self.channel_ranges.all_sd_output)
        try:
            custom_signals_indices = self._telstate_sdisp_values['sdisp_custom_signals']
        except KeyError:
            custom_signals_indices = np.array([], dtype=np.uint32)

        try:
            full_mask = self._telstate_sdisp_values['sdisp_timeseries_mask']
            if full_mask.shape != (all_channels,):
                raise ValueError
        except (KeyError, ValueError, TypeError):
            full_mask = np.ones(all_channels, np.float32) / all_channels

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
            output_idx, custom_signals_indices, mask))

    async def _flush_sd_job(
            self,
            proc_a: resource.ResourceAllocation,
            sd_input_a: resource.ResourceAllocation,
            host_sd_input_a: resource.ResourceAllocation,
            sd_output_a: resource.ResourceAllocation,
            host_sd_output_a: resource.ResourceAllocation,
            output_idx: int,
            custom_signals_indices: np.ndarray,
            mask: np.ndarray) -> None:
        with proc_a as proc, \
                sd_input_a as sd_input_buffers, \
                host_sd_input_a as host_sd_input, \
                sd_output_a as sd_output_buffers, \
                host_sd_output_a as host_sd_output:
            spec_channels = self.channel_ranges.sd_output.relative_to(
                self.channel_ranges.computed).asslice()
            assert spec_channels.start is not None    # needed just for mypy
            assert spec_channels.stop is not None     # needed just for mypy
            cont_channels = utils.Range(
                spec_channels.start // self.channel_ranges.sd_cont_factor,
                spec_channels.stop // self.channel_ranges.sd_cont_factor).asslice()
            # Copy inputs to HostArrays
            await host_sd_input_a.wait_events()
            host_sd_input['timeseries_weights'][:] = mask

            # Transfer to device
            events = await sd_input_a.wait()
            self.command_queue.enqueue_wait_for_events(events)
            for name in host_sd_input:
                sd_input_buffers[name].set_async(self.command_queue, host_sd_input[name])
            transfer_in_done = self.command_queue.enqueue_marker()
            host_sd_input_a.ready([transfer_in_done])
            self.command_queue.flush()

            # Compute
            events = await proc_a.wait()
            events += await sd_output_a.wait()
            self.command_queue.enqueue_wait_for_events(events)
            proc.end_sd_sum()
            sd_input_a.ready([self.command_queue.enqueue_marker()])
            self.command_queue.flush()

            # Transfer back to host
            events = await host_sd_output_a.wait()
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
            await resource.async_wait_for_events([transfer_out_done])
            assert self.rx is not None    # keeps mypy happy
            ts_rel = _mid_timestamp_rel(self._sd_avg, self.rx, output_idx)
            ts = self.cbf_attr['sync_time'] + ts_rel
            cont_vis = host_sd_output['sd_cont_vis']
            cont_flags = host_sd_output['sd_cont_flags']
            spec_vis = host_sd_output['sd_spec_vis']
            spec_flags = host_sd_output['sd_spec_flags']
            timeseries = host_sd_output['timeseries']
            timeseriesabs = host_sd_output['timeseriesabs']
            flag_counts = host_sd_output['sd_flag_counts']
            flag_counts_scale = self._sd_avg.ratio * len(self.channel_ranges.sd_output)
            flag_fraction = flag_counts.astype(np.float32) / np.float32(flag_counts_scale)
            assert flag_fraction.dtype == np.float32
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
                logger.warning('sdisp_custom_signals out of range, not updating (%s)',
                               custom_signals_indices)
            self.ig_sd['sd_blmxdata'].value = _split_array(cont_vis[cont_channels, ...], np.float32)
            self.ig_sd['sd_blmxflags'].value = cont_flags[cont_channels, ...]
            self.ig_sd['sd_timeseries'].value = _split_array(timeseries, np.float32)
            self.ig_sd['sd_timeseriesabs'].value = timeseriesabs
            self.ig_sd['sd_percspectrum'].value = np.vstack(percentiles).transpose()
            self.ig_sd['sd_percspectrumflags'].value = np.vstack(percentiles_flags).transpose()
            self.ig_sd['sd_flag_fraction'].value = flag_fraction

            # Update sensors
            flag_any_count = int(np.sum(host_sd_output['sd_flag_any_counts']))
            n_baselines = len(self.bls_ordering.sdp_bls_ordering)
            now = time.time()
            self.output_flagged_sensor.increment(flag_any_count, now)
            self.output_vis_sensor.increment(flag_counts_scale * n_baselines, now)

            await self._send_sd_data(self.ig_sd.get_heap(descriptors='all', data='all'))
            host_sd_output_a.ready()
            logger.debug("Finished SD group with index %d", output_idx)

    def _set_external_flags(self, baseline_flags: np.ndarray, channel_mask: np.ndarray,
                            dump_idx: int) -> None:
        """Query telstate for per-baseline flags and per-channel flags to set.

        The last value set prior to the end of the dump is used.

        This method must be called with the :attr:`sensor_resource` held, to
        ensure that :meth:`SensorHistory.get` is called with increasing dump
        indices.
        """
        static_flag = np.uint8(STATIC)
        cam_flag = np.uint8(CAM)
        channel_slice = self.channel_ranges.input.asslice()

        channel_mask_sensor = self._telstate_values['channel_mask'].get(dump_idx)
        if channel_mask_sensor is not None:
            if channel_mask_sensor.ndim == 2:
                logger.warning('2D channel_mask is no longer supported - update katsdpcam2telstate')
                channel_mask_sensor = channel_mask_sensor[0]
            dynamic_masks = channel_mask_sensor[channel_slice] * static_flag
            channel_mask[:] = self.static_masks | dynamic_masks
        else:
            channel_mask[:] = self.static_masks

        if self.use_data_suspect:
            channel_data_suspect = self._telstate_values['channel_data_suspect'].get(dump_idx)
            if channel_data_suspect is not None:
                channel_mask[:] |= channel_data_suspect[np.newaxis, channel_slice] * cam_flag

        baselines = self.bls_ordering.sdp_bls_ordering
        input_labels = self.cbf_attr['input_labels']
        input_suspect: Dict[str, bool] = {}
        if self.use_data_suspect:
            input_suspect_sensor = self._telstate_values['input_data_suspect'].get(dump_idx)
            if input_suspect_sensor is not None:
                input_suspect = {label: input_suspect_sensor[i]
                                 for i, label in enumerate(input_labels)}

        for i, baseline in enumerate(baselines):
            # [:-1] indexing strips off h/v pol
            a = baseline[0][:-1]
            b = baseline[1][:-1]
            flagged = False
            for antenna in (a, b):
                if self._telstate_values['{}_data_suspect'.format(antenna)].get(dump_idx):
                    flagged = True
            for input_ in baseline:
                if input_suspect.get(input_):
                    flagged = True
            baseline_flags[i] = cam_flag if flagged else 0

    async def _frame_job(
            self,
            proc_a: resource.ResourceAllocation,
            sensor_a: resource.ResourceAllocation,
            input_a: resource.ResourceAllocation,
            host_input_a: resource.ResourceAllocation,
            frame: receiver.Frame,
            timestamp: float) -> None:
        with proc_a as proc, sensor_a, input_a as input_buffers, host_input_a as host_input:
            vis_in_buffer = input_buffers['vis_in']
            vis_in = host_input['vis_in']
            baseline_flags = host_input['baseline_flags']
            channel_mask = host_input['channel_mask']
            # Load data
            await host_input_a.wait_events()
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

            await sensor_a.wait()
            self._set_external_flags(baseline_flags, channel_mask, frame.idx)
            sensor_a.ready()

            for item in frame.items:
                item_range = utils.Range(item_channel, item_channel + channels_per_item)
                item_channel = item_range.stop
                use_range = item_range.intersection(self.channel_ranges.input)
                if not use_range:
                    continue
                dest_range = use_range.relative_to(self.channel_ranges.input)
                src_range = use_range.relative_to(item_range)
                if item is None:
                    channel_mask[:, dest_range.asslice()] = DATA_LOST
                    vis_in[dest_range.asslice()] = 0
                else:
                    vis_in[dest_range.asslice()] = item[src_range.asslice()]
            del frame      # Free the memory back to the frame pool as soon as possible

            # Transfer data to the device
            events = await input_a.wait()
            self.command_queue.enqueue_wait_for_events(events)
            for name in input_buffers:
                input_buffers[name].set_async(self.command_queue, host_input[name])
            transfer_done = self.command_queue.enqueue_marker()
            self.command_queue.flush()
            host_input_a.ready([transfer_done])

            # Perform data processing
            events = await proc_a.wait()
            self.command_queue.enqueue_wait_for_events(events)
            proc()
            done_event = self.command_queue.enqueue_marker()
            input_a.ready([done_event])
            proc_a.ready([done_event])

    @property
    def capturing(self) -> bool:
        return self._run_task is not None

    def start(self, capture_block_id: str):
        assert self._run_task is None
        assert not self._sidecar_tasks
        assert self.rx is None
        assert self._stopped
        self._stopped = False
        self._telstate_sdisp_values = {}
        self._telstate_values = {}
        self.capture_block_id = capture_block_id
        loop = asyncio.get_event_loop()
        self._run_task = loop.create_task(self.run())

    async def stop(self) -> bool:
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
        future = self._run_task
        if future is not None:
            self._stopped = True
            # Give it a chance to stop on its own (due to stop items)
            logger.info('Waiting for run to stop (5s timeout)...')
            done, _ = await asyncio.wait([future], timeout=5)
            if future not in done:
                logger.info('Stopping receiver...')
                if self.rx is not None:
                    self.rx.stop()
                logger.info('Waiting for run to stop...')
                await future
            logger.info('Run stopped')
            # If multiple callers arrive here, we want only the first to
            # return True and clean up. We also need to protect against a prior
            # task having cleaned up and immediately started a new capture
            # session. In this case _run_task will be non-None (and hence
            # capturing will be True), but the object identity of _run_task
            # will no longer match future.
            if self._run_task is future:
                ret = True
                for task in self._sidecar_tasks:
                    task.cancel()
                await asyncio.wait(self._sidecar_tasks)
                self._sidecar_tasks = []
                self._run_task = None
                self.capture_block_id = None
                self.rx = None
            # spead2 versions up to 1.14.0 were known to produce cyclic
            # garbage, which could lead to OOM errors if not explicitly
            # collected.
            gc.collect()
        return ret

    def close(self) -> None:
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

    async def _watch_sensor(self, telstate: katsdptelstate.aio.TelescopeState, key: str,
                            notify: Callable[[Any, float], None]) -> None:
        """Subscribe to updates for a sensor.

        Each time the sensor is updated with a newer timestamp, the `notify`
        callback is called. If an update is received for an earlier point in
        time it is ignored. If there is already data in the sensor when called,
        the last value is immediately notified.
        """
        ts = None

        def callback(value: Any, new_ts: Optional[float]) -> bool:
            nonlocal ts
            if new_ts is None:
                logger.warning('%s was set to an immutable value, should be mutable', key)
                return False
            if ts is None or new_ts > ts:
                ts = new_ts
                notify(value, ts)
            else:
                logger.warning('%s timestamp went backwards (%.3f < %.3f)', new_ts, ts)
            return False

        # The callback always returns false, so this call to wait_key will
        # never return. Instead, it will be cancelled when the generator is
        # shut down.
        await telstate.wait_key(key, callback)

    async def _update_sdisp_sensor(self, key: str, dtype: np.dtype) -> None:
        def notify(value: Any, ts: float) -> None:
            value = np.array(value, dtype=dtype, copy=False)
            self._telstate_sdisp_values[key] = value
            logger.info('%s updated with timestamp %.3f and shape %s', key, ts, value.shape)

        await self._watch_sensor(self.telstate_sdisp, key, notify)

    async def _update_sensor(self, telstate: katsdptelstate.aio.TelescopeState, key: str) -> None:
        logger.debug('Starting _update_sensor for %s', key)
        assert self.rx is not None
        sh = self._telstate_values[key] = SensorHistory(key)
        time_base = self.cbf_attr['sync_time']
        time_base += self.rx.timestamp_base / self.cbf_attr['scale_factor_timestamp']

        def notify(value: Any, ts: float) -> None:
            # Compute dump index containing this timestamp
            dump_idx = max(0, math.ceil((ts - time_base) / self.cbf_attr['int_time']))
            sh.add(dump_idx, value)
            logger.info('%s updated with timestamp %.3f / index %d', key, ts, dump_idx)

        await self._watch_sensor(telstate, key, notify)

    def _add_sidecar_task(self, name: str, func: Callable[..., Awaitable[None]],
                          *args, **kwargs) -> None:
        decorator = task_wrapper(name, DeviceStatus.DEGRADED, restart=True)
        decorated = decorator(func)
        loop = asyncio.get_event_loop()
        self._sidecar_tasks.append(loop.create_task(decorated(*args, **kwargs)))

    async def _get_data(self) -> None:
        """Receive data. This is called after the metadata has been retrieved."""
        idx = 0
        self.status_sensor.value = Status.WAIT_DATA
        assert self.rx is not None     # keeps mypy happy
        while True:
            try:
                frame = await self.rx.get()
            except spead2.Stopped:
                logger.info('Detected receiver stopped')
                await self.rx.join()
                return

            st = time.time()
            # Configure datasets and other items now that we have complete metadata
            if idx == 0:
                self.status_sensor.value = Status.CAPTURING

            # Generate timestamps
            current_ts_rel = frame.timestamp / self.cbf_attr['scale_factor_timestamp']
            current_ts = self.cbf_attr['sync_time'] + current_ts_rel
            self._my_sensors["last-dump-timestamp"].value = current_ts

            await self._output_avg.add_index(frame.idx)
            await self._sd_avg.add_index(frame.idx)

            proc_a = self.proc_resource.acquire()
            sensor_a = self.sensor_resource.acquire()
            input_a, host_input_a = self.input_resource.acquire()
            # Limit backlog by waiting for previous job to get as far as
            # start to transfer its data before trying to carry on.
            await host_input_a.wait()
            self.jobs.add(
                self._frame_job(proc_a, sensor_a, input_a, host_input_a, frame, current_ts))

            # Done with reading this frame
            idx += 1
            tt = time.time() - st
            logger.debug(
                "Captured CBF frame with timestamp %i (process_time: %.2f, index: %i)",
                current_ts, tt, frame.idx)
            del frame       # Frees memory back to the memory pool
            # Clear completed processing, so that any related exceptions are
            # thrown as soon as possible.
            self.jobs.clean()

    @task_wrapper('CBFIngest session', DeviceStatus.FAIL)
    async def run(self) -> None:
        # Ensure we have clean state. Some of this is unnecessary in normal
        # use, but important if the previous session crashed.
        self._zero_counters()
        await self._output_avg.finish(flush=False)
        await self._sd_avg.finish(flush=False)
        self._init_ig_sd()
        # Send start-of-stream packets.
        await self._send_sd_data(self.ig_sd.get_start())
        for tx in self.tx.values():
            await tx.start()
        # Initialise the input stream
        assert self.capture_block_id is not None
        prefixes = [self.telstate.join(self.capture_block_id, l0_name)
                    for l0_name in self.l0_names]
        telstates = [self.telstate.view(prefix) for prefix in prefixes]
        self.rx = TelstateReceiver(
            self.rx_spead_endpoints, self.rx_spead_ifaddr, self.rx_spead_ibv,
            self.rx_spead_max_streams,
            max_packet_size=self.rx_spead_max_packet_size,
            buffer_size=self.rx_spead_buffer_size,
            channel_range=self.channel_ranges.subscribed,
            cbf_channels=len(self.channel_ranges.cbf),
            sensors=self._my_sensors,
            cbf_attr=self.cbf_attr,
            telstates=telstates,
            l0_int_time=self.cbf_attr['int_time'] * self._output_avg.ratio)
        # If stop() was called before we create self.rx, it won't have been able
        # to call self.rx.stop(), but it will have set _stopped.
        if self._stopped:
            self.rx.stop()

        for name, dtype in [('sdisp_timeseries_mask', np.float32),
                            ('sdisp_custom_signals', np.int32)]:
            self._add_sidecar_task(f'{name} updater', self._update_sdisp_sensor, name, dtype)
        for name in ['channel_mask', 'channel_data_suspect', 'input_data_suspect']:
            self._add_sidecar_task(f'{name} updater', self._update_sensor, self.telstate_cbf, name)

        antennas = set()
        for a, b in self.bls_ordering.sdp_bls_ordering:
            antennas.add(a[:-1])
            antennas.add(b[:-1])
        for antenna in antennas:
            name = f'{antenna}_data_suspect'
            self._add_sidecar_task(f'{name} updater', self._update_sensor, self.telstate, name)

        # The main loop
        await self._get_data()

        logger.info('Joined with receiver. Flushing final groups...')
        await self._output_avg.finish()
        await self._sd_avg.finish()
        logger.info('Waiting for jobs to complete...')
        await self.jobs.finish()
        logger.info('Jobs complete')
        for (name, tx) in self.tx.items():
            logger.info('Stopping %s tx stream...', name)
            await tx.stop()
        for sdisp_tx in self._sdisp_ips.values():
            logger.info('Stopping signal display stream...')
            await self._stop_stream(sdisp_tx, self.ig_sd)
        logger.info("CBF ingest complete")
        self.status_sensor.value = Status.COMPLETE
