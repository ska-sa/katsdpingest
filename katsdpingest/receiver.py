"""Receives from multiple SPEAD streams and combines heaps into frames."""

from __future__ import print_function, absolute_import, division
import spead2
import spead2.recv.trollius
import trollius
from trollius import From, Return
import logging
from collections import deque
from . import utils


_logger = logging.getLogger(__name__)
# CBF SPEAD metadata items that should be stored as sensors rather than
# attributes. Don't use this directly; use :func:`is_cbf_sensor` instead,
# which handles cases that aren't fixed strings.
CBF_SPEAD_SENSORS = frozenset(["flags_xeng_raw"])
# Attributes that are required for data to be correctly ingested
CBF_CRITICAL_ATTRS = frozenset([
    'adc_sample_rate', 'n_chans', 'n_accs', 'n_bls', 'bls_ordering',
    'bandwidth', 'sync_time', 'int_time', 'scale_factor_timestamp'])


def is_cbf_sensor(name):
    return name in CBF_SPEAD_SENSORS or name.startswith('eq_coef_')


class Frame(object):
    """A group of xeng_raw data with a common timestamp"""
    def __init__(self, timestamp, n_streams):
        self.timestamp = timestamp
        self.items = [None] * n_streams

    def ready(self):
        return all(item is not None for item in self.items)

    @property
    def nbytes(self):
        return sum([(item.nbytes if item is not None else 0) for item in self.items])


class Receiver(object):
    """Class that receives from multiple SPEAD streams and combines heaps into
    frames. It also collects CBF metadata from the first stream and uses it to
    populate telescope state.

    Parameters
    ----------
    endpoints : list of :class:`katsdptelstate.Endpoint`
        Endpoints for SPEAD streams. These must be listed in order
        of increasing channel number.
    telstate : :class:`katsdptelstate.TelescopeState`, optional
        Telescope state to be populated with CBF attributes
    cbf_name : str
        Name to prepend to CBF metadata in telstate
    active_frames : int, optional
        Maximum number of incomplete frames to keep at one time
    loop : :class:`trollius.BaseEventLoop`, optional
        I/O loop used for asynchronous operations

    Attributes
    ----------
    telstate : :class:`katsdptelstate.TelescopeState`, optional
        Telescope state passed to constructor
    cbf_attr : dict
        Attributes read from CBF metadata
    cbf_name : str
        Value of `cbf_name` passed to the constructor
    active_frames : int
        Value of `active_frames` passed to constructor
    bandwidth : int
        Total bandwidth across all the streams, or ``None`` if not yet known
    n_chans : int
        Total channels across all the streams, or ``None`` if not yet known
    _interval : int
        Expected timestamp change between successive frames. This is initially ``None``,
        and is computed once the necessary metadata is available.
    _frames : :class:`deque`
        Deque of :class:`Frame` objects representing incomplete frames. After
        initialization, it always contains exactly `active_frames`
        elements, with timestamps separated by the inter-dump interval.
    _frames_complete : :class:`trollius.Queue`
        Queue of complete frames of type :class:`Frame`. It may also contain
        integers, which are the numbers of finished streams.
    _running : int
        Number of streams still running
    _futures : list of :class:`trollius.Future`
        Futures associated with each call to :meth:`_read_stream`
    _streams : list of :class:`spead2.recv.trollius.Stream`
        Individual SPEAD streams
    """
    def __init__(self, endpoints, telstate=None, cbf_name='cbf', active_frames=3, loop=None):
        self.telstate = telstate
        self.cbf_attr = {}
        self.cbf_name = cbf_name
        self.active_frames = active_frames
        self._streams = []
        self._frames = None
        self._frames_complete = trollius.Queue(loop=loop)
        self._futures = []
        self._interval = None
        for i, endpoint in enumerate(endpoints):
            stream = spead2.recv.trollius.Stream(spead2.ThreadPool(), max_heaps=2, ring_heaps=8, loop=loop)
            # This is a quick hack with the maximum size for AR1. Ideally as soon
            # as we have the necessary metadata we should compute the actual size,
            # but _initialise is only called once we've grabbed a heap, and at
            # full speed we can't capture a heap without the memory pool.
            xeng_raw_size = 16 * 17 * 2 * 32768 * 8 // len(endpoints)
            memory_pool = spead2.MemoryPool(xeng_raw_size, xeng_raw_size + 512, 16, 16)
            stream.set_memory_allocator(memory_pool)
            stream.add_udp_reader(endpoint.port, bind_hostname=endpoint.host)
            self._streams.append(stream)
            self._futures.append(trollius.async(self._read_stream(stream, i), loop=loop))
        self._running = len(self._streams)

    @property
    def bandwidth(self):
        try:
            return self.cbf_attr['bandwidth'] * len(self._streams)
        except KeyError:
            return None

    @property
    def n_chans(self):
        try:
            return self.cbf_attr['n_chans'] * len(self._streams)
        except KeyError:
            return None

    def stop(self):
        """Stop all the individual streams and wait to join them."""
        for stream in self._streams:
            stream.stop()

    @trollius.coroutine
    def join(self):
        """Wait for all the individual streams to stop. This must not
        be called concurrently with :meth:`get`.
        """
        while self._running > 0:
            frame = yield From(self._frames_complete.get())
            if isinstance(frame, int):
                yield From(self._futures[frame])
                self._futures[frame] = None
                self._running -= 1

    def _set_telstate_entry(self, name, value, attribute=True):
        utils.set_telstate_entry(self.telstate, name, value,
                                 prefix=self.cbf_name,
                                 attribute=attribute)

    def _update_telstate(self, updated):
        """Updates the telescope state from new values in the item group."""
        for item_name, item in updated.iteritems():
            # bls_ordering is set later by _initialise, after permuting it.
            # The other items are data rather than metadata, and so do not
            # live in the telescope state.
            if item_name not in ['bls_ordering', 'timestamp', 'xeng_raw']:
                # store as an attribute unless item is a sensor (e.g. flags_xeng_raw)
                self._set_telstate_entry(item_name, item.value,
                                         attribute=not is_cbf_sensor(item_name))

    def _update_cbf_attr(self, updated):
        """Updates the internal cbf_attr dictionary from new values in the item group."""
        for item_name, item in updated.iteritems():
            if (item_name not in ['timestamp', 'xeng_raw'] and
                    not is_cbf_sensor(item_name) and
                    item.value is not None):
                if item_name not in self.cbf_attr:
                    self.cbf_attr[item_name] = item.value
                else:
                    _logger.warning('Item %s is already set to %s, not setting to %s',
                                    item_name, self.cbf_attr[item_name], item.value)

    def _pop_frame(self):
        """Remove the oldest element of :attr:`_frames`, and replace it with
        a new frame at the other end.
        """
        # TODO: store timestep in the class
        next_timestamp = self._frames[-1].timestamp + self._interval
        self._frames.popleft()
        self._frames.append(Frame(next_timestamp, len(self._streams)))

    @trollius.coroutine
    def _flush_frames(self):
        """Remove any completed frames from the head of :attr:`_frames`."""
        while self._frames[0].ready():
            # Note: _pop_frame must be done *before* trying to put the
            # item onto the queue, because other coroutines may run and
            # operate on _frames while we're waiting for space in the
            # queue.
            frame = self._frames[0]
            _logger.debug('Flushing frame with timestamp %d', frame.timestamp)
            self._pop_frame()
            yield From(self._frames_complete.put(frame))

    @trollius.coroutine
    def _read_stream(self, stream, stream_idx):
        """Co-routine that sucks data from a single stream and populates
        :attr:`_frames_complete`."""
        try:
            prev_ts = -1
            ts_wrap_offset = 0        # Value added to compensate for CBF timestamp wrapping
            ts_wrap_period = 2**48
            ig_cbf = spead2.ItemGroup()
            while True:
                try:
                    heap = yield From(stream.get())
                except spead2.Stopped:
                    break
                updated = ig_cbf.update(heap)
                if stream_idx == 0:
                    self._update_telstate(updated)
                    self._update_cbf_attr(updated)
                if 'xeng_raw' not in updated:
                    _logger.info(
                        "CBF non-data heap received on stream %d", stream_idx)
                    continue
                if 'timestamp' not in updated:
                    _logger.warning(
                        "CBF heap without timestamp received on stream %d", stream_idx)
                    continue

                data_ts = ig_cbf['timestamp'].value + ts_wrap_offset
                data_item = ig_cbf['xeng_raw'].value
                _logger.info('Received heap with timestamp %d on stream %d', data_ts, stream_idx)
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
                        _logger.warning('Data timestamps wrapped')
                prev_ts = data_ts
                # we have new data...

                # check to see if our CBF attributes are complete
                # i.e. make sure any attributes marked as critical are present
                if not CBF_CRITICAL_ATTRS.issubset(self.cbf_attr.keys()):
                    _logger.warning("CBF Component Model is not currently valid as critical attribute items are missing. Data will be discarded until these become available.")
                    continue

                if self._interval is None:
                    self._interval = int(round(self.cbf_attr['n_chans'] *
                                               self.cbf_attr['n_accs'] *
                                               self.cbf_attr['scale_factor_timestamp'] /
                                               self.cbf_attr['bandwidth']))
                if self._frames is None:
                    self._frames = deque()
                    for i in range(self.active_frames):
                        self._frames.append(Frame(data_ts + self._interval * i, len(self._streams)))
                ts0 = self._frames[0].timestamp
                if data_ts < ts0:
                    _logger.warning('Timestamp %d on stream %d is too far in the past, discarding',
                                        data_ts, stream_idx)
                    continue
                elif (data_ts - ts0) % self._interval != 0:
                    _logger.warning('Timestamp %d on stream %d does not match expected period, discarding',
                                        data_ts, stream_idx)
                    continue
                while data_ts >= ts0 + self._interval * self.active_frames:
                    _logger.warning('Frame with timestamp %d is incomplete, discarding', ts0)
                    self._pop_frame()
                    yield From(self._flush_frames())
                    ts0 = self._frames[0].timestamp
                frame_idx = (data_ts - ts0) // self._interval
                self._frames[frame_idx].items[stream_idx] = data_item
                yield From(self._flush_frames())
        finally:
            yield From(self._frames_complete.put(stream_idx))

    @trollius.coroutine
    def get(self):
        """Return the next frame.

        This is a coroutine.

        Raises
        ------
        spead2.Stopped
            if all the streams have stopped
        """
        while self._running > 0:
            frame = yield From(self._frames_complete.get())
            if isinstance(frame, int):
                # It's actually the index of a finished stream
                self._streams[frame].stop()   # In case the co-routine exited with an exception
                yield From(self._futures[frame])
                self._futures[frame] = None
                self._running -= 1
            else:
                raise Return(frame)
        # Check for frames still in the queue
        while self._frames:
            frame = self._frames[0]
            self._frames.popleft()
            if frame.ready():
                _logger.debug('Flushing frame with timestamp %d', frame.timestamp)
                raise Return(frame)
            else:
                _logger.warning('Frame with timestamp %d is incomplete, discarding', frame.timestamp)
        raise spead2.Stopped('End of streams')
