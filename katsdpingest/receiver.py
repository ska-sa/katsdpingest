"""Receives from multiple SPEAD streams and combines heaps into frames."""

from __future__ import print_function, absolute_import, division
import logging
from collections import deque
import spead2
import spead2.recv.trollius
import trollius
from trollius import From, Return
import numpy as np
from .utils import Range


_logger = logging.getLogger(__name__)


class Frame(object):
    """A group of xeng_raw data with a common timestamp"""
    def __init__(self, timestamp, n_xengs):
        self.timestamp = timestamp
        self.items = [None] * n_xengs

    def ready(self):
        return all(item is not None for item in self.items)

    def empty(self):
        return all(item is None for item in self.items)

    @property
    def nbytes(self):
        return sum([(item.nbytes if item is not None else 0) for item in self.items])


class Receiver(object):
    """Class that receives from multiple SPEAD streams and combines heaps into
    frames.

    Parameters
    ----------
    endpoints : list of :class:`katsdptelstate.Endpoint`
        Endpoints for SPEAD streams. These must be listed in order
        of increasing channel number.
    interface_address : str
        Address of interface to subscribe to for endpoints
    ibv : bool
        If true, use ibverbs for acceleration
    channel_range : :class:`katsdpingest.utils.Range`
        Channels to capture. These must be aligned to the stream boundaries.
    cbf_channels : int
        Total number of channels represented by `endpoints`.
    sensors : dict
        Dictionary mapping sensor names to sensor objects
    cbf_attr : dict
        Dictionary mapping CBF attribute names to value
    active_frames : int, optional
        Maximum number of incomplete frames to keep at one time
    loop : :class:`trollius.BaseEventLoop`, optional
        I/O loop used for asynchronous operations

    Attributes
    ----------
    cbf_attr : dict
        Dictionary mapping CBF attribute names to value
    active_frames : int
        Value of `active_frames` passed to constructor
    _interval : int
        Expected timestamp change between successive frames.
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
    def __init__(self, endpoints, interface_address, ibv, channel_range, cbf_channels, sensors,
                 cbf_attr, active_frames=2, loop=None):
        # Determine the endpoints to actually use
        if cbf_channels % len(endpoints):
            raise ValueError('cbf_channels not divisible by the number of endpoints')
        self._endpoint_channels = cbf_channels // len(endpoints)
        if not channel_range.isaligned(self._endpoint_channels):
            raise ValueError('channel_range is not aligned to the stream boundaries')
        if self._endpoint_channels % cbf_attr['n_chans_per_substream'] != 0:
            raise ValueError('Number of channels in substream does not divide into per-endpoint channels')
        use_endpoints = endpoints[channel_range.start // self._endpoint_channels :
                                  channel_range.stop // self._endpoint_channels]

        if loop is None:
            loop = trollius.get_event_loop()
        self.cbf_attr = cbf_attr
        self.active_frames = active_frames
        self.channel_range = channel_range
        self.cbf_channels = cbf_channels
        self._interface_address = interface_address
        self._ibv = ibv
        self._streams = []
        self._frames = None
        self._frames_complete = trollius.Queue(maxsize=1, loop=loop)
        self._futures = []
        self._interval = cbf_attr['ticks_between_spectra'] * cbf_attr['n_accs']
        self._loop = loop
        self._input_bytes = 0
        self._input_heaps = 0
        self._input_dumps = 0
        self._input_bytes_sensor = sensors['input-bytes-total']
        self._input_bytes_sensor.set_value(0)
        self._input_heaps_sensor = sensors['input-heaps-total']
        self._input_heaps_sensor.set_value(0)
        self._input_dumps_sensor = sensors['input-dumps-total']
        self._input_dumps_sensor.set_value(0)
        # 2 threads (with a stream per thread) should be enough to keep up, and
        # more risks excessive context switching. endpoints are distributed
        # amongst the streams.
        n_streams = min(2, len(use_endpoints))
        for i in range(n_streams):
            first = len(use_endpoints) * i // n_streams
            last = len(use_endpoints) * (i + 1) // n_streams
            self._streams.append(self._make_stream(use_endpoints[first:last]))
            self._futures.append(trollius.async(
                self._read_stream(self._streams[-1], i), loop=loop))
        self._running = n_streams

    def stop(self):
        """Stop all the individual streams."""
        for stream in self._streams:
            if stream is not None:
                stream.stop()

    @trollius.coroutine
    def join(self):
        """Wait for all the individual streams to stop. This must not
        be called concurrently with :meth:`get`.

        This is a coroutine.
        """
        while self._running > 0:
            frame = yield From(self._frames_complete.get())
            if isinstance(frame, int):
                yield From(self._futures[frame])
                self._futures[frame] = None
                self._running -= 1

    def _pop_frame(self):
        """Remove the oldest element of :attr:`_frames`, and replace it with
        a new frame at the other end.
        """
        xengs = len(self._frames[-1].items)
        next_timestamp = self._frames[-1].timestamp + self._interval
        self._frames.popleft()
        self._frames.append(Frame(next_timestamp, xengs))

    @trollius.coroutine
    def _put_frame(self, frame):
        """Put a frame onto :attr:`_frames_complete` and update the sensor."""
        self._input_dumps += 1
        self._input_dumps_sensor.set_value(self._input_dumps)
        yield From(self._frames_complete.put(frame))

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
            yield From(self._put_frame(frame))

    def _add_readers(self, stream, endpoints):
        """Subscribe a stream to a list of endpoints."""
        ifaddr = self._interface_address
        if self._ibv:
            if ifaddr is None:
                raise ValueError('Cannot use ibverbs without an interface address')
            endpoint_tuples = [(endpoint.host, endpoint.port) for endpoint in endpoints]
            stream.add_udp_ibv_reader(endpoint_tuples, ifaddr,
                                      buffer_size=64 * 1024**2)
        else:
            for endpoint in endpoints:
                if ifaddr is None:
                    stream.add_udp_reader(endpoint.port, bind_hostname=endpoint.host)
                else:
                    stream.add_udp_reader(endpoint.host, endpoint.port,
                                          interface_address=ifaddr)
        _logger.info(
            "CBF SPEAD stream reception on %s via %s%s",
            [str(x) for x in endpoints],
            ifaddr if ifaddr is not None else 'default interface',
            ' with ibv' if self._ibv else '')

    def _make_stream(self, endpoints):
        """Prepare a stream, which may combine multiple endpoints."""
        # Figure out how many heaps will have the same timestamp, and set
        # up the stream.
        heap_channels = self.cbf_attr['n_chans_per_substream']
        stream_channels = len(endpoints) * self._endpoint_channels
        baselines = len(self.cbf_attr['bls_ordering'])
        heap_data_size = np.dtype(np.complex64).itemsize * heap_channels * baselines
        stream_xengs = stream_channels // heap_channels
        ring_heaps = stream_xengs
        # CBF currently sends 2 metadata heaps in a row, hence the + 2
        # We assume that each xengine will not overlap packets between
        # heaps, and that there is enough of a gap between heaps that
        # reordering in the network is a non-issue.
        max_heaps = stream_xengs + 2 * len(endpoints)
        # We need space in the memory pool for:
        # - live heaps (max_heaps, plus a newly incoming heap)
        # - ringbuffer heaps
        # - per X-engine:
        #   - heap that has just been popped from the ringbuffer (1)
        #   - active frames
        #   - complete frames queue (1)
        #   - frame being processed by ingest_session (which could be several, depending on
        #     latency of the pipeline, but assume 3 to be on the safe side)
        memory_pool_heaps = ring_heaps + max_heaps + stream_xengs * (self.active_frames + 5)
        stream = spead2.recv.trollius.Stream(
            spead2.ThreadPool(),
            max_heaps=max_heaps,
            ring_heaps=ring_heaps, loop=self._loop)
        memory_pool = spead2.MemoryPool(16384, heap_data_size + 512,
                                        memory_pool_heaps, memory_pool_heaps)
        stream.set_memory_allocator(memory_pool)
        stream.set_memcpy(spead2.MEMCPY_NONTEMPORAL)
        self._add_readers(stream, endpoints)
        return stream

    @trollius.coroutine
    def _read_stream(self, stream, stream_idx):
        """Co-routine that sucks data from a single stream and populates
        :attr:`_frames_complete`."""
        try:
            heap_channels = self.cbf_attr['n_chans_per_substream']
            xengs = len(self.channel_range) // heap_channels
            prev_ts = None
            ts_wrap_offset = 0        # Value added to compensate for CBF timestamp wrapping
            ts_wrap_period = 2**48
            ig_cbf = spead2.ItemGroup()
            while True:
                try:
                    heap = yield From(stream.get())
                except spead2.Stopped:
                    break
                updated = ig_cbf.update(heap)
                if 'xeng_raw' not in updated:
                    _logger.debug("CBF non-data heap received on stream %d", stream_idx)
                    continue
                if 'timestamp' not in updated:
                    _logger.warning("CBF heap without timestamp received on stream %d", stream_idx)
                    continue
                if 'frequency' not in updated:
                    _logger.warning("CBF heap without frequency received on stream %d", stream_idx)
                    continue
                channel0 = updated['frequency'].value
                heap_channel_range = Range(channel0, channel0 + heap_channels)
                if not (heap_channel_range.isaligned(heap_channels) and
                        heap_channel_range.issubset(self.channel_range)):
                    _logger.warning("CBF heap with invalid channel %d on stream %d", channel0, stream_idx)
                    continue
                xeng_idx = (channel0 - self.channel_range.start) // heap_channels

                data_ts = ig_cbf['timestamp'].value + ts_wrap_offset
                data_item = ig_cbf['xeng_raw'].value
                if prev_ts is not None and data_ts < prev_ts - ts_wrap_period // 2:
                    # This happens either because packets ended up out-of-order,
                    # or because the CBF timestamp wrapped. Out-of-order should
                    # jump backwards a tiny amount while wraps should jump back by
                    # close to ts_wrap_period.
                    ts_wrap_offset += ts_wrap_period
                    data_ts += ts_wrap_period
                    _logger.warning('Data timestamps wrapped')
                elif prev_ts is not None and data_ts > prev_ts + ts_wrap_period // 2:
                    # This happens if we wrapped, then received another heap
                    # (probably from a different X engine) from before the
                    # wrap. We need to undo the wrap.
                    ts_wrap_offset -= ts_wrap_period
                    data_ts -= ts_wrap_period
                    _logger.warning('Data timestamps reverse wrapped')
                _logger.debug('Received heap with timestamp %d on stream %d, channel %d', data_ts, stream_idx, channel0)
                prev_ts = data_ts
                # we have new data...

                self._input_bytes += data_item.nbytes
                self._input_heaps += 1
                self._input_bytes_sensor.set_value(self._input_bytes)
                self._input_heaps_sensor.set_value(self._input_heaps)
                if self._frames is None:
                    self._frames = deque()
                    for i in range(self.active_frames):
                        self._frames.append(Frame(data_ts + self._interval * i, xengs))
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
                    frame = self._frames[0]
                    self._pop_frame()
                    if frame.empty():
                        _logger.warning('Frame with timestamp %d is empty, discarding', ts0)
                    else:
                        expected = len(frame.items)
                        actual = sum(item is not None for item in frame.items)
                        _logger.warning('Frame with timestamp %d is %d/%d complete', ts0,
                                        actual, expected)
                        yield From(self._put_frame(frame))
                    del frame   # Free it up, particularly if discarded
                    yield From(self._flush_frames())
                    ts0 = self._frames[0].timestamp
                frame_idx = (data_ts - ts0) // self._interval
                self._frames[frame_idx].items[xeng_idx] = data_item
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
            elif not frame.empty():
                _logger.warning('Frame with timestamp %d is incomplete, discarding', frame.timestamp)
        raise spead2.Stopped('End of streams')
