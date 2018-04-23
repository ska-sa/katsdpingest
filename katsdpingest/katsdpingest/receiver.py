"""Receives from multiple SPEAD streams and combines heaps into frames."""

from __future__ import print_function, absolute_import, division
import logging
from collections import deque

import katcp
import spead2
import spead2.recv
import spead2.recv.trollius
import trollius
from trollius import From, Return
import numpy as np
from katsdptelstate.endpoint import endpoints_to_str

from .utils import Range, SensorWrapper


_logger = logging.getLogger(__name__)


REJECT_HEAP_TYPES = {
    'incomplete': 'incomplete',
    'no-descriptor': 'descriptors not yet received',
    'bad-timestamp': 'timestamp not aligned to integration boundary',
    'too-old': 'timestamp is prior to the start time',
    'bad-channel': 'channel offset is not aligned to the substreams',
    'missing': 'expected heap was not received'
}


def _warn_if_positive(value):
    return katcp.Sensor.WARN if value > 0 else katcp.Sensor.NOMINAL


class Frame(object):
    """A group of xeng_raw data with a common timestamp"""
    def __init__(self, idx, timestamp, n_xengs):
        self.idx = idx
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
    max_streams : int
        Maximum number of separate streams to use. The endpoints are spread
        across the streams, with a thread per stream.
    max_packet_size : int
        Maximum packet size in bytes.
    buffer_size : int
        Buffer size. It is split across the streams.
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
    interval : int
        Timestamp change between successive frames.
    timestamp_base : int
        Timestamp associated with the frame with index 0. It is initially
        ``None``, and is set when the first dump is received. The raw
        timestamp of any other frame can be computed as
        ``timestamp_base + idx * interval``.
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
    _stopping : bool
        Set to try by stop(). Note that some streams may still be running
        (:attr:`_running` > 0) at the same time.
    """
    def __init__(self, endpoints, interface_address, ibv,
                 max_streams, max_packet_size, buffer_size,
                 channel_range, cbf_channels, sensors,
                 cbf_attr, active_frames=4, loop=None):
        # Determine the endpoints to actually use
        if cbf_channels % len(endpoints):
            raise ValueError('cbf_channels not divisible by the number of endpoints')
        self._endpoint_channels = cbf_channels // len(endpoints)
        if not channel_range.isaligned(self._endpoint_channels):
            raise ValueError('channel_range is not aligned to the stream boundaries')
        if self._endpoint_channels % cbf_attr['n_chans_per_substream'] != 0:
            raise ValueError('Number of channels in substream does not divide '
                             'into per-endpoint channels')
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
        self._stopping = False
        self.interval = cbf_attr['ticks_between_spectra'] * cbf_attr['n_accs']
        self.timestamp_base = 0
        self._loop = loop
        self._ig_cbf = spead2.ItemGroup()

        self._input_bytes = SensorWrapper(sensors['input-bytes-total'], 0)
        self._input_heaps = SensorWrapper(sensors['input-heaps-total'], 0)
        self._input_dumps = SensorWrapper(sensors['input-dumps-total'], 0)
        self._descriptors_received = SensorWrapper(sensors['descriptors-received'], False)
        self._reject_heaps = {name: SensorWrapper(sensors['input-' + name + '-heaps-total'], 0,
                                                  _warn_if_positive)
                              for name in REJECT_HEAP_TYPES}

        n_streams = min(max_streams, len(use_endpoints))
        stream_buffer_size = buffer_size // n_streams
        for i in range(n_streams):
            first = len(use_endpoints) * i // n_streams
            last = len(use_endpoints) * (i + 1) // n_streams
            self._streams.append(self._make_stream(use_endpoints[first:last],
                                                   max_packet_size, stream_buffer_size))
            self._futures.append(trollius.async(
                self._read_stream(self._streams[-1], i, last - first), loop=loop))
        self._running = n_streams

    def stop(self):
        """Stop all the individual streams."""
        self._stopping = True
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
        next_idx = self._frames[-1].idx + 1
        next_timestamp = self._frames[-1].timestamp + self.interval
        self._frames.popleft()
        self._frames.append(Frame(next_idx, next_timestamp, xengs))

    @trollius.coroutine
    def _put_frame(self, frame):
        """Put a frame onto :attr:`_frames_complete` and update the sensor."""
        self._input_dumps.value += 1
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

    def _add_readers(self, stream, endpoints, max_packet_size, buffer_size):
        """Subscribe a stream to a list of endpoints."""
        ifaddr = self._interface_address
        if self._ibv:
            if ifaddr is None:
                raise ValueError('Cannot use ibverbs without an interface address')
            endpoint_tuples = [(endpoint.host, endpoint.port) for endpoint in endpoints]
            stream.add_udp_ibv_reader(endpoint_tuples, ifaddr,
                                      max_size=max_packet_size, buffer_size=buffer_size)
        else:
            for endpoint in endpoints:
                if ifaddr is None:
                    stream.add_udp_reader(endpoint.port, bind_hostname=endpoint.host)
                else:
                    stream.add_udp_reader(endpoint.host, endpoint.port,
                                          interface_address=ifaddr)
        _logger.info(
            "CBF SPEAD stream reception on %s via %s%s",
            endpoints_to_str(endpoints),
            ifaddr if ifaddr is not None else 'default interface',
            ' with ibv' if self._ibv else '')

    def _make_stream(self, endpoints, max_packet_size, buffer_size):
        """Prepare a stream, which may combine multiple endpoints."""
        # Figure out how many heaps will have the same timestamp, and set
        # up the stream.
        heap_channels = self.cbf_attr['n_chans_per_substream']
        stream_channels = len(endpoints) * self._endpoint_channels
        baselines = len(self.cbf_attr['bls_ordering'])
        heap_data_size = np.dtype(np.complex64).itemsize * heap_channels * baselines
        stream_xengs = stream_channels // heap_channels
        # It's possible for a heap from each X engine and a descriptor heap
        # per endpoint to all arrive at once. We assume that each xengine will
        # not overlap packets between heaps, and that there is enough of a gap
        # between heaps that reordering in the network is a non-issue.
        ring_heaps = stream_xengs + len(endpoints)
        max_heaps = ring_heaps
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
            ring_heaps=ring_heaps, loop=self._loop,
            contiguous_only=False)
        memory_pool = spead2.MemoryPool(16384, heap_data_size + 512,
                                        memory_pool_heaps, memory_pool_heaps)
        stream.set_memory_allocator(memory_pool)
        stream.set_memcpy(spead2.MEMCPY_NONTEMPORAL)
        stream.stop_on_stop_item = False
        self._add_readers(stream, endpoints, max_packet_size, buffer_size)
        return stream

    def _first_timestamp(self, candidate):
        """Get raw ADC timestamp of the first frame across all ingests.

        This is called when the first valid dump is received for this
        receiver, and returns the raw timestamp of the first valid dump
        across all receivers. Note that the return value may be greater
        than `candidate` if another receiver received a heap first but with
        a larger timestamp.

        In the base implementation, it simply returns `candidate`. Subclasses
        may override this to implement inter-receiver communication.
        """
        return candidate

    @trollius.coroutine
    def _read_stream(self, stream, stream_idx, n_endpoints):
        """Co-routine that sucks data from a single stream and populates
        :attr:`_frames_complete`."""
        try:
            heap_channels = self.cbf_attr['n_chans_per_substream']
            xengs = len(self.channel_range) // heap_channels
            prev_ts = None
            ts_wrap_offset = 0        # Value added to compensate for CBF timestamp wrapping
            ts_wrap_period = 2**48
            n_stop = 0
            while True:
                try:
                    heap = yield From(stream.get())
                except spead2.Stopped:
                    break
                if heap.is_end_of_stream():
                    n_stop += 1
                    _logger.debug("%d/%d endpoints stopped on stream %d",
                                  n_stop, n_endpoints, stream_idx)
                    if n_stop == n_endpoints:
                        stream.stop()
                        break
                    else:
                        continue
                elif isinstance(heap, spead2.recv.IncompleteHeap):
                    # Don't warn if we've already been asked to stop. There may
                    # be some heaps still in the network at the time we were
                    # asked to stop.
                    if not self._stopping:
                        _logger.debug('dropped incomplete heap %d (%d/%d bytes of payload)',
                                      heap.cnt, heap.received_length, heap.heap_length)
                        self._reject_heaps['incomplete'].value += 1
                    continue
                elif not self._descriptors_received.value and not heap.get_descriptors():
                    _logger.debug('Received non-descriptor heap before descriptors')
                    self._reject_heaps['no-descriptor'].value += 1
                    continue
                updated = self._ig_cbf.update(heap)
                # The _ig_cbf is shared between streams, so we need to use the values
                # before next yielding.
                if not self._descriptors_received.value and 'xeng_raw' in self._ig_cbf:
                    # This heap added the descriptors
                    self._descriptors_received.value = True
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
                    _logger.debug("CBF heap with invalid channel %d on stream %d",
                                  channel0, stream_idx)
                    self._reject_heaps['bad-channel'].value += 1
                    continue
                xeng_idx = (channel0 - self.channel_range.start) // heap_channels

                data_ts = self._ig_cbf['timestamp'].value + ts_wrap_offset
                data_item = self._ig_cbf['xeng_raw'].value
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
                _logger.debug('Received heap with timestamp %d on stream %d, channel %d',
                              data_ts, stream_idx, channel0)
                prev_ts = data_ts
                # we have new data...

                if self._frames is None:
                    self.timestamp_base = self._first_timestamp(data_ts)
                    self._frames = deque()
                    for i in range(self.active_frames):
                        self._frames.append(
                            Frame(i, self.timestamp_base + self.interval * i, xengs))
                ts0 = self._frames[0].timestamp
                if data_ts < ts0:
                    _logger.debug('Timestamp %d is too far in the past, discarding '
                                  '(frequency %d)', data_ts, channel0)
                    self._reject_heaps['too-old'].value += 1
                    continue
                elif (data_ts - ts0) % self.interval != 0:
                    _logger.debug('Timestamp %d does not conform to %d + %dn, '
                                  'discarding (frequency %d)',
                                  data_ts, ts0, self.interval, channel0)
                    self._reject_heaps['bad-timestamp'].value += 1
                    continue
                while data_ts >= ts0 + self.interval * self.active_frames:
                    frame = self._frames[0]
                    self._pop_frame()
                    expected = len(frame.items)
                    actual = sum(item is not None for item in frame.items)
                    if actual == 0:
                        _logger.debug('Frame with timestamp %d is empty, discarding', ts0)
                    else:
                        _logger.debug('Frame with timestamp %d is %d/%d complete', ts0,
                                      actual, expected)
                        yield From(self._put_frame(frame))
                    self._reject_heaps['missing'].value += expected - actual
                    del frame   # Free it up, particularly if discarded
                    yield From(self._flush_frames())
                    ts0 = self._frames[0].timestamp
                frame_idx = (data_ts - ts0) // self.interval
                self._frames[frame_idx].items[xeng_idx] = data_item
                self._input_bytes.value += data_item.nbytes
                self._input_heaps.value += 1
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
                _logger.warning('Frame with timestamp %d is incomplete, discarding',
                                frame.timestamp)
        raise spead2.Stopped('End of streams')
