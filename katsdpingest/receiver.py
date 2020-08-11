"""Receives from multiple SPEAD streams and combines heaps into frames."""

import logging
from collections import deque
import asyncio
import typing   # noqa: F401
from typing import List, Sequence, Mapping, Any, Optional, Union   # noqa: F401

import spead2
import spead2.recv
import spead2.recv.asyncio
from aiokatcp import Sensor

import numpy as np
from katsdptelstate.endpoint import endpoints_to_str, Endpoint

from .utils import Range


_logger = logging.getLogger(__name__)


REJECT_HEAP_TYPES = {
    'incomplete': 'incomplete',
    'no-descriptor': 'descriptors not yet received',
    'bad-timestamp': 'timestamp not aligned to integration boundary',
    'too-old': 'timestamp is prior to the start time',
    'bad-channel': 'channel offset is not aligned to the substreams',
    'missing': 'expected heap was not received',
    'bad-heap': 'heap items are missing, wrong shape etc'
}


class Frame:
    """A group of xeng_raw data with a common timestamp"""
    def __init__(self, idx: int, timestamp: int, n_xengs: int) -> None:
        self.idx = idx
        self.timestamp = timestamp
        self.items = [None] * n_xengs    # type: List[Optional[np.ndarray]]

    def ready(self) -> bool:
        return all(item is not None for item in self.items)

    def empty(self) -> bool:
        return all(item is None for item in self.items)

    @property
    def nbytes(self) -> int:
        return sum([(item.nbytes if item is not None else 0) for item in self.items])


class Receiver:
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
    loop : :class:`asyncio.AbstractEventLoop`, optional
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
    _frames_complete : :class:`asyncio.Queue`
        Queue of complete frames of type :class:`Frame`. It may also contain
        integers, which are the numbers of finished streams.
    _running : int
        Number of streams still running
    _futures : list of :class:`asyncio.Future`
        Futures associated with each call to :meth:`_read_stream`
    _streams : list of :class:`spead2.recv.asyncio.Stream`
        Individual SPEAD streams
    _stopping : bool
        Set to try by stop(). Note that some streams may still be running
        (:attr:`_running` > 0) at the same time.
    """
    def __init__(
            self,
            endpoints: List[Endpoint],
            interface_address: str, ibv: bool,
            max_streams: int, max_packet_size: int, buffer_size: int,
            channel_range: Range, cbf_channels: int,
            sensors: Mapping[str, Sensor],
            cbf_attr: Mapping[str, Any],
            active_frames: int = 1,
            loop: asyncio.AbstractEventLoop = None) -> None:
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
            loop = asyncio.get_event_loop()
        self.cbf_attr = cbf_attr
        self.active_frames = active_frames
        self.channel_range = channel_range
        self.cbf_channels = cbf_channels
        self._interface_address = interface_address
        self._ibv = ibv
        self._streams = []      # type: List[spead2.recv.asyncio.Stream]
        self._frames = deque()  # type: typing.Deque[Frame]
        self._frames_complete = \
            asyncio.Queue(maxsize=1, loop=loop)  # type: asyncio.Queue[Union[Frame, int]]
        self._futures = []      # type: List[Optional[asyncio.Future]]
        self._stopping = False
        self.interval = cbf_attr['ticks_between_spectra'] * cbf_attr['n_accs']
        self.timestamp_base = 0
        self._loop = loop
        self._ig_cbf = spead2.ItemGroup()

        self._input_bytes = sensors['input-bytes-total']
        self._input_bytes.value = 0
        self._input_heaps = sensors['input-heaps-total']
        self._input_heaps.value = 0
        self._input_dumps = sensors['input-dumps-total']
        self._input_dumps.value = 0
        self._descriptors_received = sensors['descriptors-received']
        self._descriptors_received.value = False
        self._metadata_heaps = sensors['input-metadata-heaps-total']
        self._metadata_heaps.value = 0
        self._reject_heaps = {
            name: sensors['input-' + name + '-heaps-total'] for name in REJECT_HEAP_TYPES
        }
        for sensor in self._reject_heaps.values():
            sensor.value = 0

        n_streams = min(max_streams, len(use_endpoints))
        stream_buffer_size = buffer_size // n_streams
        for i in range(n_streams):
            first = len(use_endpoints) * i // n_streams
            last = len(use_endpoints) * (i + 1) // n_streams
            self._streams.append(self._make_stream(use_endpoints[first:last],
                                                   max_packet_size, stream_buffer_size))
            self._futures.append(loop.create_task(
                self._read_stream(self._streams[-1], i, last - first)))
        self._running = n_streams

    def stop(self) -> None:
        """Stop all the individual streams."""
        self._stopping = True
        for stream in self._streams:
            if stream is not None:
                stream.stop()

    async def join(self) -> None:
        """Wait for all the individual streams to stop. This must not
        be called concurrently with :meth:`get`.

        This is a coroutine.
        """
        while self._running > 0:
            frame = await self._frames_complete.get()
            if isinstance(frame, int):
                future = self._futures[frame]
                assert future is not None
                await future
                self._futures[frame] = None
                self._running -= 1

    def _pop_frame(self, replace=True) -> Optional[Frame]:
        """Remove the oldest element of :attr:`_frames`.

        Replace it with a new frame at the other end (unless `replace` is
        false), warn if it is incomplete, and update the missing heaps
        counter.

        Returns
        -------
        frame
            The popped frame, or ``None`` if it was empty
        """
        xengs = len(self._frames[-1].items)
        next_idx = self._frames[-1].idx + 1
        next_timestamp = self._frames[-1].timestamp + self.interval
        frame = self._frames.popleft()
        if replace:
            self._frames.append(Frame(next_idx, next_timestamp, xengs))
        actual = sum(item is not None for item in frame.items)
        self._reject_heaps['missing'].value += xengs - actual
        if actual == 0:
            _logger.debug('Frame with timestamp %d is empty, discarding', frame.timestamp)
            return None
        else:
            _logger.debug('Frame with timestamp %d is %d/%d complete',
                          frame.timestamp, actual, xengs)
        return frame

    async def _put_frame(self, frame: Frame) -> None:
        """Put a frame onto :attr:`_frames_complete` and update the sensor."""
        self._input_dumps.value += 1
        await self._frames_complete.put(frame)

    def _add_readers(self, stream: spead2.recv.asyncio.Stream,
                     endpoints: Sequence[Endpoint],
                     max_packet_size: int, buffer_size: int) -> None:
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

    def _make_stream(self, endpoints: Sequence[Endpoint],
                     max_packet_size: int, buffer_size: int) -> spead2.recv.asyncio.Stream:
        """Prepare a stream, which may combine multiple endpoints."""
        # Figure out how many heaps will have the same timestamp, and set
        # up the stream.
        heap_channels = self.cbf_attr['n_chans_per_substream']
        stream_channels = len(endpoints) * self._endpoint_channels
        baselines = len(self.cbf_attr['bls_ordering'])
        heap_data_size = np.dtype(np.complex64).itemsize * heap_channels * baselines
        stream_xengs = stream_channels // heap_channels
        # It's possible for a heap from each X engine and a descriptor heap
        # per endpoint to all arrive at once.
        ring_heaps = stream_xengs + len(endpoints)
        # Additionally, reordering in the network can cause the end of one dump
        # to overlap with the start of the next, for which we need to allow for
        # an extra stream_xengs.
        max_heaps = ring_heaps + stream_xengs
        # We need space in the memory pool for:
        # - live heaps (max_heaps, plus a newly incoming heap)
        # - ringbuffer heaps
        # - per X-engine:
        #   - heap that has just been popped from the ringbuffer (1)
        #   - active frames
        #   - complete frames queue (1)
        #   - frame being processed by ingest_session (which could be several, depending on
        #     latency of the pipeline, but assume 4 to be on the safe side)
        memory_pool_heaps = ring_heaps + max_heaps + stream_xengs * (self.active_frames + 6)
        stream = spead2.recv.asyncio.Stream(
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

    async def _first_timestamp(self, candidate: int) -> int:
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

    async def _read_stream(self, stream: spead2.recv.asyncio.Stream,
                           stream_idx: int, n_endpoints: int) -> None:
        """Co-routine that sucks data from a single stream and populates
        :attr:`_frames_complete`."""
        try:
            heap_channels = self.cbf_attr['n_chans_per_substream']
            xengs = len(self.channel_range) // heap_channels
            prev_ts = None
            ts_wrap_offset = 0        # Value added to compensate for CBF timestamp wrapping
            ts_wrap_period = 2**48
            n_stop = 0

            async def process_heap(heap):
                """Process one heap and return a classification for it.

                The classification is one of:
                - None (normal)
                - 'stop'
                - 'metadata'
                - a key from REJECT_HEAP_TYPES
                """
                nonlocal prev_ts, ts_wrap_offset, n_stop

                heap_type = None
                data_ts = None
                data_item = None
                channel0 = None

                if heap.is_end_of_stream():
                    self._metadata_heaps.value += 1
                    n_stop += 1
                    _logger.debug("%d/%d endpoints stopped on stream %d",
                                  n_stop, n_endpoints, stream_idx)
                    return 'stop'
                elif isinstance(heap, spead2.recv.IncompleteHeap):
                    heap_type = 'incomplete'
                    _logger.debug('dropped incomplete heap %d (%d/%d bytes of payload)',
                                  heap.cnt, heap.received_length, heap.heap_length)
                    # Attempt to extract the timestamp. We can't use
                    # self._ig_cbf.update because that requires a complete
                    # heap, so this emulates some of its functionality.
                    try:
                        item = self._ig_cbf['timestamp']
                    except KeyError:
                        pass   # We don't have the descriptor for it yet
                    else:
                        for raw_item in heap.get_items():
                            if raw_item.id == item.id:
                                try:
                                    item.set_from_raw(raw_item)
                                    item.version += 1
                                except ValueError:
                                    _logger.warning('Exception updating item from heap',
                                                    exc_info=True)
                                    return 'bad-heap'
                                data_ts = item.value
                                break
                    # Note: no return here. We carry on to process the timestamp
                elif not self._descriptors_received.value and not heap.get_descriptors():
                    _logger.debug('Received non-descriptor heap before descriptors')
                    return 'no-descriptor'
                else:
                    try:
                        # We suppress the conversion to little endian. The data
                        # gets copied later anyway and numpy will do the endian
                        # swapping then without an extraneous copy.
                        updated = self._ig_cbf.update(heap, new_order='|')
                    except ValueError:
                        _logger.warning('Exception updating item group from heap', exc_info=True)
                        return 'bad-heap'
                    # The _ig_cbf is shared between streams, so we need to use the values
                    # before next yielding.
                    if 'timestamp' in updated:
                        data_ts = updated['timestamp'].value
                    if 'xeng_raw' in updated:
                        data_item = updated['xeng_raw'].value
                    if 'frequency' in updated:
                        channel0 = updated['frequency'].value
                    if not self._descriptors_received.value and 'xeng_raw' in self._ig_cbf:
                        # This heap added the descriptors
                        self._descriptors_received.value = True

                if data_ts is None:
                    _logger.debug("Heap without timestamp received on stream %d", stream_idx)
                    return heap_type or 'metadata'

                # Process the timestamp, even if this is an incomplete heap, so
                # that we age out partial frames timeously.
                data_ts += ts_wrap_offset
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
                _logger.debug('Received heap with timestamp %d on stream %d, channel %s',
                              data_ts, stream_idx, channel0)
                prev_ts = data_ts
                if not self._frames:
                    self.timestamp_base = await self._first_timestamp(data_ts)
                    for i in range(self.active_frames):
                        self._frames.append(
                            Frame(i, self.timestamp_base + self.interval * i, xengs))
                ts0 = self._frames[0].timestamp
                if data_ts < ts0:
                    _logger.debug('Timestamp %d is too far in the past, discarding '
                                  '(channel %s)', data_ts, channel0)
                    return heap_type or 'too-old'
                elif (data_ts - ts0) % self.interval != 0:
                    _logger.debug('Timestamp %d does not conform to %d + %dn, '
                                  'discarding (channel %s)',
                                  data_ts, ts0, self.interval, channel0)
                    return heap_type or 'bad-timestamp'
                while data_ts >= ts0 + self.interval * self.active_frames:
                    frame = self._pop_frame()
                    if frame:
                        await self._put_frame(frame)
                    del frame   # Free it up, particularly if discarded
                    ts0 = self._frames[0].timestamp

                if heap_type == 'incomplete':
                    return heap_type

                # From here on we expect we have proper data
                if data_item is None:
                    _logger.warning("CBF heap without xeng_raw received on stream %d", stream_idx)
                    return 'bad-heap'
                if channel0 is None:
                    _logger.warning("CBF heap without frequency received on stream %d", stream_idx)
                    return 'bad-heap'
                heap_channel_range = Range(channel0, channel0 + heap_channels)
                if not (heap_channel_range.isaligned(heap_channels)
                        and heap_channel_range.issubset(self.channel_range)):
                    _logger.debug("CBF heap with invalid channel %d on stream %d",
                                  channel0, stream_idx)
                    return 'bad-channel'
                xeng_idx = (channel0 - self.channel_range.start) // heap_channels
                frame_idx = (data_ts - ts0) // self.interval
                self._frames[frame_idx].items[xeng_idx] = data_item
                self._input_bytes.value += data_item.nbytes
                self._input_heaps.value += 1
                return heap_type

            async for heap in stream:
                heap_type = await process_heap(heap)
                if heap_type == 'stop':
                    if n_stop == n_endpoints:
                        stream.stop()
                        break
                elif heap_type == 'metadata':
                    self._metadata_heaps.value += 1
                elif heap_type in REJECT_HEAP_TYPES:
                    # Don't warn about incomplete heaps if we've already been
                    # asked to stop. There may be some heaps still in the
                    # network at the time we were asked to stop.
                    if heap_type != 'incomplete' or not self._stopping:
                        self._reject_heaps[heap_type].value += 1
                else:
                    assert heap_type is None
        finally:
            await self._frames_complete.put(stream_idx)

    async def get(self) -> Frame:
        """Return the next frame.

        This is a coroutine.

        Raises
        ------
        spead2.Stopped
            if all the streams have stopped
        """
        while self._running > 0:
            frame = await self._frames_complete.get()
            if isinstance(frame, int):
                # It's actually the index of a finished stream
                self._streams[frame].stop()   # In case the co-routine exited with an exception
                future = self._futures[frame]
                assert future is not None
                await future
                self._futures[frame] = None
                self._running -= 1
            else:
                return frame
        # Check for frames still in the queue
        while self._frames:
            tail_frame = self._pop_frame(replace=False)
            if tail_frame:
                return tail_frame
        raise spead2.Stopped('End of streams')
