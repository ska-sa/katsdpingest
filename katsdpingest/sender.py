"""Helper classes encapsulating the details of sending SPEAD streams."""

import logging
import asyncio
from typing import List, Dict, Sequence, Any   # noqa: F401

import numpy as np
from katsdptelstate.endpoint import Endpoint
import spead2.send.asyncio

from .utils import Range


_logger = logging.getLogger(__name__)


class Data:
    """Bundles visibilities, flags and weights"""
    def __init__(self,
                 vis: np.ndarray,
                 flags: np.ndarray,
                 weights: np.ndarray,
                 weights_channel: np.ndarray) -> None:
        self.vis = vis
        self.flags = flags
        self.weights = weights
        self.weights_channel = weights_channel

    def __getitem__(self, idx) -> 'Data':
        """Do numpy slicing on all fields at once"""
        return Data(self.vis[idx], self.flags[idx],
                    self.weights[idx], self.weights_channel[idx])

    @property
    def nbytes(self) -> int:
        return (self.vis.nbytes + self.flags.nbytes
                + self.weights.nbytes + self.weights_channel.nbytes)


async def async_send_heap(stream: spead2.send.asyncio.UdpStream,
                          heap: spead2.send.Heap) -> None:
    """Send a heap on a stream and wait for it to complete, but log and
    suppress exceptions."""
    try:
        await stream.async_send_heap(heap)
    except Exception:
        _logger.warn("Error sending heap", exc_info=True)


class VisSender:
    """A single output SPEAD stream of L0 visibility data.

    Parameters
    ----------
    thread_pool : `spead2.ThreadPool`
        Thread pool servicing the stream
    endpoint : `katsdptelstate.endpoint.Endpoint`
        Stream endpoint
    interface_address : str
        IP address of network interface to use, or ``None``
    flavour : `spead2.Flavour`
        SPEAD flavour to use on `stream`
    int_time : float
        Time between dumps, in seconds of wall clock time (which may be
        different to data timestamp time if ``--clock-ratio`` is used).
    channel_range : :class:`katsdpingest.utils.Range`
        Range of channel numbers to be placed into this stream (of those passed to :meth:`send`)
    channel0 : int
        Index of first channel, within the full bandwidth of the L0 output
    all_channels : int
        Number of channels in the full L0 output
    baselines : int
        number of baselines in output
    """
    def __init__(self, thread_pool: spead2.ThreadPool,
                 endpoint: Endpoint, interface_address: str,
                 flavour: spead2.Flavour,
                 int_time: float, channel_range: Range,
                 channel0: int, all_channels: int, baselines: int) -> None:
        channels = len(channel_range)
        item_size = np.dtype(np.complex64).itemsize + 2 * np.dtype(np.uint8).itemsize
        dump_size = channels * baselines * item_size
        dump_size += channels * np.dtype(np.float32).itemsize
        # Add a guess for SPEAD protocol overhead (including descriptors). This just needs
        # to be conservative, to make sure we don't try to send too slow.
        dump_size += 2048
        # Send slightly faster to allow for other network overheads (e.g. overhead per
        # packet, which is a fraction of total size) and to allow us to catch
        # up if we temporarily fall behind the rate.
        rate = dump_size / int_time * 1.05 if int_time else 0.0
        kwargs = {}      # type: Dict[str, Any]
        if interface_address is not None:
            kwargs['interface_address'] = interface_address
            kwargs['ttl'] = 1
        self._stream = spead2.send.asyncio.UdpStream(
            thread_pool, endpoint.host, endpoint.port,
            spead2.send.StreamConfig(max_packet_size=8872, rate=rate), **kwargs)
        self._stream.set_cnt_sequence(channel0, all_channels)
        self._ig = spead2.send.ItemGroup(descriptor_frequency=1, flavour=flavour)
        self._channel_range = channel_range
        self._channel0 = channel0
        self._ig.add_item(id=None, name='correlator_data',
                          description="Visibilities",
                          shape=(channels, baselines), dtype=np.complex64)
        self._ig.add_item(id=None, name='flags',
                          description="Flags for visibilities",
                          shape=(channels, baselines), dtype=np.uint8)
        self._ig.add_item(id=None, name='weights',
                          description="Detailed weights, to be scaled by weights_channel",
                          shape=(channels, baselines), dtype=np.uint8)
        self._ig.add_item(id=None, name='weights_channel',
                          description="Coarse (per-channel) weights",
                          shape=(channels,), dtype=np.float32)
        self._ig.add_item(id=None, name='timestamp',
                          description="Seconds since CBF sync time",
                          shape=(), dtype=None, format=[('f', 64)])
        self._ig.add_item(id=None, name='dump_index',
                          description='Index in time',
                          shape=(), dtype=None, format=[('u', 64)])
        self._ig.add_item(id=0x4103, name='frequency',
                          description="Channel index of first channel in the heap",
                          shape=(), dtype=np.uint32)

    async def start(self):
        """Send a start packet to the stream."""
        await async_send_heap(self._stream, self._ig.get_start())

    async def stop(self):
        """Send a stop packet to the stream. To ensure that it won't be lost
        on the sending side, the stream is first flushed, then the stop
        heap is sent and waited for."""
        await self._stream.async_flush()
        await self._stream.async_send_heap(self._ig.get_end())

    async def send(self, data, idx, ts_rel):
        """Asynchronously send visibilities to the receiver"""
        data = data[self._channel_range.asslice()]
        self._ig['correlator_data'].value = data.vis
        self._ig['flags'].value = data.flags
        self._ig['weights'].value = data.weights
        self._ig['weights_channel'].value = data.weights_channel
        self._ig['timestamp'].value = ts_rel
        self._ig['dump_index'].value = idx
        self._ig['frequency'].value = self._channel0
        await async_send_heap(self._stream, self._ig.get_heap())


class VisSenderSet:
    """Manages a collection of :class:`VisSender` objects, and provides similar
    functions that work collectively on all the streams.
    """
    def __init__(self,
                 thread_pool: spead2.ThreadPool,
                 endpoints: Sequence[Endpoint],
                 interface_address: str,
                 flavour: spead2.Flavour,
                 int_time: float,
                 channel_range: Range,
                 channel0: int, all_channels: int, baselines: int) -> None:
        channels = len(channel_range)
        n = len(endpoints)
        if channels % n != 0:
            raise ValueError('Number of channels not evenly divisible by number of endpoints')
        sub_channels = channels // n
        self.sub_channels = sub_channels
        self._senders = []     # type: List[VisSender]
        for i in range(n):
            a = channel_range.start + i * sub_channels
            b = a + sub_channels
            self._senders.append(
                VisSender(thread_pool, endpoints[i], interface_address, flavour, int_time,
                          Range(a, b), channel0 + i * sub_channels, all_channels, baselines))

    @property
    def size(self) -> int:
        return len(self._senders)

    async def start(self) -> None:
        """Send a start heap to all streams."""
        await asyncio.gather(*(sender.start() for sender in self._senders))

    async def stop(self) -> None:
        """Send a stop heap to all streams."""
        await asyncio.gather(*(sender.stop() for sender in self._senders))

    async def send(self, data: Data, idx: int, ts_rel: float) -> None:
        """Send a data heap to all streams, splitting the data between them."""
        await asyncio.gather(*(sender.send(data, idx, ts_rel)
                               for sender in self._senders))
