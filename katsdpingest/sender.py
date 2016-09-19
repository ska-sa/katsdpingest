"""Helper classes encapsulating the details of sending SPEAD streams."""

from __future__ import print_function, division, absolute_import
import numpy as np
import spead2.send.trollius
import logging
import trollius
from trollius import From
from .utils import Range


_logger = logging.getLogger(__name__)


class Data(object):
    """Bundles visibilities, flags and weights"""
    def __init__(self, vis=None, flags=None, weights=None, weights_channel=None):
        self.vis = vis
        self.flags = flags
        self.weights = weights
        self.weights_channel = weights_channel

    def __getitem__(self, idx):
        """Do numpy slicing on all fields at once"""
        return Data(self.vis[idx], self.flags[idx],
                    self.weights[idx], self.weights_channel[idx])

    @property
    def nbytes(self):
        return (self.vis.nbytes + self.flags.nbytes +
                self.weights.nbytes + self.weights_channel.nbytes)


@trollius.coroutine
def async_send_heap(stream, heap):
    """Send a heap on a stream and wait for it to complete, but log and
    suppress exceptions."""
    try:
        yield From(stream.async_send_heap(heap))
    except Exception:
        _logger.warn("Error sending heap", exc_info=True)


class VisSender(object):
    """A single output SPEAD stream of L0 visibility data.

    Parameters
    ----------
    thread_pool : `spead2.ThreadPool`
        Thread pool servicing the stream
    endpoint : `katsdptelstate.endpoint.Endpoint`
        Stream endpoint
    flavour : `spead2.Flavour`
        SPEAD flavour to use on `stream`
    int_time : float
        Time between dumps, in seconds
    channel_range : :class:`katsdpingest.utils.Range`
        Range of channel numbers to be placed into this stream (of those passed to :meth:`send`)
    baselines : number of baselines in output
    """
    def __init__(self, thread_pool, endpoint, flavour, int_time, channel_range, baselines):
        channels = len(channel_range)
        dump_size = channels * baselines * (np.dtype(np.complex64).itemsize + 2 * np.dtype(np.uint8).itemsize)
        dump_size += channels * np.dtype(np.float32).itemsize
        # Scaling by 1.1 is to account for network overheads and to allow
        # catchup if we temporarily fall behind the rate.
        rate = dump_size / int_time * 1.1
        self._stream = spead2.send.trollius.UdpStream(
            thread_pool, endpoint.host, endpoint.port,
            spead2.send.StreamConfig(max_packet_size=8972, rate=rate))
        self._ig = spead2.send.ItemGroup(descriptor_frequency=1, flavour=flavour)
        self._channel_range = channel_range
        self._ig.add_item(id=None, name='correlator_data', description="Visibilities",
                          shape=(channels, baselines), dtype=np.complex64)
        self._ig.add_item(id=None, name='flags', description="Flags for visibilities",
                          shape=(channels, baselines), dtype=np.uint8)
        self._ig.add_item(id=None, name='weights', description="Detailed weights, to be scaled by weights_channel",
                          shape=(channels, baselines), dtype=np.uint8)
        self._ig.add_item(id=None, name='weights_channel', description="Coarse (per-channel) weights",
                          shape=(channels,), dtype=np.float32)
        self._ig.add_item(id=None, name='timestamp', description="Seconds since sync time",
                          shape=(), dtype=None, format=[('f', 64)])

    @trollius.coroutine
    def start(self):
        """Send a start packet to the stream."""
        yield From(async_send_heap(self._stream, self._ig.get_start()))

    @trollius.coroutine
    def stop(self):
        """Send a stop packet to the stream. To ensure that it won't be lost
        on the sending side, the stream is first flushed, then the stop
        heap is sent and waited for."""
        yield From(self._stream.async_flush())
        yield From(self._stream.async_send_heap(self._ig.get_end()))

    @trollius.coroutine
    def send(self, data, ts_rel):
        """Asynchronously send visibilities to the receiver, returning a
        future."""
        data = data[self._channel_range.asslice()]
        self._ig['correlator_data'].value = data.vis
        self._ig['flags'].value = data.flags
        self._ig['weights'].value = data.weights
        self._ig['weights_channel'].value = data.weights_channel
        self._ig['timestamp'].value = ts_rel
        return trollius.async(async_send_heap(self._stream, self._ig.get_heap()))


class VisSenderSet(object):
    """Manages a collection of :class:`VisSender` objects, and provides similar
    functions that work collectively on all the streams.
    """
    def __init__(self, thread_pool, endpoints, flavour, int_time, channel_range, baselines):
        channels = len(channel_range)
        n = len(endpoints)
        if channels % n != 0:
            raise ValueError('Number of channels not evenly divisible by number of endpoints')
        sub_channels = channels // n
        self._senders = []
        for i in range(n):
            a = channel_range.start + i * sub_channels
            b = a + sub_channels
            self._senders.append(
                VisSender(thread_pool, endpoints[i], flavour, int_time, Range(a, b), baselines))

    @trollius.coroutine
    def start(self):
        """Send a start heap to all streams."""
        return trollius.gather(*(trollius.async(sender.start()) for sender in self._senders))

    @trollius.coroutine
    def stop(self):
        """Send a stop heap to all streams."""
        return trollius.gather(*(trollius.async(sender.stop()) for sender in self._senders))

    @trollius.coroutine
    def send(self, data, ts_rel):
        """Send a data heap to all streams, splitting the data between them."""
        return trollius.gather(*(trollius.async(sender.send(data, ts_rel))
                                 for sender in self._senders))
