"""Tests for receiver module"""

import logging
from unittest import mock
import asyncio

import numpy as np
import spead2
import spead2.send
import spead2.recv.asyncio
import asynctest
import async_timeout

from katsdpingest.receiver import Receiver
from katsdpingest.sigproc import Range
from katsdpingest.test.test_ingest_session import fake_cbf_attr
import katsdptelstate.endpoint
from nose.tools import assert_equal, assert_is_none, assert_raises


class QueueStream:
    """A simulated SPEAD stream, stored in memory as a queue of heaps. A value
    of None indicates that the stream has been shut down (because putting an
    actual stop heap in the queue won't have the desired effect, as normally
    this is processed at a lower level).
    """
    _streams = {}

    def __init__(self, loop=None):
        self._queue = asyncio.Queue(loop=loop)

    async def get(self):
        heap = await self._queue.get()
        if heap is None:
            self._queue.put_nowait(None)
            raise spead2.Stopped()
        else:
            return heap

    def stop(self):
        self._queue.put_nowait(None)

    def send_heap(self, heap):
        tp = spead2.ThreadPool()
        encoder = spead2.send.BytesStream(tp)
        encoder.send_heap(heap)
        raw = encoder.getvalue()
        decoder = spead2.recv.Stream(tp)
        decoder.stop_on_stop_item = False
        decoder.add_buffer_reader(raw)
        heap = decoder.get()
        self._queue.put_nowait(heap)

    @classmethod
    def get_instance(cls, multicast_group, port, loop=None):
        key = (multicast_group, port)
        if key not in cls._streams:
            logging.debug('Creating stream %s', key)
            cls._streams[key] = QueueStream(loop)
        else:
            logging.debug('Connecting to existing stream %s', key)
        return cls._streams[key]

    @classmethod
    def clear_instances(cls):
        cls._streams.clear()


class QueueRecvStream:
    """Replacement for :class:`spead2.recv.asyncio.Stream` that lets us
    feed in heaps directly."""
    def __init__(self,  *args, **kwargs):
        self._loop = kwargs.pop('loop', None)
        self._stream = None

    def add_udp_reader(self, multicast_group, port, *args, **kwargs):
        if self._stream is not None:
            raise RuntimeError('QueueRecvStream only supports one reader')
        self._stream = QueueStream.get_instance(multicast_group, port, self._loop)

    async def get(self):
        heap = await self._stream.get()
        return heap

    def stop(self):
        # Note: don't call stop on the stream, because that will cause the
        # next stream connected to the same endpoint to also be in a stopped
        # state.
        pass

    def set_memory_allocator(self, allocator):
        pass

    def set_memory_pool(self, memory_pool):
        pass

    def set_memcpy(self, id):
        pass


class TestReceiver(asynctest.TestCase):
    def setUp(self):
        patcher = mock.patch('spead2.recv.asyncio.Stream', QueueRecvStream)
        patcher.start()
        self.addCleanup(patcher.stop)

        self.n_streams = 2
        endpoints = katsdptelstate.endpoint.endpoint_list_parser(7148)(
            '239.0.0.1+{}'.format(self.n_streams - 1))
        self.n_xengs = 4
        sensors = mock.MagicMock()
        self.cbf_attr = fake_cbf_attr(4, self.n_xengs)
        self.n_chans = self.cbf_attr['n_chans']
        self.n_bls = len(self.cbf_attr['bls_ordering'])
        self.rx = Receiver(endpoints, '127.0.0.1', False, self.n_streams, 9200, 32 * 1024**2,
                           Range(0, self.n_chans), self.n_chans,
                           sensors, self.cbf_attr, active_frames=3, loop=self.loop)
        self.tx = [QueueStream.get_instance('239.0.0.{}'.format(i + 1), 7148, loop=self.loop)
                   for i in range(self.n_streams)]
        self.addCleanup(QueueStream.clear_instances)
        self.tx_ig = [spead2.send.ItemGroup() for tx in self.tx]
        for i, ig in enumerate(self.tx_ig):
            ig.add_item(0x1600, 'timestamp',
                        'Timestamp of start of this integration. '
                        'uint counting multiples of ADC samples since last sync '
                        '(sync_time, id=0x1027). Divide this number by timestamp_scale '
                        '(id=0x1046) to get back to seconds since last sync when this '
                        'integration was actually started. Note that the receiver will need '
                        'to figure out the centre timestamp of the accumulation '
                        '(eg, by adding half of int_time, id 0x1016).',
                        (), None, format=[('u', 48)])
            ig.add_item(0x4103, 'frequency',
                        'Identifies the first channel in the band of frequencies '
                        'in the SPEAD heap. Can be used to reconstruct the full spectrum.',
                        (), format=[('u', 48)])
            ig.add_item(0x1800, 'xeng_raw',
                        'Raw data stream from all the X-engines in the system. '
                        'For KAT-7, this item represents a full spectrum '
                        '(all frequency channels) assembled from lowest frequency '
                        'to highest frequency. Each frequency channel contains the data '
                        'for all baselines (n_bls given by SPEAD Id=0x1008). '
                        'Each value is a complex number - '
                        'two (real and imaginary) signed integers.',
                        (self.n_chans // self.n_xengs, self.n_bls, 2), np.int32)
        for i, tx in enumerate(self.tx):
            tx.send_heap(self.tx_ig[i].get_heap())

    async def test_stop(self):
        """The receiver must stop once all streams stop"""
        data_future = self.loop.create_task(self.rx.get())
        for i, tx in enumerate(self.tx):
            tx.send_heap(self.tx_ig[i].get_end())
        # Check that we get the end-of-stream notification; using a timeout
        # to ensure that we don't hang if the test fails.
        with assert_raises(spead2.Stopped):
            with async_timeout.timeout(30, loop=self.loop):
                await data_future

    def _make_data(self, n_frames):
        """Generates made-up timestamps and correlator data

        Parameters
        ----------
        n_frames : int
            Number of frames to generate

        Returns
        -------
        xeng_raw : np.ndarray
            5D array of integer correlator data, indexed by time, stream,
            channel, baseline, and real/complex
        indices : np.ndarray
            1D array of input dump indices
        timestamps : np.ndarray
            1D array of timestamps
        """
        xeng_raw = np.random.uniform(
            -1000, 1000,
            size=(n_frames, self.n_xengs, self.n_chans // self.n_xengs, self.n_bls, 2))
        xeng_raw = xeng_raw.astype(np.int32)
        interval = self.cbf_attr['ticks_between_spectra'] * self.cbf_attr['n_accs']
        indices = np.arange(n_frames, dtype=np.uint64)
        timestamps = indices * interval + 1234567890123
        return xeng_raw, indices, timestamps

    async def _send_in_order(self, xeng_raw, timestamps):
        for t in range(len(xeng_raw)):
            for i in range(self.n_xengs):
                stream_idx = i * self.n_streams // self.n_xengs
                self.tx_ig[stream_idx]['timestamp'].value = timestamps[t]
                self.tx_ig[stream_idx]['frequency'].value = i * self.n_chans // self.n_xengs
                self.tx_ig[stream_idx]['xeng_raw'].value = xeng_raw[t, i]
                self.tx[stream_idx].send_heap(self.tx_ig[stream_idx].get_heap())
            await asyncio.sleep(0.02, loop=self.loop)
        for i in range(self.n_streams):
            self.tx[i].send_heap(self.tx_ig[i].get_end())

    async def test_in_order(self):
        """Test normal case with data arriving in the expected order"""
        n_frames = 10
        xeng_raw, indices, timestamps = self._make_data(n_frames)
        send_future = asyncio.ensure_future(self._send_in_order(xeng_raw, timestamps),
                                            loop=self.loop)
        for t in range(n_frames):
            frame = await asyncio.wait_for(self.rx.get(), 3, loop=self.loop)
            assert_equal(indices[t], frame.idx)
            assert_equal(timestamps[t], frame.timestamp)
            assert_equal(self.n_xengs, len(frame.items))
            for i in range(self.n_xengs):
                np.testing.assert_equal(xeng_raw[t, i], frame.items[i])
        with assert_raises(spead2.Stopped):
            with async_timeout.timeout(3, loop=self.loop):
                await self.rx.get()
        await send_future

    async def _send_out_of_order(self, xeng_raw, timestamps):
        order = [
            # Send parts of frames 0, 1
            (0, 0), (0, 1), (0, 3),
            (1, 1), (1, 3), (1, 2),
            # Finish frame 1, start frame 2
            (1, 0), (2, 2),
            # Finish frame 0; frames 0, 1 should flush
            (0, 2),
            # Jump ahead by more than the window; frames 2-4 should be flushed/dropped
            (7, 0),
            # Finish off frame 2; should be discarded
            (2, 0), (2, 1), (2, 3),
            # Fill in a frame that's not at the start of the window
            (6, 0), (6, 1), (6, 2), (6, 3),
            # Force the window to advance, flushing 6
            (9, 0), (9, 2),
            # Fill in frame that's not at the start of the window; it should flush
            # when the stream stops
            (8, 0), (9, 3), (8, 1), (8, 3), (8, 2)
        ]
        for (t, i) in order:
            stream_idx = i * self.n_streams // self.n_xengs
            self.tx_ig[stream_idx]['timestamp'].value = timestamps[t]
            self.tx_ig[stream_idx]['frequency'].value = i * self.n_chans // self.n_xengs
            self.tx_ig[stream_idx]['xeng_raw'].value = xeng_raw[t, i]
            self.tx[stream_idx].send_heap(self.tx_ig[stream_idx].get_heap())
            # Longish sleep to ensure the ordering is respected
            await asyncio.sleep(0.02, loop=self.loop)
        for i in range(self.n_streams):
            self.tx[i].send_heap(self.tx_ig[i].get_end())

    async def test_out_of_order(self):
        """Test various edge behaviour for out-of-order data"""
        n_frames = 10
        xeng_raw, indices, timestamps = self._make_data(n_frames)
        send_future = self.loop.create_task(self._send_out_of_order(xeng_raw, timestamps))
        try:
            for t, missing in [(0, []), (1, []), (2, [0, 1, 3]), (6, []), (8, [])]:
                with async_timeout.timeout(3, loop=self.loop):
                    frame = await self.rx.get()
                assert_equal(indices[t], frame.idx)
                assert_equal(timestamps[t], frame.timestamp)
                assert_equal(self.n_xengs, len(frame.items))
                for i in range(self.n_xengs):
                    if i in missing:
                        assert_is_none(frame.items[i])
                    else:
                        np.testing.assert_equal(xeng_raw[t, i], frame.items[i])
            with assert_raises(spead2.Stopped):
                with async_timeout.timeout(3, loop=self.loop):
                    await self.rx.get()
        finally:
            await send_future
