"""Tests for receiver module"""

from unittest import mock
import asyncio
from typing import Dict, Tuple     # noqa: F401

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
from katsdptelstate.endpoint import Endpoint
from nose.tools import assert_equal, assert_is_none, assert_raises


class TestReceiver(asynctest.TestCase):
    def setUp(self):
        self._streams = {}    # Dict[Endpoint, spead2.send.InprocStream]

        def add_udp_reader(rx, multicast_group, port, *args, **kwargs):
            endpoint = Endpoint(multicast_group, port)
            tx = self._streams[endpoint]
            rx.add_inproc_reader(tx.queue)

        patcher = mock.patch.object(
            spead2.recv.asyncio.Stream, 'add_udp_reader', add_udp_reader)
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
        tx_thread_pool = spead2.ThreadPool()
        self.tx = [spead2.send.InprocStream(tx_thread_pool, [spead2.InprocQueue()])
                   for endpoint in endpoints]
        self._streams = dict(zip(endpoints, self.tx))
        for tx in self.tx:
            # asyncio.iscoroutinefunction doesn't like pybind11 functions, so
            # we have to hide it inside a lambda.
            self.addCleanup(lambda: tx.queues[0].stop())
        self.rx = Receiver(endpoints, '127.0.0.1', False, self.n_streams, 32 * 1024**2,
                           Range(0, self.n_chans), self.n_chans,
                           sensors, self.cbf_attr, active_frames=3)
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
                        (self.n_chans // self.n_xengs, self.n_bls, 2), np.dtype('>i4'))
        for ig, tx in zip(self.tx_ig, self.tx):
            tx.send_heap(ig.get_heap())

    async def test_stop(self):
        """The receiver must stop once all streams stop"""
        data_future = self.loop.create_task(self.rx.get())
        for ig, tx in zip(self.tx_ig, self.tx):
            tx.send_heap(ig.get_end())
        # Check that we get the end-of-stream notification; using a timeout
        # to ensure that we don't hang if the test fails.
        with assert_raises(spead2.Stopped):
            with async_timeout.timeout(30):
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
        xeng_raw = xeng_raw.astype('>i4')
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
            await asyncio.sleep(0.02)
        for i in range(self.n_streams):
            self.tx[i].send_heap(self.tx_ig[i].get_end())

    async def test_in_order(self):
        """Test normal case with data arriving in the expected order"""
        n_frames = 10
        xeng_raw, indices, timestamps = self._make_data(n_frames)
        send_future = asyncio.ensure_future(self._send_in_order(xeng_raw, timestamps))
        for t in range(n_frames):
            frame = await asyncio.wait_for(self.rx.get(), 3)
            assert_equal(indices[t], frame.idx)
            assert_equal(timestamps[t], frame.timestamp)
            assert_equal(self.n_xengs, len(frame.items))
            for i in range(self.n_xengs):
                np.testing.assert_equal(xeng_raw[t, i], frame.items[i])
        with assert_raises(spead2.Stopped):
            with async_timeout.timeout(3):
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
            await asyncio.sleep(0.02)
        for i in range(self.n_streams):
            self.tx[i].send_heap(self.tx_ig[i].get_end())

    async def test_out_of_order(self):
        """Test various edge behaviour for out-of-order data"""
        n_frames = 10
        xeng_raw, indices, timestamps = self._make_data(n_frames)
        send_future = self.loop.create_task(self._send_out_of_order(xeng_raw, timestamps))
        try:
            for t, missing in [(0, []), (1, []), (2, [0, 1, 3]), (6, []),
                               (7, [1, 2, 3]), (8, []), (9, [1])]:
                with async_timeout.timeout(3):
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
                with async_timeout.timeout(3):
                    await self.rx.get()
        finally:
            await send_future
