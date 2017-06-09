"""Tests for receiver module"""

from __future__ import print_function, absolute_import, division
import numpy as np
import spead2
import spead2.send
import spead2.recv.trollius
import trollius
from trollius import From, Return
from katsdpingest.receiver import Receiver
from katsdpingest.sigproc import Range
import katsdptelstate.endpoint
from katsdpsigproc.test.test_resource import async_test
from nose.tools import *
import mock
import logging


class QueueStream(object):
    """A simulated SPEAD stream, stored in memory as a queue of heaps. A value
    of None indicates that the stream has been shut down (because putting an
    actual stop heap in the queue won't have the desired effect, as normally
    this is processed at a lower level).
    """
    _streams = {}

    def __init__(self, loop=None):
        self._queue = trollius.Queue(loop=loop)

    @trollius.coroutine
    def get(self):
        heap = yield From(self._queue.get())
        if heap is None:
            self._queue.put_nowait(None)
            raise spead2.Stopped()
        else:
            raise Return(heap)

    def stop(self):
        self._queue.put_nowait(None)

    def send_heap(self, heap):
        tp = spead2.ThreadPool()
        encoder = spead2.send.BytesStream(tp)
        encoder.send_heap(heap)
        raw = encoder.getvalue()
        decoder = spead2.recv.Stream(tp)
        decoder.add_buffer_reader(raw)
        try:
            heap = decoder.get()
            self._queue.put_nowait(heap)
        except spead2.Stopped:
            self.stop()

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


class QueueRecvStream(object):
    """Replacement for :class:`spead2.recv.trollius.Stream` that lets us
    feed in heaps directly."""
    def __init__(self,  *args, **kwargs):
        self._loop = kwargs.pop('loop', None)
        self._stream = None

    def add_udp_reader(self, multicast_group, port, *args, **kwargs):
        if self._stream is not None:
            raise RuntimeError('QueueRecvStream only supports one reader')
        self._stream = QueueStream.get_instance(multicast_group, port, self._loop)

    @trollius.coroutine
    def get(self):
        heap = yield From(self._stream.get())
        raise Return(heap)

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


class TestReceiver(object):
    def setup(self):
        self.patcher = mock.patch('spead2.recv.trollius.Stream', QueueRecvStream)
        self.patcher.start()
        self.loop = trollius.get_event_loop_policy().new_event_loop()
        endpoints = katsdptelstate.endpoint.endpoint_list_parser(7148)('239.0.0.1+1')
        self.n_streams = 2
        self.n_xengs = 4
        self.n_chans = 4096
        sensors = mock.MagicMock()
        self.adc_sample_rate = 1712000000
        self.n_accs = 256000
        self.n_ants = 4
        self.n_bls = self.n_ants * (self.n_ants + 1) * 2
        baselines = []
        for i in range(self.n_ants):
            for j in range(i, self.n_ants):
                for pol1 in ('v', 'h'):
                    for pol2 in ('v', 'h'):
                        name1 = 'm{:03}{}'.format(i, pol1)
                        name2 = 'm{:03}{}'.format(j, pol2)
                        baselines.append([name1, name2])
        baselines = np.array(baselines)
        cbf_attr = {
            'adc_sample_rate': self.adc_sample_rate,
            'bandwidth': self.adc_sample_rate / 2,
            'n_accs': self.n_accs,
            'n_ants': self.n_ants,
            'n_bls': self.n_bls,
            'n_chans': self.n_chans,
            'n_chans_per_substream': self.n_chans // self.n_xengs,
            'int_time': self.n_accs * self.n_chans * 2 / self.adc_sample_rate,
            'sync_time': 14000000.0,
            'scale_factor_timestamp': self.adc_sample_rate,
            'ticks_between_spectra': 2 * self.n_chans,
            'bls_ordering': baselines
        }
        self.rx = Receiver(endpoints, '127.0.0.1', False, Range(0, self.n_chans), self.n_chans,
                           sensors, cbf_attr, active_frames=3, loop=self.loop)
        self.tx = [QueueStream.get_instance('239.0.0.{}'.format(i + 1), 7148, loop=self.loop)
                   for i in range(self.n_streams)]
        self.tx_ig = [spead2.send.ItemGroup() for tx in self.tx]
        for i, ig in enumerate(self.tx_ig):
            ig.add_item(0x1600, 'timestamp', 'Timestamp of start of this integration. uint counting multiples of ADC samples since last sync (sync_time, id=0x1027). Divide this number by timestamp_scale (id=0x1046) to get back to seconds since last sync when this integration was actually started. Note that the receiver will need to figure out the centre timestamp of the accumulation (eg, by adding half of int_time, id 0x1016).',
                (), None, format=[('u', 48)])
            ig.add_item(0x4103, 'frequency', 'Identifies the first channel in the band of frequencies in the SPEAD heap. Can be used to reconstruct the full spectrum.',
                (), format=[('u', 48)])
            ig.add_item(0x1800, 'xeng_raw', 'Raw data stream from all the X-engines in the system. For KAT-7, this item represents a full spectrum (all frequency channels) assembled from lowest frequency to highest frequency. Each frequency channel contains the data for all baselines (n_bls given by SPEAD Id=0x1008). Each value is a complex number - two (real and imaginary) signed integers.',
                (self.n_chans // self.n_xengs, self.n_bls, 2), np.int32)
        for i, tx in enumerate(self.tx):
            tx.send_heap(self.tx_ig[i].get_heap())

    def teardown(self):
        self.patcher.stop()
        QueueStream.clear_instances()
        self.loop.close()

    @async_test
    def test_stop(self):
        """The receiver must stop once all streams stop"""
        data_future = trollius.async(self.rx.get(), loop=self.loop)
        for i, tx in enumerate(self.tx):
            tx.send_heap(self.tx_ig[i].get_end())
        # Check that we get the end-of-stream notification; using a timeout
        # to ensure that we don't hang if the test fails.
        with assert_raises(spead2.Stopped):
            yield From(trollius.wait_for(data_future, 30, loop=self.loop))

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
        timestamps : np.ndarray
            1D array of timestamps
        """
        xeng_raw = np.random.uniform(
            -1000, 1000,
            size=(n_frames, self.n_xengs, self.n_chans // self.n_xengs, self.n_bls, 2))
        xeng_raw = xeng_raw.astype(np.int32)
        interval = 2 * self.n_accs * self.n_chans
        timestamps = np.arange(n_frames, dtype=np.uint64) * interval + 1234567890123
        return xeng_raw, timestamps

    @trollius.coroutine
    def _send_in_order(self, xeng_raw, timestamps):
        for t in range(len(xeng_raw)):
            for i in range(self.n_xengs):
                stream_idx = i * self.n_streams // self.n_xengs
                self.tx_ig[stream_idx]['timestamp'].value = timestamps[t]
                self.tx_ig[stream_idx]['frequency'].value = i * self.n_chans // self.n_xengs
                self.tx_ig[stream_idx]['xeng_raw'].value = xeng_raw[t, i]
                self.tx[stream_idx].send_heap(self.tx_ig[stream_idx].get_heap())
            yield From(trollius.sleep(0.02, loop=self.loop))
        for i in range(self.n_streams):
            self.tx[i].send_heap(self.tx_ig[i].get_end())

    @async_test
    def test_in_order(self):
        """Test normal case with data arriving in the expected order"""
        n_frames = 10
        xeng_raw, timestamps = self._make_data(n_frames)
        send_future = trollius.async(self._send_in_order(xeng_raw, timestamps), loop=self.loop)
        for t in range(n_frames):
            frame = yield From(trollius.wait_for(self.rx.get(), 3, loop=self.loop))
            assert_equal(timestamps[t], frame.timestamp)
            assert_equal(self.n_xengs, len(frame.items))
            for i in range(self.n_xengs):
                np.testing.assert_equal(xeng_raw[t, i], frame.items[i])
        with assert_raises(spead2.Stopped):
            yield From(trollius.wait_for(self.rx.get(), 3, loop=self.loop))
        yield From(send_future)

    @trollius.coroutine
    def _send_out_of_order(self, xeng_raw, timestamps):
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
            yield From(trollius.sleep(0.02, loop=self.loop))
        for i in range(self.n_streams):
            self.tx[i].send_heap(self.tx_ig[i].get_end())

    @async_test
    def test_out_of_order(self):
        """Test various edge behaviour for out-of-order data"""
        n_frames = 10
        xeng_raw, timestamps = self._make_data(n_frames)
        send_future = trollius.async(self._send_out_of_order(xeng_raw, timestamps), loop=self.loop)
        try:
            for t, missing in [(0, []), (1, []), (2, [0, 1, 3]), (6, []), (8, [])]:
                frame = yield From(trollius.wait_for(self.rx.get(), 3, loop=self.loop))
                assert_equal(timestamps[t], frame.timestamp)
                assert_equal(self.n_xengs, len(frame.items))
                for i in range(self.n_xengs):
                    if i in missing:
                        assert_is_none(frame.items[i])
                    else:
                        np.testing.assert_equal(xeng_raw[t, i], frame.items[i])
            with assert_raises(spead2.Stopped):
                yield From(trollius.wait_for(self.rx.get(), 3, loop=self.loop))
        finally:
            yield From(send_future)
