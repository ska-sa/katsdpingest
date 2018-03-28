"""Tests for :mod:`katsdpingest.ingest_server`."""

from __future__ import print_function, absolute_import, division
import argparse
import functools
import logging
import mock
import copy
import numpy as np
from nose.tools import (assert_in, assert_is_not_none, assert_is_instance, assert_true,
                        assert_equal, assert_almost_equal, assert_regexp_matches,
                        nottest)
import tornado.gen
from tornado.platform.asyncio import AsyncIOMainLoop
import trollius
from trollius import From, Return

import spead2
import spead2.recv
import spead2.send
import katcp
import katsdptelstate
from katsdptelstate.endpoint import Endpoint
from katsdpsigproc.test.test_accel import device_test
import katsdpingest.sigproc
from katsdpingest.utils import Range
from katsdpingest.ingest_server import IngestDeviceServer
from katsdpingest.ingest_session import ChannelRanges, BaselineOrdering
from katsdpingest.test.test_ingest_session import fake_cbf_attr
from katsdpingest.receiver import Frame
from katsdpingest.sender import Data


STATIC_FLAG = 1 << katsdpingest.sigproc.IngestTemplate.flag_names.index('static')
CAM_FLAG = 1 << katsdpingest.sigproc.IngestTemplate.flag_names.index('cam')


class MockReceiver(object):
    """Replacement for :class:`katsdpingest.receiver.Receiver`.

    It has a predefined list of frames and yields them with no delay. However,
    one can request a pause prior to a particular frame.

    Parameters
    ----------
    data : ndarray
        3D array of visibilities indexed by time, frequency and baseline.
        The array contains data for the entire CBF channel range.
    timestamps : array-like
        1D array of CBF timestamps
    """
    def __init__(self, data, timestamps,
                 endpoints, interface_address, ibv,
                 max_streams, max_packet_size, buffer_size,
                 channel_range, cbf_channels, sensors,
                 cbf_attr, active_frames=2, loop=None, telstates=None,
                 l0_int_time=None, pauses=None):
        assert data.shape[0] == len(timestamps)
        self._next_frame = 0
        self._data = data
        self._timestamps = timestamps
        self._stop_event = trollius.Event()
        self._channel_range = channel_range
        self._substreams = len(channel_range) // cbf_attr['n_chans_per_substream']
        self._pauses = {} if pauses is None else pauses
        self._loop = loop
        # Set values to match Receiver
        self.cbf_attr = cbf_attr
        self.interval = cbf_attr['ticks_between_spectra'] * cbf_attr['n_accs']
        self.timestamp_base = timestamps[0]

    def stop(self):
        self._stop_event.set()

    @trollius.coroutine
    def join(self):
        yield From(self._stop_event.wait())

    @trollius.coroutine
    def get(self):
        event = self._pauses.get(self._next_frame)
        if event is None:
            event = trollius.sleep(0, loop=self._loop)
        yield From(event)
        if self._next_frame >= len(self._data):
            raise spead2.Stopped('end of frame list')
        frame = Frame(self._next_frame, self._timestamps[self._next_frame], self._substreams)
        item_channels = len(self._channel_range) // self._substreams
        for i in range(self._substreams):
            start = self._channel_range.start + i * item_channels
            stop = start + item_channels
            frame.items[i] = self._data[self._next_frame, start:stop, ...]
        self._next_frame += 1
        raise Return(frame)


class DeepCopyMock(mock.MagicMock):
    """Mock that takes deep copies of its arguments when called."""
    def __call__(self, *args, **kwargs):
        return super(DeepCopyMock, self).__call__(*copy.deepcopy(args), **copy.deepcopy(kwargs))


@nottest
def async_test(func):
    """Decorator to run a test inside the Tornado event loop"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return tornado.ioloop.IOLoop.current().run_sync(lambda: func(*args, **kwargs))
    return wrapper


def decode_heap(heap):
    """Converts a :class:`spead2.send.Heap` to a :class:`spead2.recv.Heap`.

    If the input heap contains a stop packet, returns ``None``.
    """
    out_stream = spead2.send.BytesStream(spead2.ThreadPool())
    out_stream.send_heap(heap)
    in_stream = spead2.recv.Stream(spead2.ThreadPool())
    in_stream.add_buffer_reader(out_stream.getvalue())
    try:
        heap = next(in_stream)
    except StopIteration:
        heap = None
    in_stream.stop()
    return heap


def decode_heap_ig(heap):
    ig = spead2.ItemGroup()
    heap = decode_heap(heap)
    assert_is_not_none(heap)
    ig.update(heap)
    return ig


def is_start(heap):
    heap = decode_heap(heap)
    return heap.is_start_of_stream()


def is_stop(heap):
    heap = decode_heap(heap)
    return heap is None


class TestIngestDeviceServer(object):
    """Tests for :class:`katsdpingest.ingest_server.IngestDeviceServer.

    This does not test all the intricacies of flagging, timeseries masking,
    lost data and so on. It is intended to check that the katcp commands
    function and that the correct channels are sent to the correct places.
    """
    def _patch(self, *args, **kwargs):
        self._patchers.append(mock.patch(*args, **kwargs))
        mock_obj = self._patchers[-1].start()
        return mock_obj

    def _get_tx(self, thread_pool, endpoints, interface_address, flavour,
                int_time, channel_range, channel0, all_channels, baselines):
        if endpoints == self.user_args.l0_spectral_spead[1:2]:
            return self._tx['spectral']
        elif endpoints == self.user_args.l0_continuum_spead[1:2]:
            return self._tx['continuum']
        else:
            raise KeyError('VisSenderSet created with unrecognised endpoints')

    def _get_sd_tx(self, thread_pool, host, port, config):
        done_future = trollius.Future()
        done_future.set_result(None)
        tx = mock.MagicMock()
        tx.async_send_heap.return_value = done_future
        tx.async_flush.return_value = done_future
        self._sd_tx[(host, port)] = tx
        return tx

    def _create_data(self):
        start_ts = 100000000
        interval = self.cbf_attr['n_accs'] * self.cbf_attr['ticks_between_spectra']
        n_dumps = 19
        shape = (n_dumps, self.cbf_attr['n_chans'], len(self.cbf_attr['bls_ordering']), 2)
        rs = np.random.RandomState(seed=1)
        data = (rs.standard_normal(shape) * 1000).astype(np.int32)
        # Make autocorrelations real, and also set a fixed value. This gives
        # all visibilities the same weight, making it easier to compute the
        # expected values.
        for i, (a, b) in enumerate(self.cbf_attr['bls_ordering']):
            if a == b:
                data[:, :, i, 0] = 1000
                data[:, :, i, 1] = 0
        timestamps = (np.arange(n_dumps) * interval + start_ts).astype(np.uint64)
        return data, timestamps

    def fake_channel_mask(self):
        channel_mask = np.zeros(self.cbf_attr['n_chans'], np.bool_)
        channel_mask[464] = True
        channel_mask[700:800] = True
        channel_mask[900] = True
        return channel_mask

    @device_test
    def setup(self, context, command_queue):
        done_future = trollius.Future()
        done_future.set_result(None)
        self._patchers = []
        self._telstate = katsdptelstate.TelescopeState()
        self._telstate.clear()   # Prevent state leaks from other tests
        self._ioloop = AsyncIOMainLoop()
        self._ioloop.install()
        n_xengs = 16
        self.user_args = user_args = argparse.Namespace(
            sdisp_spead=[Endpoint('127.0.0.2', 7149)],
            cbf_spead=[Endpoint('239.102.250.{}'.format(i), 7148) for i in range(n_xengs)],
            cbf_interface='dummyif1',
            cbf_ibv=False,
            cbf_name='i0_baseline_correlation_products',
            l0_spectral_spead=[Endpoint('239.102.251.{}'.format(i), 7148) for i in range(4)],
            l0_spectral_interface='dummyif2',
            l0_spectral_name='sdp_l0',
            l0_continuum_spead=[Endpoint('239.102.252.{}'.format(i), 7148) for i in range(4)],
            l0_continuum_interface='dummyif3',
            l0_continuum_name='sdp_l0_continuum',
            output_int_time=4.0,
            sd_int_time=4.0,
            antenna_mask=['m090', 'm091', 'm092'],
            output_channels=Range(464, 1744),
            sd_output_channels=Range(640, 1664),
            continuum_factor=16,
            sd_continuum_factor=128,
            guard_channels=64,
            input_streams=2,
            input_max_packet_size=9200,
            input_buffer=32*1024**2,
            sd_spead_rate=1000000000.0,
            excise=False,
            servers=4,
            server_id=2,
            host='localhost',
            port=7147,
            telstate=self._telstate,
            name='sdp.ingest.1'
        )
        self.cbf_attr = fake_cbf_attr(4, n_xengs=n_xengs)
        self.channel_mask = self.fake_channel_mask()
        # Put them in at the beginning of time, to ensure they apply to every dump
        self._telstate.add('i0_baseline_correlation_products_src_streams',
                           ['i0_antenna_channelised_voltage'], immutable=True)
        self._telstate.add('i0_antenna_channelised_voltage_instrument_dev_name',
                           'i0', immutable=True)
        self._telstate.add('i0_antenna_channelised_voltage_channel_mask', self.channel_mask, ts=0)
        self._telstate.add('m090_data_suspect', False, ts=0)
        self._telstate.add('m091_data_suspect', True, ts=0)
        self.channel_ranges = ChannelRanges(
            user_args.servers, user_args.server_id - 1,
            self.cbf_attr['n_chans'], user_args.continuum_factor, user_args.sd_continuum_factor,
            len(user_args.cbf_spead), 64,
            user_args.output_channels, user_args.sd_output_channels)

        self._data, self._timestamps = self._create_data()
        self._pauses = None
        self._Receiver = self._patch(
            'katsdpingest.ingest_session.TelstateReceiver',
            side_effect=lambda *args, **kwargs:
                MockReceiver(self._data, self._timestamps, *args, pauses=self._pauses, **kwargs))
        self._tx = {'continuum': mock.MagicMock(), 'spectral': mock.MagicMock()}
        for tx in self._tx.values():
            tx.start.return_value = done_future
            tx.stop.return_value = done_future
            tx.send = DeepCopyMock()
            tx.send.return_value = done_future
            tx.sub_channels = len(self.channel_ranges.output)
        self._tx['continuum'].sub_channels //= self.channel_ranges.cont_factor
        self._VisSenderSet = self._patch(
            'katsdpingest.sender.VisSenderSet', side_effect=self._get_tx)
        self._sd_tx = {}
        self._UdpStream = self._patch('spead2.send.trollius.UdpStream',
                                      side_effect=self._get_sd_tx)
        self._patch('katsdpservices.get_interface_address',
                    side_effect=lambda interface: '127.0.0.' + interface[-1])
        self._server = IngestDeviceServer(
            user_args, self.channel_ranges, self.cbf_attr, context,
            host=user_args.host, port=user_args.port)
        self._server.start()
        self._client = katcp.AsyncClient(user_args.host, user_args.port, timeout=15)
        self._client.set_ioloop(self._ioloop)
        self._client.start()
        self._ioloop.run_sync(self._client.until_protocol)

    @tornado.gen.coroutine
    def _teardown(self):
        self._client.disconnect()
        self._client.stop()
        self._server.stop()

    def teardown(self):
        self._ioloop.run_sync(self._teardown)
        for patcher in reversed(self._patchers):
            patcher.stop()
        tornado.ioloop.IOLoop.clear_instance()

    @tornado.gen.coroutine
    def make_request(self, name, *args):
        """Issue a request to the server, and check that the result is an ok.

        Parameters
        ----------
        name : str
            Request name
        args : list
            Arguments to the request

        Returns
        -------
        informs : list
            Informs returned with the reply
        """
        reply, informs = yield self._client.future_request(katcp.Message.request(name, *args))
        assert_true(reply.reply_ok(), str(reply))
        raise tornado.gen.Return(informs)

    @tornado.gen.coroutine
    def assert_request_fails(self, msg_re, name, *args):
        """Assert that a request fails, and test the error message against
        a regular expression."""
        reply, informs = yield self._client.future_request(katcp.Message.request(name, *args))
        assert_equal(2, len(reply.arguments))
        assert_equal('fail', reply.arguments[0])
        assert_regexp_matches(reply.arguments[1], msg_re)

    def _get_expected(self):
        """Return expected visibilities, flags and timestamps.

        The timestamps are in seconds since the sync time. The full CBF channel
        range is returned.
        """
        # Convert to complex64 from pairs of real and imag int
        vis = (self._data[..., 0] + self._data[..., 1] * 1j).astype(np.complex64)
        # Scaling
        vis /= self.cbf_attr['n_accs']
        # Time averaging
        time_ratio = int(np.round(self._telstate['sdp_l0_int_time'] / self.cbf_attr['int_time']))
        batch_edges = np.arange(0, vis.shape[0], time_ratio)
        batch_sizes = np.minimum(batch_edges + time_ratio, vis.shape[0]) - batch_edges
        vis = np.add.reduceat(vis, batch_edges, axis=0)
        vis /= batch_sizes[:, np.newaxis, np.newaxis]
        timestamps = self._timestamps[::time_ratio] / self.cbf_attr['scale_factor_timestamp'] \
            + 0.5 * self._telstate['sdp_l0_int_time']
        # Baseline permutation
        bls = BaselineOrdering(self.cbf_attr['bls_ordering'], self.user_args.antenna_mask)
        inv_permutation = np.empty(len(bls.sdp_bls_ordering), np.int)
        for i, p in enumerate(bls.permutation):
            if p != -1:
                inv_permutation[p] = i
        vis = vis[..., inv_permutation]
        # Sanity check that we've constructed inv_permutation correctly
        np.testing.assert_array_equal(
            self._telstate['sdp_l0_bls_ordering'],
            self.cbf_attr['bls_ordering'][inv_permutation])
        flags = np.empty(vis.shape, np.uint8)
        flags[:] = self.channel_mask[np.newaxis, :, np.newaxis] * np.uint8(STATIC_FLAG)
        for i, (a, b) in enumerate(bls.sdp_bls_ordering):
            if a.startswith('m091') or b.startswith('m091'):
                # data suspect sensor is True
                flags[:, :, i] |= CAM_FLAG
        return vis, flags, timestamps

    def _channel_average(self, vis, factor):
        return np.add.reduceat(vis, np.arange(0, vis.shape[1], factor), axis=1) / factor

    def _channel_average_flags(self, flags, factor):
        return np.bitwise_or.reduceat(flags, np.arange(0, flags.shape[1], factor), axis=1)

    def _check_output(self, tx, expected_vis, expected_flags, expected_ts, send_slice):
        """Checks that the visibilities and timestamps are correct."""
        tx.start.assert_called_once_with()
        tx.stop.assert_called_once_with()
        calls = tx.send.mock_calls
        assert_equal(len(expected_vis), len(calls))
        for i, (vis, flags, ts, call) in enumerate(
                zip(expected_vis, expected_flags, expected_ts, calls)):
            data, idx, ts_rel = call[1]
            assert_is_instance(data, Data)
            np.testing.assert_allclose(vis, data.vis[send_slice], rtol=1e-5, atol=1e-6)
            np.testing.assert_array_equal(flags, data.flags[send_slice])
            assert_equal(i, idx)
            assert_almost_equal(ts, ts_rel)

    def test_init_telstate(self):
        """Test the output metadata in telstate"""
        def get_ts(key):
            return self._telstate[prefix + '_' + key]

        bls_ordering = []
        for a in self.user_args.antenna_mask:
            for b in self.user_args.antenna_mask:
                if a <= b:
                    for ap in 'hv':
                        for bp in 'hv':
                            bls_ordering.append([a + ap, b + bp])
        bls_ordering.sort()
        for prefix in ['sdp_l0', 'sdp_l0_continuum']:
            factor = 1 if prefix == 'sdp_l0' else self.user_args.continuum_factor
            assert_equal(1280 // factor, get_ts('n_chans'))
            assert_equal(get_ts('n_chans') // 4, get_ts('n_chans_per_substream'))
            assert_equal(len(bls_ordering), get_ts('n_bls'))
            assert_equal(bls_ordering, sorted(get_ts('bls_ordering').tolist()))
            assert_equal(self.cbf_attr['sync_time'], get_ts('sync_time'))
            assert_equal(267500000.0, get_ts('bandwidth'))
            assert_equal(1086718750.0, get_ts('center_freq'))
            assert_equal(8 * self.cbf_attr['int_time'], get_ts('int_time'))
            assert_equal((464, 1744), get_ts('channel_range'))

    @async_test
    @tornado.gen.coroutine
    def test_capture(self):
        """Test the core data capture process."""
        yield self.make_request('capture-init', 'cb1')
        yield self.make_request('capture-done')
        l0_flavour = spead2.Flavour(4, 64, 48)
        l0_int_time = 8 * self.cbf_attr['int_time']
        expected_vis, expected_flags, expected_ts = self._get_expected()
        expected_output_vis = expected_vis[:, self.channel_ranges.output.asslice(), :]
        expected_output_flags = expected_flags[:, self.channel_ranges.output.asslice(), :]

        # This server sends channels 784:1104 to L0 and 896:1152 to sdisp.
        # Aligning to the sd_continuum_factor (128) gives computed = 768:1152.
        assert_equal(Range(784, 1104), self.channel_ranges.output)
        assert_equal(Range(896, 1152), self.channel_ranges.sd_output)
        send_range = Range(16, 336)
        self._VisSenderSet.assert_any_call(
            mock.ANY, self.user_args.l0_spectral_spead[1:2], '127.0.0.2',
            l0_flavour, l0_int_time, send_range, 320, 1280, 24)
        self._check_output(self._tx['spectral'], expected_output_vis, expected_output_flags,
                           expected_ts, send_range.asslice())
        self._tx['spectral'].stop.assert_called_once_with()

        send_range = Range(1, 21)
        self._VisSenderSet.assert_any_call(
            mock.ANY, self.user_args.l0_continuum_spead[1:2], '127.0.0.3',
            l0_flavour, l0_int_time, send_range, 20, 80, 24)
        self._check_output(
            self._tx['continuum'],
            self._channel_average(expected_output_vis, self.user_args.continuum_factor),
            self._channel_average_flags(expected_output_flags, self.user_args.continuum_factor),
            expected_ts, send_range.asslice())

        assert_equal([('127.0.0.2', 7149)], self._sd_tx.keys())
        sd_tx = self._sd_tx[('127.0.0.2', 7149)]
        expected_sd_vis = self._channel_average(
            expected_vis[:, self.channel_ranges.sd_output.asslice(), :],
            self.user_args.sd_continuum_factor)
        expected_sd_flags = self._channel_average_flags(
            expected_flags[:, self.channel_ranges.sd_output.asslice(), :],
            self.user_args.sd_continuum_factor)
        calls = sd_tx.async_send_heap.mock_calls
        # First heap should be start-of-stream marker
        assert_true(is_start(calls[0][1][0]))
        # Following heaps should contain averaged visibility data
        assert_equal(len(expected_sd_vis), len(calls) - 2)
        for i, call in enumerate(calls[1:-1]):
            ig = decode_heap_ig(call[1][0])
            vis = ig['sd_blmxdata'].value
            # Signal displays take complex values as pairs of floats; reconstitute them.
            vis = vis[..., 0] + 1j * vis[..., 1]
            flags = ig['sd_blmxflags'].value
            np.testing.assert_allclose(expected_sd_vis[i], vis, rtol=1e-5, atol=1e-6)
            np.testing.assert_array_equal(expected_sd_flags[i], flags)
        # Final call must send a stop
        assert_true(is_stop(calls[-1][1][0]))

    @async_test
    @tornado.gen.coroutine
    def test_done_when_not_capturing(self):
        """Calling capture-stop when not capturing fails"""
        yield self.assert_request_fails(r'No existing capture session', 'capture-done')

    @async_test
    @tornado.gen.coroutine
    def test_init_when_capturing(self):
        """Calling capture-init when capturing fails"""
        yield self.make_request('capture-init', 'cb1')
        yield self.assert_request_fails(r'Existing capture session found', 'capture-init', 'cb2')
        yield self.make_request('capture-done')

    @async_test
    @tornado.gen.coroutine
    def test_enable_disable_debug(self):
        """?enable-debug and ?disable-debug change the log level of session logger"""
        assert_equal(logging.NOTSET, logging.getLogger('katsdpingest.ingest_session').level)
        yield self.make_request('enable-debug')
        assert_equal(logging.DEBUG, logging.getLogger('katsdpingest.ingest_session').level)
        yield self.make_request('disable-debug')
        assert_equal(logging.NOTSET, logging.getLogger('katsdpingest.ingest_session').level)

    @async_test
    @tornado.gen.coroutine
    def test_add_sdisp_ip(self):
        """Add additional addresses with add-sdisp-ip."""
        yield self.make_request('add-sdisp-ip', '127.0.0.3:8000')
        yield self.make_request('add-sdisp-ip', '127.0.0.4')
        # A duplicate
        yield self.make_request('add-sdisp-ip', '127.0.0.3:8001')
        yield self.make_request('capture-init', 'cb1')
        yield self.make_request('capture-done')
        assert_equal([('127.0.0.2', 7149), ('127.0.0.3', 8000), ('127.0.0.4', 7149)],
                     list(sorted(self._sd_tx.keys())))
        # We won't check the contents, since that is tested elsewhere. Just
        # check that all the streams got the expected number of heaps.
        for tx in self._sd_tx.values():
            assert_equal(5, len(tx.async_send_heap.mock_calls))

    @async_test
    @tornado.gen.coroutine
    def test_drop_sdisp_ip_not_capturing(self):
        """Dropping a sdisp IP when not capturing sends no data at all."""
        yield self.make_request('drop-sdisp-ip', '127.0.0.2')
        yield self.make_request('capture-init', 'cb1')
        yield self.make_request('capture-done')
        sd_tx = self._sd_tx[('127.0.0.2', 7149)]
        sd_tx.async_send_heap.assert_not_called()

    @async_test
    @tornado.gen.coroutine
    def test_drop_sdisp_ip_capturing(self):
        """Dropping a sdisp IP when capturing sends a stop heap."""
        self._pauses = {10: trollius.Future()}
        yield self.make_request('capture-init', 'cb1')
        sd_tx = self._sd_tx[('127.0.0.2', 7149)]
        # Ensure the pause point gets reached, and wait for
        # the signal display data to be sent.
        for i in range(1000):
            if len(sd_tx.async_send_heap.mock_calls) >= 2:
                break
            yield tornado.gen.sleep(0.01)
        else:
            raise tornado.gen.TimeoutError(
                'Timed out waiting for signal display tx call to be made')
        yield self.make_request('drop-sdisp-ip', '127.0.0.2')
        self._pauses[10].set_result(None)
        yield self.make_request('capture-done')
        calls = sd_tx.async_send_heap.mock_calls
        assert_equal(3, len(calls))     # start, one data, and stop heaps
        assert_true(is_start(calls[0][1][0]))
        ig = decode_heap_ig(calls[1][1][0])
        assert_in('sd_blmxdata', ig)
        assert_true(is_stop(calls[2][1][0]))

    @async_test
    @tornado.gen.coroutine
    def test_drop_sdisp_ip_missing(self):
        """Dropping an unregistered IP address fails"""
        yield self.assert_request_fails('does not exist', 'drop-sdisp-ip', '127.0.0.3')

    @async_test
    @tornado.gen.coroutine
    def test_internal_log_level_query_all(self):
        """Test internal-log-level query with no parameters"""
        informs = yield self.make_request('internal-log-level')
        levels = {}
        for inform in informs:
            levels[inform.arguments[0]] = inform.arguments[1]
        # Check that some known logger appears in the list
        assert_in('katcp.server', levels)
        assert_equal('NOTSET', levels['katcp.server'])

    @async_test
    @tornado.gen.coroutine
    def test_internal_log_level_query_one(self):
        """Test internal-log-level query with one parameter"""
        informs = yield self.make_request('internal-log-level', 'katcp.server')
        assert_equal(1, len(informs))
        assert_equal(katcp.Message.inform('internal-log-level', 'katcp.server', 'NOTSET',
                                          mid=informs[0].mid),
                     informs[0])

    @async_test
    @tornado.gen.coroutine
    def test_internal_log_level_query_one_missing(self):
        """Querying internal-log-level with a non-existent logger fails"""
        self.assert_request_fails('Unknown logger', 'internal-log-level', 'notalogger')

    @async_test
    @tornado.gen.coroutine
    def test_internal_log_level_set(self):
        """Set a logger level via internal-log-level"""
        yield self.make_request('internal-log-level', 'katcp.server', 'INFO')
        assert_equal(logging.INFO, logging.getLogger('katcp.server').level)
        yield self.make_request('internal-log-level', 'katcp.server', 'NOTSET')
        assert_equal(logging.NOTSET, logging.getLogger('katcp.server').level)
        yield self.assert_request_fails(
            'Unknown log level', 'internal-log-level', 'katcp.server', 'DUMMY')
