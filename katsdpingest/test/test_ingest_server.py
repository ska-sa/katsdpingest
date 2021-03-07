"""Tests for :mod:`katsdpingest.ingest_server`."""

import argparse
import logging
import asyncio
import copy
import concurrent.futures
from unittest import mock
from typing import List, Dict, Any

import asynctest
import async_timeout
import numpy as np
from nose.tools import (assert_in, assert_is_not_none, assert_is_instance,
                        assert_true, assert_equal, assert_almost_equal, assert_raises_regex)

import spead2
import spead2.recv
import spead2.send
import aiokatcp
import katsdptelstate.aio.memory
from katsdptelstate.endpoint import Endpoint
from katsdpsigproc.test.test_accel import device_test
from katdal.flags import CAM, STATIC
import katsdpmodels.rfi_mask
import katsdpmodels.band_mask
import katpoint
import astropy.table
import astropy.units as u

from katsdpingest.utils import Range, cbf_telstate_view
from katsdpingest.ingest_server import IngestDeviceServer
from katsdpingest.ingest_session import ChannelRanges, BaselineOrdering, SystemAttrs
from katsdpingest.test.test_ingest_session import fake_cbf_attr
from katsdpingest.receiver import Frame
from katsdpingest.sender import Data


class MockReceiver:
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
                 cbf_attr, active_frames=2, telstates=None,
                 l0_int_time=None, pauses=None):
        assert data.shape[0] == len(timestamps)
        self._next_frame = 0
        self._data = data
        self._timestamps = timestamps
        self._stop_event = asyncio.Event()
        self._channel_range = channel_range
        self._substreams = len(channel_range) // cbf_attr['n_chans_per_substream']
        self._pauses = {} if pauses is None else pauses
        # Set values to match Receiver
        self.cbf_attr = cbf_attr
        self.interval = cbf_attr['ticks_between_spectra'] * cbf_attr['n_accs']
        self.timestamp_base = timestamps[0]

    def stop(self):
        self._stop_event.set()

    @asyncio.coroutine
    def join(self):
        yield from(self._stop_event.wait())

    @asyncio.coroutine
    def get(self):
        event = self._pauses.get(self._next_frame)
        if event is None:
            event = asyncio.sleep(0)
        yield from(event)
        if self._next_frame >= len(self._data):
            raise spead2.Stopped('end of frame list')
        frame = Frame(self._next_frame, self._timestamps[self._next_frame], self._substreams)
        item_channels = len(self._channel_range) // self._substreams
        for i in range(self._substreams):
            start = self._channel_range.start + i * item_channels
            stop = start + item_channels
            frame.items[i] = self._data[self._next_frame, start:stop, ...]
        self._next_frame += 1
        return frame


class DeepCopyMock(mock.MagicMock):
    """Mock that takes deep copies of its arguments when called."""

    def __call__(self, *args, **kwargs):
        return super().__call__(*copy.deepcopy(args), **copy.deepcopy(kwargs))


def decode_heap_ig(heap):
    ig = spead2.ItemGroup()
    assert_is_not_none(heap)
    ig.update(heap)
    return ig


def get_heaps(tx):
    rx = spead2.recv.Stream(
        spead2.ThreadPool(),
        spead2.recv.StreamConfig(stop_on_stop_item=False)
    )
    tx.queues[0].stop()
    rx.add_inproc_reader(tx.queues[0])
    return list(rx)


class TestIngestDeviceServer(asynctest.TestCase):
    """Tests for :class:`katsdpingest.ingest_server.IngestDeviceServer.

    This does not test all the intricacies of flagging, timeseries masking,
    lost data and so on. It is intended to check that the katcp commands
    function and that the correct channels are sent to the correct places.
    """

    def _patch(self, *args, **kwargs):
        patcher = mock.patch(*args, **kwargs)
        mock_obj = patcher.start()
        self.addCleanup(patcher.stop)
        return mock_obj

    def _get_tx(self, thread_pool, endpoints, interface_address, flavour,
                int_time, channel_range, channel0, all_channels, baselines):
        if endpoints == self.user_args.l0_spectral_spead[1:2]:
            return self._tx['spectral']
        elif endpoints == self.user_args.l0_continuum_spead[1:2]:
            return self._tx['continuum']
        else:
            raise KeyError('VisSenderSet created with unrecognised endpoints')

    def _get_sd_tx(self, thread_pool, endpoints, config):
        assert_equal(len(endpoints), 1)
        tx = spead2.send.asyncio.InprocStream(thread_pool, [spead2.InprocQueue()])
        self._sd_tx[Endpoint(*endpoints[0])] = tx
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

    def fake_channel_mask(self) -> np.ndarray:
        channel_mask = np.zeros((self.cbf_attr['n_chans']), np.bool_)
        channel_mask[704] = True
        channel_mask[750:800] = True
        channel_mask[900] = True
        return channel_mask

    def fake_rfi_mask_model(self) -> katsdpmodels.rfi_mask.RFIMask:
        # Channels 852:857 and 1024
        ranges = astropy.table.QTable(
            [[1034e6, 1070.0e6] * u.Hz,
             [1034.95e6, 1070.0e6] * u.Hz,
             [1500, np.inf] * u.m],
            names=('min_frequency', 'max_frequency', 'max_baseline')
        )
        return katsdpmodels.rfi_mask.RFIMaskRanges(ranges, False)

    def fake_band_mask_model(self) -> katsdpmodels.band_mask.BandMask:
        # Channels 820:840
        ranges = astropy.table.Table(
            [[0.2001], [0.2049]], names=('min_fraction', 'max_fraction')
        )
        return katsdpmodels.band_mask.BandMaskRanges(ranges)

    def fake_channel_data_suspect(self):
        bad = np.zeros(self.cbf_attr['n_chans'], np.bool_)
        bad[300] = True
        bad[650:750] = True
        return bad

    @device_test
    async def setUp(self, context, command_queue) -> None:
        done_future = asyncio.Future()     # type: asyncio.Future[None]
        done_future.set_result(None)
        self._patchers = []                              # type: List[Any]
        self._telstate = katsdptelstate.aio.TelescopeState()
        n_xengs = 16
        self.user_args = user_args = argparse.Namespace(
            sdisp_spead=[Endpoint('127.0.0.2', 7149)],
            sdisp_interface=None,
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
            antenna_mask=['m090', 'm091', 'm093'],
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
            use_data_suspect=True,
            servers=4,
            server_id=2,
            clock_ratio=1.0,
            host='127.0.0.1',
            port=7147,
            name='sdp.ingest.1'
        )
        self.loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=8))
        self.cbf_attr = fake_cbf_attr(4, n_xengs=n_xengs)
        # Put them in at the beginning of time, to ensure they apply to every dump
        await self._telstate.set('i0_baseline_correlation_products_src_streams',
                                 ['i0_antenna_channelised_voltage'])
        await self._telstate.set('i0_antenna_channelised_voltage_instrument_dev_name', 'i0')
        await self._telstate.add('i0_antenna_channelised_voltage_channel_mask',
                                 self.fake_channel_mask(), ts=0)
        await self._telstate.add('m090_data_suspect', False, ts=0)
        await self._telstate.add('m091_data_suspect', True, ts=0)
        input_data_suspect = np.zeros(len(self.cbf_attr['input_labels']), np.bool_)
        input_data_suspect[1] = True     # Corresponds to m090v
        await self._telstate.add('i0_antenna_channelised_voltage_input_data_suspect',
                                 input_data_suspect, ts=0)
        await self._telstate.add('i0_baseline_correlation_products_channel_data_suspect',
                                 self.fake_channel_data_suspect(), ts=0)
        # These correspond to three core and one outlying MeerKAT antennas,
        # so that baselines to m093 are long while the others are short.
        antennas = [
            katpoint.Antenna('m090, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, -8.258 -207.289 1.2075 5874.184 5875.444, -0:00:39.7 0 -0:04:04.4 -0:04:53.0 0:00:57.8 -0:00:13.9 0:13:45.2 0:00:59.8, 1.14'),     # noqa: E501
            katpoint.Antenna('m091, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, 1.126 -171.761 1.0605 5868.979 5869.998, -0:42:08.0 0 0:01:44.0 0:01:11.9 -0:00:14.0 -0:00:21.0 -0:36:13.1 0:01:36.2, 1.14'),      # noqa: E501
            katpoint.Antenna('m002, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, -32.1085 -224.2365 1.248 5871.207 5872.205, 0:40:20.2 0 -0:02:41.9 -0:03:46.8 0:00:09.4 -0:00:01.1 0:03:04.7, 1.14'),              # noqa: E501
            katpoint.Antenna('m093, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, -1440.6235 -2503.7705 14.288 5932.94 5934.732, -0:15:23.0 0 0:00:04.6 -0:03:30.4 0:01:12.2 0:00:37.5 0:00:15.6 0:01:11.8, 1.14')   # noqa: E501
        ]
        self._telstate_cbf = await cbf_telstate_view(self._telstate,
                                                     'i0_baseline_correlation_products')
        self.system_attrs = SystemAttrs(
            self.cbf_attr, self.fake_rfi_mask_model(), self.fake_band_mask_model(),
            antennas)
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
                MockReceiver(self._data, self._timestamps, *args,     # type: ignore
                             pauses=self._pauses, **kwargs))
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
        self._sd_tx: Dict[Endpoint, spead2.send.asyncio.InprocStream] = {}
        self._UdpStream = self._patch('spead2.send.asyncio.UdpStream',
                                      side_effect=self._get_sd_tx)
        self._patch('katsdpservices.get_interface_address',
                    side_effect=lambda interface: '127.0.0.' + interface[-1] if interface else None)
        self._server = IngestDeviceServer(
            user_args, self._telstate_cbf, self.channel_ranges, self.system_attrs, context,
            host=user_args.host, port=user_args.port)
        await self._server.start()
        self.addCleanup(self._server.stop)
        self._client = await aiokatcp.Client.connect(user_args.host, user_args.port)
        self.addCleanup(self._client.wait_closed)
        self.addCleanup(self._client.close)

    async def make_request(self, name: str, *args) -> List[aiokatcp.Message]:
        """Issue a request to the server, timing out if it takes too long.

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
        with async_timeout.timeout(15):
            reply, informs = await self._client.request(name, *args)
        return informs

    async def assert_request_fails(self, msg_re, name, *args):
        """Assert that a request fails, and test the error message against
        a regular expression."""
        with assert_raises_regex(aiokatcp.FailReply, msg_re):
            with async_timeout.timeout(15):
                await self._client.request(name, *args)

    async def _get_expected(self):
        """Return expected visibilities, flags and timestamps.

        The timestamps are in seconds since the sync time. The full CBF channel
        range is returned.
        """
        # Convert to complex64 from pairs of real and imag int
        vis = (self._data[..., 0] + self._data[..., 1] * 1j).astype(np.complex64)
        # Scaling
        vis /= self.cbf_attr['n_accs']
        # Time averaging
        time_ratio = int(np.round(await self._telstate['sdp_l0_int_time']
                                  / self.cbf_attr['int_time']))
        batch_edges = np.arange(0, vis.shape[0], time_ratio)
        batch_sizes = np.minimum(batch_edges + time_ratio, vis.shape[0]) - batch_edges
        vis = np.add.reduceat(vis, batch_edges, axis=0)
        vis /= batch_sizes[:, np.newaxis, np.newaxis]
        timestamps = self._timestamps[::time_ratio] / self.cbf_attr['scale_factor_timestamp'] \
            + 0.5 * (await self._telstate['sdp_l0_int_time'])
        # Baseline permutation
        bls = BaselineOrdering(self.cbf_attr['bls_ordering'], self.user_args.antenna_mask)
        inv_permutation = np.empty(len(bls.sdp_bls_ordering), np.int)
        for i, p in enumerate(bls.permutation):
            if p != -1:
                inv_permutation[p] = i
        vis = vis[..., inv_permutation]
        # Sanity check that we've constructed inv_permutation correctly
        np.testing.assert_array_equal(
            await self._telstate['sdp_l0_bls_ordering'],
            self.cbf_attr['bls_ordering'][inv_permutation])
        flags = np.empty(vis.shape, np.uint8)
        channel_mask = self.fake_channel_mask()
        channel_mask[820:840] = True     # Merge in band mask
        channel_data_suspect = self.fake_channel_data_suspect()[np.newaxis, :, np.newaxis]
        flags[:] = channel_data_suspect * np.uint8(CAM)
        for i, (a, b) in enumerate(bls.sdp_bls_ordering):
            if a.startswith('m091') or b.startswith('m091'):
                # data suspect sensor is True
                flags[:, :, i] |= CAM
            if a == 'm090v' or b == 'm090v':
                # input_data_suspect is True
                flags[:, :, i] |= CAM
            flags[:, :, i] |= channel_mask * np.uint8(STATIC)
            if a[:-1] != b[:-1]:
                # RFI model, which doesn't apply to auto-correlations
                flags[:, 1024, i] |= np.uint8(STATIC)
                if a.startswith('m093') == b.startswith('m093'):
                    # Short baseline
                    flags[:, 852:857, i] |= np.uint8(STATIC)
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

    async def test_init_telstate(self):
        """Test the output metadata in telstate"""
        async def get_ts(key):
            return await self._telstate[prefix + '_' + key]

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
            assert_equal(1280 // factor, await get_ts('n_chans'))
            assert_equal((await get_ts('n_chans')) // 4, await get_ts('n_chans_per_substream'))
            assert_equal(len(bls_ordering), await get_ts('n_bls'))
            assert_equal(bls_ordering, sorted((await get_ts('bls_ordering')).tolist()))
            assert_equal(self.cbf_attr['sync_time'], await get_ts('sync_time'))
            assert_equal(267500000.0, await get_ts('bandwidth'))
            assert_equal(8 * self.cbf_attr['int_time'], await get_ts('int_time'))
            assert_equal((464, 1744), await get_ts('channel_range'))
        assert_equal(1086718750.0, await self._telstate['sdp_l0_center_freq'])
        # Offset by 7.5 channels to identify the centre of a continuum channel
        assert_equal(1088286132.8125, await self._telstate['sdp_l0_continuum_center_freq'])

    async def test_capture(self):
        """Test the core data capture process."""
        await self.make_request('capture-init', 'cb1')
        await self.make_request('capture-done')
        l0_flavour = spead2.Flavour(4, 64, 48)
        l0_int_time = 8 * self.cbf_attr['int_time']
        expected_vis, expected_flags, expected_ts = await self._get_expected()
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

        assert_equal([Endpoint('127.0.0.2', 7149)], list(self._sd_tx.keys()))
        sd_tx = self._sd_tx[Endpoint('127.0.0.2', 7149)]
        expected_sd_vis = self._channel_average(
            expected_vis[:, self.channel_ranges.sd_output.asslice(), :],
            self.user_args.sd_continuum_factor)
        expected_sd_flags = self._channel_average_flags(
            expected_flags[:, self.channel_ranges.sd_output.asslice(), :],
            self.user_args.sd_continuum_factor)
        heaps = get_heaps(sd_tx)
        # First heap should be start-of-stream marker
        assert_true(heaps[0].is_start_of_stream())
        # Following heaps should contain averaged visibility data
        assert_equal(len(expected_sd_vis), len(heaps) - 2)
        for i, heap in enumerate(heaps[1:-1]):
            ig = decode_heap_ig(heap)
            vis = ig['sd_blmxdata'].value
            # Signal displays take complex values as pairs of floats; reconstitute them.
            vis = vis[..., 0] + 1j * vis[..., 1]
            flags = ig['sd_blmxflags'].value
            np.testing.assert_allclose(expected_sd_vis[i], vis, rtol=1e-5, atol=1e-6)
            np.testing.assert_array_equal(expected_sd_flags[i], flags)
        # Final call must send a stop
        assert_true(heaps[-1].is_end_of_stream())

    async def test_done_when_not_capturing(self):
        """Calling capture-stop when not capturing fails"""
        await self.assert_request_fails(r'No existing capture session', 'capture-done')

    async def test_init_when_capturing(self):
        """Calling capture-init when capturing fails"""
        await self.make_request('capture-init', 'cb1')
        await self.assert_request_fails(r'Existing capture session found', 'capture-init', 'cb2')
        await self.make_request('capture-done')

    async def test_enable_disable_debug(self):
        """?enable-debug and ?disable-debug change the log level of session logger"""
        assert_equal(logging.NOTSET, logging.getLogger('katsdpingest.ingest_session').level)
        await self.make_request('enable-debug')
        assert_equal(logging.DEBUG, logging.getLogger('katsdpingest.ingest_session').level)
        await self.make_request('disable-debug')
        assert_equal(logging.NOTSET, logging.getLogger('katsdpingest.ingest_session').level)

    async def test_add_sdisp_ip(self):
        """Add additional addresses with add-sdisp-ip."""
        await self.make_request('add-sdisp-ip', '127.0.0.3:8000')
        await self.make_request('add-sdisp-ip', '127.0.0.4')
        # A duplicate
        await self.make_request('add-sdisp-ip', '127.0.0.3:8001')
        await self.make_request('capture-init', 'cb1')
        await self.make_request('capture-done')
        assert_equal({Endpoint('127.0.0.2', 7149),
                      Endpoint('127.0.0.3', 8000),
                      Endpoint('127.0.0.4', 7149)},
                     self._sd_tx.keys())
        # We won't check the contents, since that is tested elsewhere. Just
        # check that all the streams got the expected number of heaps.
        for tx in self._sd_tx.values():
            assert_equal(5, len(get_heaps(tx)))

    async def test_drop_sdisp_ip_not_capturing(self):
        """Dropping a sdisp IP when not capturing sends no data at all."""
        await self.make_request('drop-sdisp-ip', '127.0.0.2')
        await self.make_request('capture-init', 'cb1')
        await self.make_request('capture-done')
        sd_tx = self._sd_tx[Endpoint('127.0.0.2', 7149)]
        assert_equal([], get_heaps(sd_tx))

    async def test_drop_sdisp_ip_capturing(self):
        """Dropping a sdisp IP when capturing sends a stop heap."""
        self._pauses = {10: asyncio.Future()}
        await self.make_request('capture-init', 'cb1')
        sd_tx = self._sd_tx[Endpoint('127.0.0.2', 7149)]
        # Ensure the pause point gets reached, and wait for
        # the signal display data to be sent.
        sd_rx = spead2.recv.asyncio.Stream(
            spead2.ThreadPool(),
            spead2.recv.StreamConfig(stop_on_stop_item=False)
        )
        sd_rx.add_inproc_reader(sd_tx.queues[0])
        heaps = []
        with async_timeout.timeout(10):
            for i in range(2):
                heaps.append(await sd_rx.get())
        await self.make_request('drop-sdisp-ip', '127.0.0.2')
        self._pauses[10].set_result(None)
        await self.make_request('capture-done')
        sd_tx.queues[0].stop()
        while True:
            try:
                heaps.append(await sd_rx.get())
            except spead2.Stopped:
                break
        assert_equal(3, len(heaps))     # start, one data, and stop heaps
        assert_true(heaps[0].is_start_of_stream())
        ig = decode_heap_ig(heaps[1])
        assert_in('sd_blmxdata', ig)
        assert_true(heaps[2].is_end_of_stream())

    async def test_drop_sdisp_ip_missing(self):
        """Dropping an unregistered IP address fails"""
        await self.assert_request_fails('does not exist', 'drop-sdisp-ip', '127.0.0.3')

    async def test_internal_log_level_query_all(self):
        """Test internal-log-level query with no parameters"""
        informs = await self.make_request('internal-log-level')
        levels = {}
        for inform in informs:
            levels[inform.arguments[0]] = inform.arguments[1]
        # Check that some known logger appears in the list
        assert_in(b'aiokatcp.connection', levels)
        assert_equal(b'NOTSET', levels[b'aiokatcp.connection'])

    async def test_internal_log_level_query_one(self):
        """Test internal-log-level query with one parameter"""
        informs = await self.make_request('internal-log-level', 'aiokatcp.connection')
        assert_equal(1, len(informs))
        assert_equal(aiokatcp.Message.inform('internal-log-level', b'aiokatcp.connection',
                                             b'NOTSET', mid=informs[0].mid),
                     informs[0])

    async def test_internal_log_level_query_one_missing(self):
        """Querying internal-log-level with a non-existent logger fails"""
        await self.assert_request_fails('Unknown logger', 'internal-log-level', 'notalogger')

    async def test_internal_log_level_set(self):
        """Set a logger level via internal-log-level"""
        await self.make_request('internal-log-level', 'katcp.server', 'INFO')
        assert_equal(logging.INFO, logging.getLogger('katcp.server').level)
        await self.make_request('internal-log-level', 'katcp.server', 'NOTSET')
        assert_equal(logging.NOTSET, logging.getLogger('katcp.server').level)
        await self.assert_request_fails(
            'Unknown log level', 'internal-log-level', 'katcp.server', 'DUMMY')
