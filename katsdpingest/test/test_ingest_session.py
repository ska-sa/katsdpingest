"""Tests for the ingest_session module"""

import logging
import re
from unittest import mock

import numpy as np
from nose.tools import (
    assert_equal, assert_is, assert_regex, assert_is_none, assert_is_not_none,
    assert_logs, assert_raises
)
import asynctest
from katsdpsigproc.test.test_accel import device_test
import katsdptelstate.aio
import katsdpmodels.fetch.aiohttp
import katsdpmodels.rfi_mask
import katsdpmodels.band_mask
import katpoint

from katsdpingest import ingest_session
from katsdpingest.utils import Range, cbf_telstate_view


def fake_cbf_attr(n_antennas, n_xengs=4):
    cbf_attr = dict(
        scale_factor_timestamp=1712000000.0,
        n_chans=4096,
        n_chans_per_substream=1024,
        n_accs=408 * 256,
        sync_time=1400000000.0
    )
    cbf_attr['bandwidth'] = cbf_attr['scale_factor_timestamp'] / 2
    cbf_attr['center_freq'] = cbf_attr['bandwidth'] * 3 / 2   # Reasonable for L band
    cbf_attr['ticks_between_spectra'] = 2 * cbf_attr['n_chans']
    cbf_attr['n_chans_per_substream'] = cbf_attr['n_chans'] // n_xengs
    cbf_attr['int_time'] = (cbf_attr['n_accs'] * cbf_attr['ticks_between_spectra']
                            / cbf_attr['scale_factor_timestamp'])
    bls_ordering = []
    input_labels = []
    antennas = ['m{:03}'.format(90 + i) for i in range(n_antennas)]
    for ib, b in enumerate(antennas):
        for a in antennas[:ib+1]:
            bls_ordering.append((a + 'h', b + 'h'))
            bls_ordering.append((a + 'v', b + 'v'))
            bls_ordering.append((a + 'h', b + 'v'))
            bls_ordering.append((a + 'v', b + 'h'))
        input_labels.append(b + 'h')
        input_labels.append(b + 'v')
    cbf_attr['bls_ordering'] = np.array(bls_ordering)
    cbf_attr['input_labels'] = input_labels
    return cbf_attr


class TestSystemAttrs(asynctest.TestCase):
    async def setUp(self):
        self.values = {
            'i0_bandwidth': 856000000.0,
            'i0_sync_time': 1234567890.0,
            'i0_scale_factor_timestamp': 1712000000.0,
            'i0_antenna_channelised_voltage_instrument_dev_name': 'i0',
            'i0_antenna_channelised_voltage_n_chans': 262144,  # Different, to check precedence
            'i0_antenna_channelised_voltage_center_freq': 1284000000.0,
            'i0_antenna_channelised_voltage_ticks_between_spectra': 8192,
            'i0_antenna_channelised_voltage_input_labels': ['m001h', 'm001v'],
            'i1_baseline_correlation_products_src_streams': ['i0_antenna_channelised_voltage'],
            'i1_baseline_correlation_products_instrument_dev_name': 'i1',
            'i1_baseline_correlation_products_int_time': 0.499,
            'i1_baseline_correlation_products_n_chans': 4096,
            'i1_baseline_correlation_products_n_chans_per_substream': 256,
            'i1_baseline_correlation_products_n_accs': 104448,
            'i1_baseline_correlation_products_bls_ordering': [('m001h', 'm001h')],
            'm001_observer': 'm001, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, 1.126 -171.761 1.0605 5868.979 5869.998, -0:42:08.0 0 0:01:44.0 0:01:11.9 -0:00:14.0 -0:00:21.0 -0:36:13.1 0:01:36.2, 1.14',  # noqa: E501
            'sdp_model_base_url': 'https://test.invalid/models/',
            'model_rfi_mask_fixed': 'rfi_mask/fixed/dummy.h5',
            'i0_antenna_channelised_voltage_model_band_mask_fixed': 'band_mask/fixed/dummy.h5'
        }
        self.telstate = katsdptelstate.aio.TelescopeState()
        for key, value in self.values.items():
            await self.telstate.set(key, value)
        self.expected_cbf_attr = {
            'n_chans': 4096,
            'n_chans_per_substream': 256,
            'n_accs': 104448,
            'bls_ordering': [('m001h', 'm001h')],
            'input_labels': ['m001h', 'm001v'],
            'bandwidth': 856000000.0,
            'center_freq': 1284000000.0,
            'sync_time': 1234567890.0,
            'int_time': 0.499,
            'scale_factor_timestamp': 1712000000.0,
            'ticks_between_spectra': 8192
        }

    async def test(self):
        telstate_cbf = await cbf_telstate_view(self.telstate,
                                               'i1_baseline_correlation_products')
        with asynctest.patch('katsdpmodels.fetch.aiohttp.Fetcher.get', autospec=True) as fetch:
            async with katsdpmodels.fetch.aiohttp.TelescopeStateFetcher(self.telstate) as fetcher:
                attrs = await ingest_session.SystemAttrs.create(
                    fetcher,
                    telstate_cbf,
                    ['m000', 'm001']
                )
                assert_equal(self.expected_cbf_attr, attrs.cbf_attr)
                assert_equal([katpoint.Antenna(self.values['m001_observer'])],
                             attrs.antennas)
                assert_is_not_none(attrs.rfi_mask_model)
                assert_is_not_none(attrs.band_mask_model)
                assert_equal(fetch.mock_calls, [
                    mock.call(mock.ANY, 'https://test.invalid/models/rfi_mask/fixed/dummy.h5',
                              katsdpmodels.rfi_mask.RFIMask),
                    mock.call(mock.ANY, 'https://test.invalid/models/band_mask/fixed/dummy.h5',
                              katsdpmodels.band_mask.BandMask)
                ])

    async def test_no_models(self):
        await self.telstate.delete('sdp_model_base_url')
        telstate_cbf = await cbf_telstate_view(self.telstate,
                                               'i1_baseline_correlation_products')
        async with katsdpmodels.fetch.aiohttp.TelescopeStateFetcher(self.telstate) as fetcher:
            with assert_logs(level=logging.WARNING) as cm:
                attrs = await ingest_session.SystemAttrs.create(
                    fetcher,
                    telstate_cbf,
                    ['m000', 'm001']
                )
        assert_regex(cm.output[0], re.compile('.*Failed to load rfi_mask model.*', re.M))
        assert_regex(cm.output[1], re.compile('.*Failed to load band_mask model.*', re.M))
        assert_is_none(attrs.rfi_mask_model)
        assert_is_none(attrs.band_mask_model)


class TestTimeAverage(asynctest.TestCase):
    def test_constructor(self):
        avg = ingest_session._TimeAverage(3, asynctest.CoroutineMock(name='flush'))
        assert_equal(3, avg.ratio)
        assert_is(None, avg._start_idx)

    async def test_add_index(self):
        avg = ingest_session._TimeAverage(3, asynctest.CoroutineMock(name='flush'))
        await avg.add_index(0)
        await avg.add_index(2)
        await avg.add_index(1)  # Test time reordering
        assert not avg.flush.called

        await avg.add_index(3)  # Skip first frame in the group
        avg.flush.assert_called_once_with(0)
        avg.flush.reset_mock()
        assert_equal(3, avg._start_idx)

        await avg.add_index(12)  # Skip some whole groups
        avg.flush.assert_called_once_with(1)
        avg.flush.reset_mock()
        assert_equal(12, avg._start_idx)

        await avg.finish()
        avg.flush.assert_called_once_with(4)
        assert_is(None, avg._start_idx)


def test_split_array():
    """Test _split_array"""
    c64 = (np.random.uniform(size=(4, 7))
           + 1j * np.random.uniform(size=(4, 7))).astype(np.complex64)
    # Create a view which is discontiguous
    src = c64[:3, :5].T
    actual = ingest_session._split_array(src, np.float32)
    expected = np.zeros((5, 3, 2), np.float32)
    for i in range(5):
        for j in range(3):
            expected[i, j, 0] = src[i, j].real
            expected[i, j, 1] = src[i, j].imag
    np.testing.assert_equal(actual, expected)


class TestTelstateReceiver(asynctest.TestCase):
    def setUp(self):
        self.telstate = katsdptelstate.aio.TelescopeState()

    async def test_first_timestamp(self):
        # We don't want to bother setting up a valid Receiver base class, we
        # just want to test the subclass, so we mock in a different base.
        class DummyBase:
            def __init__(self, cbf_attr):
                self.cbf_attr = cbf_attr

        patcher = mock.patch.object(ingest_session.TelstateReceiver, '__bases__', (DummyBase,))
        with patcher:
            patcher.is_local = True   # otherwise mock tries to delete __bases__
            cbf_attr = {'scale_factor_timestamp': 4.0}
            receiver = ingest_session.TelstateReceiver(cbf_attr=cbf_attr,
                                                       telstates=[self.telstate],
                                                       l0_int_time=3.0)
            # Set first value
            assert_equal(12345, await receiver._first_timestamp(12345))
            # Try a different value, first value must stick
            assert_equal(12345, await receiver._first_timestamp(54321))
            # Set same value
            assert_equal(12345, await receiver._first_timestamp(12345))
            # Check the telstate keys
            assert_equal(12345, await self.telstate['first_timestamp_adc'])
            assert_equal(3087.75, await self.telstate['first_timestamp'])


class TestSensorHistory:
    def setUp(self):
        self.sh = ingest_session.SensorHistory('test')

    def test_simple(self) -> None:
        self.sh.add(4.0, 'hello')
        self.sh.add(6.0, 'world')
        assert_equal(self.sh.get(4.0), 'hello')
        assert_equal(self.sh.get(5.0), 'hello')
        assert_equal(self.sh.get(6.0), 'world')
        assert_equal(self.sh.get(7.0), 'world')
        assert_equal(len(self.sh._data), 1, 'old data was not pruned')

    def test_query_empty(self) -> None:
        assert_is_none(self.sh.get(4.0))
        assert_equal(self.sh.get(5.0, 'default'), 'default')

    def test_query_before_first(self) -> None:
        self.sh.add(5.0, 'hello')
        assert_is_none(self.sh.get(4.0))

    def test_add_before_query(self) -> None:
        self.sh.get(5.0)
        with assert_logs(ingest_session.logger, logging.WARNING):
            self.sh.add(4.0, 'oops')
        assert_equal(self.sh.get(5.0), 'oops')

    def test_add_out_of_order(self) -> None:
        self.sh.add(5.0, 'first')
        with assert_logs(ingest_session.logger, logging.WARNING):
            self.sh.add(4.0, 'second')
        assert_is_none(self.sh.get(4))

    def test_replace_latest(self) -> None:
        self.sh.add(5.0, 'first')
        self.sh.add(5.0, 'second')
        assert_equal(self.sh.get(5.0), 'second')

    def test_query_out_of_order(self) -> None:
        self.sh.get(5.0)
        with assert_raises(ValueError):
            self.sh.get(4.0)


class TestCBFIngest:
    @device_test
    def test_create_proc(self, context, queue):
        """Test that an ingest processor can be created on the device"""
        template = ingest_session.CBFIngest.create_proc_template(context, [4, 12], 4096, True, True)
        template.instantiate(
            queue, 1024, Range(96, 1024 - 96), Range(96, 1024 - 96), 544, 512, 1,
            8, 16, [(0, 4), (500, 512)],
            threshold_args={'n_sigma': 11.0})

    def test_tune_next(self):
        assert_equal(2, ingest_session.CBFIngest._tune_next(0, [2, 4, 8, 16]))
        assert_equal(8, ingest_session.CBFIngest._tune_next(5, [2, 4, 8, 16]))
        assert_equal(8, ingest_session.CBFIngest._tune_next(8, [2, 4, 8, 16]))
        assert_equal(21, ingest_session.CBFIngest._tune_next(21, [2, 4, 8, 16]))

    def test_baseline_permutation(self):
        orig_ordering = np.array([
            ['m000v', 'm000v'],
            ['m000h', 'm000v'],
            ['m000h', 'm000h'],
            ['m000v', 'm000h'],
            ['m000v', 'm001v'],
            ['m000v', 'm001h'],
            ['m000h', 'm001v'],
            ['m000h', 'm001h'],
            ['m001h', 'm001v'],
            ['m001v', 'm001h'],
            ['m001h', 'm001h'],
            ['m001v', 'm001v']])
        expected_ordering = np.array([
            ['m000h', 'm000h'],
            ['m001h', 'm001h'],
            ['m000v', 'm000v'],
            ['m001v', 'm001v'],
            ['m000h', 'm000v'],
            ['m001h', 'm001v'],
            ['m000v', 'm000h'],
            ['m001v', 'm001h'],
            ['m000h', 'm001h'],
            ['m000v', 'm001v'],
            ['m000h', 'm001v'],
            ['m000v', 'm001h']])

        bls = ingest_session.BaselineOrdering(orig_ordering)
        np.testing.assert_equal(expected_ordering, bls.sdp_bls_ordering)
        np.testing.assert_equal([2, 4, 0, 6, 9, 11, 10, 8, 5, 7, 1, 3], bls.permutation)

    def test_baseline_permutation_masked(self):
        orig_ordering = np.array([
            ['m000v', 'm000v'],
            ['m000h', 'm000v'],
            ['m000h', 'm000h'],
            ['m000v', 'm000h'],
            ['m000v', 'm001v'],
            ['m000v', 'm001h'],
            ['m000h', 'm001v'],
            ['m000h', 'm001h'],
            ['m001h', 'm001v'],
            ['m001v', 'm001h'],
            ['m001h', 'm001h'],
            ['m001v', 'm001v']])
        expected_ordering = np.array([
            ['m001h', 'm001h'],
            ['m001v', 'm001v'],
            ['m001h', 'm001v'],
            ['m001v', 'm001h']])
        antenna_mask = set(['m001'])

        bls = ingest_session.BaselineOrdering(orig_ordering, antenna_mask)
        np.testing.assert_equal(expected_ordering, bls.sdp_bls_ordering)
        np.testing.assert_equal([-1, -1, -1, -1, -1, -1, -1, -1, 2, 3, 0, 1], bls.permutation)
