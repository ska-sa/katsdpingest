"""Tests for the ingest_session module"""

from collections import OrderedDict

import numpy as np
import mock
from nose.tools import *

from katsdpsigproc.test.test_accel import device_test
from katsdptelstate import TelescopeState

from katsdpingest import ingest_session
from katsdpingest.utils import Range


def fake_cbf_attr(n_antennas, n_xengs=4):
    cbf_attr=dict(
        adc_sample_rate=1712000000.0,
        n_chans=4096,
        n_chans_per_substream=1024,
        n_accs=408 * 256,
        sync_time=1400000000.0
    )
    cbf_attr['bandwidth'] = cbf_attr['adc_sample_rate'] / 2
    cbf_attr['center_freq'] = cbf_attr['bandwidth'] * 3 / 2   # Reasonable for L band
    cbf_attr['scale_factor_timestamp'] = cbf_attr['adc_sample_rate']
    cbf_attr['ticks_between_spectra'] = 2 * cbf_attr['n_chans']
    cbf_attr['n_chans_per_substream'] = cbf_attr['n_chans'] // n_xengs
    cbf_attr['int_time'] = (cbf_attr['n_accs'] * cbf_attr['ticks_between_spectra']
                            / cbf_attr['adc_sample_rate'])
    bls_ordering = []
    antennas = ['m{:03}'.format(90 + i) for i in range(n_antennas)]
    # This ordering matches what's currently produced by CBF
    for ib, b in enumerate(antennas):
        for a in antennas[:ib+1]:
            bls_ordering.append((a + 'h', b + 'h'))
            bls_ordering.append((a + 'v', b + 'v'))
            bls_ordering.append((a + 'h', b + 'v'))
            bls_ordering.append((a + 'v', b + 'h'))
    cbf_attr['bls_ordering'] = np.array(bls_ordering)
    return cbf_attr


class TestGetCbfAttr(object):
    def setup(self):
        self.telstate = OrderedDict({
            'cbf_i0_bandwidth': 856000000.0,
            'cbf_i0_sync_time': 1234567890.0,
            'cbf_i0_adc_sample_rate': 1712000000.0,
            'cbf_i0_scale_factor_timestamp': 1712000000.0,
            'cbf_i0_antenna_channelised_voltage_instrument_dev_name': 'i0',
            'cbf_i0_antenna_channelised_voltage_n_chans': 262144,  # Different, to check precedence
            'cbf_i0_antenna_channelised_voltage_center_freq': 1284000000.0,
            'cbf_i0_antenna_channelised_voltage_ticks_between_spectra': 8192,
            'cbf_i1_baseline_correlation_products_src_streams': ['i0_antenna_channelised_voltage'],
            'cbf_i1_baseline_correlation_products_instrument_dev_name': 'i1',
            'cbf_i1_baseline_correlation_products_int_time': 0.499,
            'cbf_i1_baseline_correlation_products_n_chans': 4096,
            'cbf_i1_baseline_correlation_products_n_chans_per_substream': 256,
            'cbf_i1_baseline_correlation_products_n_accs': 104448,
            'cbf_i1_baseline_correlation_products_bls_ordering': [('m001h', 'm001h')]
        })
        self.expected = {
            'adc_sample_rate': 1712000000.0,
            'n_chans': 4096,
            'n_chans_per_substream': 256,
            'n_accs': 104448,
            'bls_ordering': [('m001h', 'm001h')],
            'bandwidth': 856000000.0,
            'center_freq': 1284000000.0,
            'sync_time': 1234567890.0,
            'int_time': 0.499,
            'scale_factor_timestamp': 1712000000.0,
            'ticks_between_spectra': 8192
        }

    def _collapse_name(self, name):
        name = name.replace('i1_baseline_correlation_products_', '')
        name = name.replace('i0_antenna_channelised_voltage_', '')
        name = name.replace('i0_', '')
        return name

    def test_named(self):
        attrs = ingest_session.get_cbf_attr(self.telstate, 'i1_baseline_correlation_products')
        assert_equal(self.expected, attrs)

    def test_compat(self):
        flat = {}
        # telstate needs to be an ordered dict so that conflicts are resolved
        # deterministically.
        for key, value in self.telstate.items():
            flat[self._collapse_name(key)] = value
        attrs = ingest_session.get_cbf_attr(flat, None)
        assert_equal(self.expected, attrs)


class TestTimeAverage(object):
    def setup(self):
        self.cbf_attr = fake_cbf_attr(1)
        self.cbf_attr['int_time'] = 0.75  # Change to a round number for sake of the test
        self.input_interval = self.cbf_attr['n_accs'] * self.cbf_attr['ticks_between_spectra']

    def test_constructor(self):
        avg = ingest_session._TimeAverage(self.cbf_attr, 2.0)
        assert_equal(2.25, avg.int_time)
        assert_equal(3, avg.ratio)
        assert_equal(avg.ratio * self.input_interval, avg.interval)
        assert_is(None, avg._start_ts)

    def make_ts(self, idx):
        if isinstance(idx, list):
            return [self.make_ts(x) for x in idx]
        else:
            return 100000000 + idx * self.input_interval

    def test_add_timestamp(self):
        avg = ingest_session._TimeAverage(self.cbf_attr, 2.0)
        avg.flush = mock.Mock(name='flush', spec_set=avg.flush)
        avg.add_timestamp(self.make_ts(0))
        avg.add_timestamp(self.make_ts(2))
        avg.add_timestamp(self.make_ts(1))  # Test time reordering
        assert not avg.flush.called

        avg.add_timestamp(self.make_ts(4))  # Skip first packet in the group
        avg.flush.assert_called_once_with(self.make_ts(1.5))
        avg.flush.reset_mock()
        assert_equal(self.make_ts(3), avg._start_ts)

        avg.add_timestamp(self.make_ts(12))  # Skip some whole groups
        avg.flush.assert_called_once_with(self.make_ts(4.5))
        avg.flush.reset_mock()
        assert_equal(self.make_ts(12), avg._start_ts)

        avg.finish()
        avg.flush.assert_called_once_with(self.make_ts(13.5))
        assert_is(None, avg._start_ts)

    def test_alignment(self):
        """Phase must be independent of first timestamp seen"""
        for i in range(5):
            avg = ingest_session._TimeAverage(self.cbf_attr, 2.0)
            ts = self.make_ts(i)
            avg.add_timestamp(ts)
            start_ts = avg._start_ts
            interval = avg.ratio * self.input_interval
            assert_less_equal(start_ts, ts)
            assert_less(ts - interval, start_ts)
            assert_equal(0, (self.make_ts(0) - start_ts) % interval)


def test_split_array():
    """Test _split_array"""
    c64 = (np.random.uniform(size=(4, 7)) +
           1j * np.random.uniform(size=(4, 7))).astype(np.complex64)
    # Create a view which is discontiguous
    src = c64[:3, :5].T
    actual = ingest_session._split_array(src, np.float32)
    expected = np.zeros((5, 3, 2), np.float32)
    for i in range(5):
        for j in range(3):
            expected[i, j, 0] = src[i, j].real
            expected[i, j, 1] = src[i, j].imag
    np.testing.assert_equal(actual, expected)


class TestTelstateReceiver(object):
    def setup(self):
        self.telstate = TelescopeState()
        self.telstate.clear()

    def test_first_timestamp(self):
        # We don't want to bother setting up a valid Receiver base class, we
        # just want to test the subclass, so we mock in a different base.
        class DummyBase(object):
            pass

        patcher = mock.patch.object(ingest_session.TelstateReceiver, '__bases__', (DummyBase,))
        with patcher:
            patcher.is_local = True   # otherwise mock tries to delete __bases__
            receiver = ingest_session.TelstateReceiver(telstate=self.telstate)
            # Set first value
            assert_equal(12345, receiver._first_timestamp(12345))
            # Try a different value, first value must stick
            assert_equal(12345, receiver._first_timestamp(54321))
            # Set same value
            assert_equal(12345, receiver._first_timestamp(12345))


class TestCBFIngest(object):
    @device_test
    def test_create_proc(self, context, queue):
        """Test that an ingest processor can be created on the device"""
        template = ingest_session.CBFIngest.create_proc_template(context, [4, 12], 4096, True, True)
        template.instantiate(
            queue, 1024, Range(96, 1024 - 96), 16, 544, 512,
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
        expected_baseline_inputs = np.array([
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
            [0, 2],
            [1, 3],
            [2, 0],
            [3, 1],
            [0, 1],
            [2, 3],
            [0, 3],
            [2, 1]
        ])

        bls = ingest_session.BaselineOrdering(orig_ordering)
        np.testing.assert_equal(expected_ordering, bls.sdp_bls_ordering)
        np.testing.assert_equal([2, 4, 0, 6, 9, 11, 10, 8, 5, 7, 1, 3], bls.permutation)
        np.testing.assert_equal([0, 1, 2, 3], bls.input_auto_baseline)
        np.testing.assert_equal(expected_baseline_inputs, bls.baseline_inputs)

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
        np.testing.assert_equal([0, 1], bls.input_auto_baseline)
        np.testing.assert_equal([[0, 0], [1, 1], [0, 1], [1, 0]], bls.baseline_inputs)
