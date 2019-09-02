"""Tests for the ingest_session module"""

import numpy as np
from unittest import mock

from nose.tools import assert_equal, assert_is
from katsdpsigproc.test.test_accel import device_test
from katsdptelstate import TelescopeState

from katsdpingest import ingest_session
from katsdpingest.utils import Range


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


class TestGetCbfAttr:
    def setup(self):
        values = {
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
        }
        self.telstate = TelescopeState()
        self.telstate.clear()
        for key, value in values.items():
            self.telstate[key] = value
        self.expected = {
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

    def test(self):
        attrs = ingest_session.get_cbf_attr(self.telstate, 'i1_baseline_correlation_products')
        assert_equal(self.expected, attrs)


class TestTimeAverage:
    def test_constructor(self):
        avg = ingest_session._TimeAverage(3)
        assert_equal(3, avg.ratio)
        assert_is(None, avg._start_idx)

    def test_add_index(self):
        avg = ingest_session._TimeAverage(3)
        avg.flush = mock.Mock(name='flush', spec_set=avg.flush)
        avg.add_index(0)
        avg.add_index(2)
        avg.add_index(1)  # Test time reordering
        assert not avg.flush.called

        avg.add_index(3)  # Skip first frame in the group
        avg.flush.assert_called_once_with(0)
        avg.flush.reset_mock()
        assert_equal(3, avg._start_idx)

        avg.add_index(12)  # Skip some whole groups
        avg.flush.assert_called_once_with(1)
        avg.flush.reset_mock()
        assert_equal(12, avg._start_idx)

        avg.finish()
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


class TestTelstateReceiver:
    def setup(self):
        self.telstate = TelescopeState()
        self.telstate.clear()

    def test_first_timestamp(self):
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
            assert_equal(12345, receiver._first_timestamp(12345))
            # Try a different value, first value must stick
            assert_equal(12345, receiver._first_timestamp(54321))
            # Set same value
            assert_equal(12345, receiver._first_timestamp(12345))
            # Check the telstate keys
            assert_equal(12345, self.telstate['first_timestamp_adc'])
            assert_equal(3087.75, self.telstate['first_timestamp'])


class TestCBFIngest:
    @device_test
    def test_create_proc(self, context, queue):
        """Test that an ingest processor can be created on the device"""
        template = ingest_session.CBFIngest.create_proc_template(context, [4, 12], 4096, True, True)
        template.instantiate(
            queue, 1024, Range(96, 1024 - 96), Range(96, 1024 - 96), 544, 512,
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
