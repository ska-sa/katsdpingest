"""Tests for the ingest_threads module"""

import numpy as np
from katsdpingest import ingest_threads
from katsdpsigproc.test.test_accel import device_test
import unittest
import mock
from nose.tools import *

class TestTimeAverage(object):
    def setup(self):
        self.cbf_attr = {
            'scale_factor_timestamp': 1712000000.0,
            'adc_sample_rate': 1712000000,
            'n_accs': 123456,
            'n_chans': 4096,
            'int_time': 0.75
        }
        self.input_interval = 123456 * 4096 * 2

    def test_constructor(self):
        avg = ingest_threads._TimeAverage(self.cbf_attr, 2.0)
        assert_equal(2.25, avg.int_time)
        assert_equal(3, avg.ratio)
        assert_equal(avg.ratio * self.input_interval, avg.interval)
        assert_is(None, avg._start_ts)
        assert_equal([], avg._ts)

    def make_ts(self, idx):
        if isinstance(idx, list):
            return [self.make_ts(x) for x in idx]
        else:
            return 1000000000 + idx * self.input_interval

    def test_add_timestamp(self):
        avg = ingest_threads._TimeAverage(self.cbf_attr, 2.0)
        avg.flush = mock.Mock(name='flush', spec_set=avg.flush)
        avg.add_timestamp(self.make_ts(0))
        avg.add_timestamp(self.make_ts(2))
        avg.add_timestamp(self.make_ts(1))  # Test time reordering
        assert not avg.flush.called

        avg.add_timestamp(self.make_ts(4))  # Skip first packet in the group
        avg.flush.assert_called_once_with(self.make_ts([0, 2, 1]))
        avg.flush.reset_mock()
        assert_equal(self.make_ts(3), avg._start_ts)
        assert_equal([self.make_ts(4)], avg._ts)

        avg.add_timestamp(self.make_ts(12))  # Skip some whole groups
        avg.flush.assert_called_once_with(self.make_ts([4]))
        avg.flush.reset_mock()
        assert_equal(self.make_ts(12), avg._start_ts)
        assert_equal([self.make_ts(12)], avg._ts)

        avg.finish()
        avg.flush.assert_called_once_with(self.make_ts([12]))
        assert_is(None, avg._start_ts)
        assert_equal([], avg._ts)

def test_split_array():
    """Test _split_array"""
    c64 = (np.random.uniform(size=(4, 7)) + 1j * np.random.uniform(size=(4,7))).astype(np.complex64)
    # Create a view which is discontiguous
    src = c64[:3, :5].T
    actual = ingest_threads._split_array(src, np.float32)
    expected = np.zeros((5, 3, 2), np.float32)
    for i in range(5):
        for j in range(3):
            expected[i, j, 0] = src[i, j].real
            expected[i, j, 1] = src[i, j].imag
    np.testing.assert_equal(actual, expected)

class TestCBFIngest(unittest.TestCase):
    @device_test
    def test_create_proc(self, context, queue):
        """Test that an ingest processor can be created on the device"""
        template = ingest_threads.CBFIngest.create_proc_template(context, 8, 4096)
        proc = template.instantiate(queue, 1024, (96, 1024 - 96), 544, 512,
                16, 64, [(0, 4), (500, 512)],
                threshold_args={'n_sigma': 11.0})

    def test_tune_next(self):
        assert_equal(2, ingest_threads.CBFIngest._tune_next(0, [2, 4, 8, 16]))
        assert_equal(8, ingest_threads.CBFIngest._tune_next(5, [2, 4, 8, 16]))
        assert_equal(8, ingest_threads.CBFIngest._tune_next(8, [2, 4, 8, 16]))
        assert_equal(21, ingest_threads.CBFIngest._tune_next(21, [2, 4, 8, 16]))

    def test_tune_next_antennas(self):
        assert_equal(2, ingest_threads.CBFIngest._tune_next_antennas(0))
        assert_equal(8, ingest_threads.CBFIngest._tune_next_antennas(5))
        assert_equal(8, ingest_threads.CBFIngest._tune_next_antennas(8))
        assert_equal(32, ingest_threads.CBFIngest._tune_next_antennas(21))

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

        permutation, new_ordering = ingest_threads.CBFIngest.baseline_permutation(orig_ordering)
        np.testing.assert_equal(expected_ordering, new_ordering)
        np.testing.assert_equal([2, 4, 0, 6, 9, 11, 10, 8, 5, 7, 1, 3], permutation)

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

        permutation, new_ordering = ingest_threads.CBFIngest.baseline_permutation(orig_ordering, antenna_mask)
        np.testing.assert_equal(expected_ordering, new_ordering)
        np.testing.assert_equal([-1, -1, -1, -1, -1, -1, -1, -1, 2, 3, 0, 1], permutation)
