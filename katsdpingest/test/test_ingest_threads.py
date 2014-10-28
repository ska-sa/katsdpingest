"""Tests for the ingest_threads module"""

import numpy as np
from katsdpingest import ingest_threads
from katsdpsigproc.test.test_accel import device_test, test_context, test_command_queue
import unittest

class TestCBFIngest(unittest.TestCase):
    @device_test
    def test_create_proc(self):
        """Test that an ingest processor can be created on the device"""
        template = ingest_threads.CBFIngest._create_proc_template(test_context)
        proc = template.instantiate(test_command_queue, 1024, (96, 1024 - 96), 544)

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
